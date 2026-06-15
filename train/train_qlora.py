"""QLoRA fine-tune Qwen2.5-Coder-7B-Instruct on curated note → FHIR pairs.

    python -m train.train_qlora [--data data/curated] [--config train/configs/default.yaml] [--out train/checkpoints]

Unsloth + TRL SFTTrainer; W&B optional (set WANDB_API_KEY to enable).
Loss is computed only on the assistant response (not the instruction/system).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_dataset(data_dir: Path, tokenizer):
    from datasets import load_dataset
    from evals.case import Case
    from train.prompt import format_prompt

    def _to_text(record: dict) -> dict:
        try:
            case = Case.model_validate(record["case"])
            if case.gold is None:
                return {"text": ""}
            return {"text": format_prompt(case, case.gold)}
        except Exception:
            return {"text": ""}

    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="train")

    train_ds = train_ds.map(_to_text, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_to_text, remove_columns=val_ds.column_names)

    train_ds = train_ds.filter(lambda x: len(x["text"]) > 10)
    val_ds = val_ds.filter(lambda x: len(x["text"]) > 10)

    return train_ds, val_ds


def train(cfg: dict[str, Any], data_dir: Path, out_dir: Path) -> dict[str, Any]:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model: {cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    print("Formatting dataset...")
    train_ds, val_ds = _build_dataset(data_dir, tokenizer)
    print(f"  train={len(train_ds)}, val={len(val_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError(f"No training examples found in {data_dir}/train.jsonl. Run data.build first.")

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=cfg["num_train_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            learning_rate=cfg["learning_rate"],
            fp16=False,
            bf16=False,
            logging_steps=5,
            eval_steps=max(10, len(train_ds) // 4),
            save_steps=max(10, len(train_ds) // 2),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=cfg["seed"],
            report_to=report_to,
            dataset_text_field="text",
            packing=False,
            max_seq_length=cfg["max_seq_length"],
        ),
    )

    # Mask loss on the instruction/system — only train on the assistant response.
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("Masking: training on assistant responses only.")
    except Exception as e:
        print(f"Response-only masking unavailable ({e}); training on full sequence.")

    print("Training...")
    train_result = trainer.train()
    metrics = train_result.metrics if hasattr(train_result, "metrics") else {}

    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved → {adapter_path}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.train_qlora")
    ap.add_argument("--data", default="data/curated", help="directory with train.jsonl, val.jsonl")
    ap.add_argument("--config", default="train/configs/default.yaml")
    ap.add_argument("--out", default=None, help="output directory (overrides config output_dir)")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    data_dir = Path(args.data)
    out_dir = Path(args.out or cfg.get("output_dir", "train/checkpoints"))

    sha, ts = _git_sha(), datetime.now(timezone.utc).isoformat()
    metrics = train(cfg, data_dir, out_dir)

    provenance = {
        "git_sha": sha, "ts": ts, "config": cfg,
        "data_dir": str(data_dir), "out_dir": str(out_dir),
        "metrics": metrics,
    }
    (out_dir / "last_run.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"status": "done", "adapter": str(out_dir / "adapter"), **metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
