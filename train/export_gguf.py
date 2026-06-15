"""Merge LoRA adapter into base model and export to GGUF for Ollama.

    python -m train.export_gguf [--adapter train/checkpoints/adapter] [--out serve/] [--quant q4_k_m]

Unsloth handles llama.cpp quantization internally — no separate install needed.
The GGUF file is written to --out; update serve/Modelfile to point at it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def export(adapter_path: Path, out_dir: Path, quant: str) -> Path:
    from unsloth import FastLanguageModel

    print(f"Loading adapter from {adapter_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting GGUF ({quant}) → {out_dir} ...")
    model.save_pretrained_gguf(str(out_dir), tokenizer, quantization_method=quant)

    # Unsloth names the file <dir>/<model>-<quant>.gguf; find it.
    gguf_files = sorted(out_dir.glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(f"No GGUF file found in {out_dir} after export.")
    gguf_path = gguf_files[-1]
    size_mb = gguf_path.stat().st_size / 1_048_576
    print(f"GGUF ready: {gguf_path} ({size_mb:.0f} MB)")
    return gguf_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.export_gguf")
    ap.add_argument("--adapter", default="train/checkpoints/adapter")
    ap.add_argument("--out", default="serve/")
    ap.add_argument("--quant", default="q4_k_m", help="llama.cpp quantization method")
    args = ap.parse_args(argv)

    adapter_path = Path(args.adapter)
    out_dir = Path(args.out)

    if not adapter_path.exists():
        raise SystemExit(f"Adapter not found: {adapter_path}. Run train.train_qlora first.")

    gguf_path = export(adapter_path, out_dir, args.quant)

    provenance = {
        "git_sha": _git_sha(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "adapter": str(adapter_path),
        "gguf": str(gguf_path),
        "quant": args.quant,
        "size_mb": round(gguf_path.stat().st_size / 1_048_576, 1),
    }
    (out_dir / "last_export.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
