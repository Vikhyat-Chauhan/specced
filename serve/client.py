"""Inference client: note + target_resources → Prediction.

Two backends, same interface:
  "ollama"  — POST to Ollama HTTP API; requires `ollama serve` + SPECCED_MODEL loaded.
  "hf"      — Direct Unsloth/HuggingFace inference; no Ollama needed.

The HFBackend is a context manager that loads the model on entry and frees GPU
memory on exit, so two passes (base then fine-tuned) can run without OOM:

    with HFBackend() as base:
        results = [base.extract(note, trs) for note, trs in cases]
    with HFBackend("train/checkpoints/adapter") as ft:
        results = [ft.extract(note, trs) for note, trs in cases]

Environment variables (see .env.example):
  OLLAMA_HOST   — default http://localhost:11434
  SPECCED_MODEL — Ollama model name, default specced-qwen7b
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from evals.case import Prediction

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SPECCED_MODEL = os.environ.get("SPECCED_MODEL", "specced-qwen7b")
_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
_MAX_SEQ_LEN = 2048
_MAX_NEW_TOKENS = 1024


@dataclass
class GenerationResult:
    prediction: Prediction
    tokens_used: Optional[int]
    latency_ms: float
    raw: str


def _parse_prediction(text: str) -> Prediction:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return Prediction()
    try:
        data = json.loads(m.group(0))
        return Prediction.model_validate(
            {"phi_spans": data.get("phi_spans", []), "resources": data.get("resources", [])}
        )
    except Exception:
        return Prediction()


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def extract_ollama(
    note: str,
    target_resources: list[str],
    *,
    model: str = SPECCED_MODEL,
    host: str = OLLAMA_HOST,
) -> GenerationResult:
    import urllib.request
    from train.prompt import format_inference_prompt

    prompt = format_inference_prompt(note, target_resources)
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.95, "num_ctx": _MAX_SEQ_LEN},
    }).encode()

    t0 = time.monotonic()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    raw = data.get("response", "")
    return GenerationResult(
        prediction=_parse_prediction(raw),
        tokens_used=data.get("eval_count"),
        latency_ms=latency_ms,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# HuggingFace / Unsloth backend
# ---------------------------------------------------------------------------

class HFBackend:
    """Context-manager that loads a model (optionally with LoRA adapter), runs
    inference, then frees GPU memory on exit."""

    def __init__(self, adapter_path: Optional[str] = None):
        self.adapter_path = adapter_path
        self._model: Any = None
        self._tokenizer: Any = None

    # -- lifecycle --

    def load(self) -> "HFBackend":
        from unsloth import FastLanguageModel

        source = self.adapter_path or _BASE_MODEL
        print(f"Loading {'adapter' if self.adapter_path else 'base model'}: {source}")
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=source,
            max_seq_length=_MAX_SEQ_LEN,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(self._model)
        return self

    def unload(self) -> None:
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "HFBackend":
        return self.load()

    def __exit__(self, *_) -> None:
        self.unload()

    # -- inference --

    def extract(self, note: str, target_resources: list[str]) -> GenerationResult:
        if self._model is None:
            raise RuntimeError("HFBackend not loaded — use as a context manager or call .load() first.")

        import torch
        from train.prompt import format_inference_prompt

        prompt = format_inference_prompt(note, target_resources)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.monotonic()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        new_tokens = output_ids[0][input_len:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_used = len(new_tokens)

        return GenerationResult(
            prediction=_parse_prediction(raw),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            raw=raw,
        )
