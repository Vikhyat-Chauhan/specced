# train/ — QLoRA fine-tuning

⬜ Planned — **US-3** in `STORIES.md`.

Fine-tune `Qwen2.5-Coder-7B-Instruct` on the curated `data/curated/train.jsonl` (`note → {phi_spans, FHIR resources}` pairs) using **Unsloth + HuggingFace TRL**, QLoRA (NF4 4-bit), on a single 16GB GPU. Track with Weights & Biases. Export a GGUF for `serve/`.

## Planned layout
- `configs/` — training configs (LoRA rank/alpha, lr, epochs, max seq len).
- `train_qlora.py` — Unsloth/TRL training entry point.
- `export_gguf.py` — merge/quantize → GGUF for Ollama.

## Notes
- The model's target is structured JSON (de-id spans + FHIR), so a code model (Qwen-Coder) is a good base.
- Record loss curves, GPU mem, wall-clock, examples-seen to W&B.
- Reproducibility: fixed seed, dataset version (git SHA), config logged with the run.
