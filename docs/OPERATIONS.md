# Operations

Running Specced on your own infrastructure, the data-handling policy, and reproducibility. The defining constraint is simple: **PHI never leaves the host.**

## Local / on-prem serving

Specced is local-first by design — there is no hosted endpoint. The fine-tuned model is served on the same machine that holds the data:

- **Ollama** (default) — the QLoRA model is exported to GGUF and run via Ollama for the CLI and agent. Configure with `OLLAMA_HOST` and `SPECCED_MODEL`.
- **vLLM** (optional) — batched inference for faster eval throughput over the held-out set.

Generation is funneled through one client (`serve/`) so base model / fine-tuned / Claude are called through an identical interface — keeping the comparison benchmark apples-to-apples and the model swap a one-line change.

## Data-handling policy (read this first)

- **No real PHI, ever.** All data in this repository is **synthetic** (Faker + the curated KB in `data/knowledge.py`) or public de-identified benchmarks. Do not commit, log, or transmit real patient data.
- **De-identification is a hard gate.** De-id recall ≥ 0.95 is required to pass an eval; a missed PHI span is treated as a leak.
- **No licensed terminology files** are committed — SNOMED/RxNorm/etc. are loaded from a local licensed copy at runtime.
- **Curated outputs** (`data/curated/*.jsonl`) and per-run eval reports are git-ignored.

## Environment

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the Claude note-writer + teacher (data pipeline) and the clinical judge. Optional — the pipeline runs offline without it. |
| `SPECCED_TEACHER_MODEL` | Bulk data-gen teacher (default `claude-sonnet-4-6`). |
| `SPECCED_JUDGE_MODEL` | Eval clinical judge (default `claude-opus-4-8`). |
| `OLLAMA_HOST` / `SPECCED_MODEL` | Local serving endpoint + served model name. |
| `WANDB_API_KEY` / `WANDB_PROJECT` | Training/eval experiment tracking. |

## Reproducibility

- The data pipeline is seeded (`python -m data.build --seed N`) — same seed, same dataset.
- Every eval report records the git SHA, timestamp, and FHIR-validator mode (`evals/report.py`).
- Training runs log loss, GPU memory, wall-clock, and dataset version to Weights & Biases.

## Model export (US-3)

QLoRA adapter → merge → quantize → GGUF → Ollama. The export step and `Modelfile` live in `train/` and `serve/`; capture tokens/latency per note for the cost/latency comparison against Claude.
