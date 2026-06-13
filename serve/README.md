# serve/ — model serving

⬜ Planned — **US-3** in `STORIES.md`.

Serve the fine-tuned model **locally** (the whole point — PHI never leaves the box) for the agent and the eval benchmark.

- **Ollama (GGUF)** — simple local serving for the demo and CLI. `SPECCED_MODEL` / `OLLAMA_HOST` in `.env`.
- **vLLM (optional)** — batched inference for faster eval throughput over the held-out set.

## Planned layout
- `Modelfile` — Ollama model definition (base GGUF + extraction system prompt + params).
- `client.py` — thin generation client (`note → {phi_spans, FHIR}`) used by the agent and benchmark.

## Notes
- Capture tokens/latency per note for the efficiency metrics (local FT vs hosted Claude).
- Keep the generation interface identical for base / fine-tuned / Claude so the benchmark is apples-to-apples.
