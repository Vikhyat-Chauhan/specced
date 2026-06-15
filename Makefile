# Specced — common commands. Override the interpreter with `make PY=python <target>`.
PY ?= .venv/bin/python
CASE ?= specs/examples/cardio-visit.json

.PHONY: help eval data data-offline test typecheck hero gate train benchmark

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

eval: ## Score the example case through the eval harness (shows the harness catching issues)
	$(PY) -m evals.cli $(CASE)

data: ## Build a curated dataset (Claude note-writer + teacher when ANTHROPIC_API_KEY is set)
	$(PY) -m data.build --n 50

data-offline: ## Build a small dataset offline (template notes + cheap filter, no API key)
	$(PY) -m data.build --n 8 --offline --out /tmp/specced-curated --seed 0

test: ## Run the pytest suite
	$(PY) -m pytest -q

typecheck: ## Type-check with mypy if installed
	$(PY) -m mypy evals data 2>/dev/null || echo "mypy not installed — skipping"

hero: ## Regenerate the README hero (docs/specced-hero.svg)
	$(PY) scripts/make_hero.py

train: ## Fine-tune Qwen2.5-Coder-7B with QLoRA (requires curated data)
	$(PY) -m train.train_qlora --data data/curated --config train/configs/default.yaml

export-gguf: ## Merge LoRA adapter and export to GGUF for Ollama
	$(PY) -m train.export_gguf --adapter train/checkpoints/adapter --out serve/

benchmark: ## Base-vs-fine-tuned eval on held-out set (requires trained adapter)
	$(PY) -m evals.benchmark --data data/curated/held_out.jsonl --adapter train/checkpoints/adapter --n 20

gate: test data-offline ## Quality gate — run before every commit
	@echo "✓ quality gate passed"
