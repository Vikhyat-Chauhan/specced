# Specced

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)
![Unsloth](https://img.shields.io/badge/Unsloth-00A67E)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![Claude](https://img.shields.io/badge/Claude-D97757?logo=anthropic&logoColor=white)
![FHIR R4](https://img.shields.io/badge/FHIR-R4-E1306C)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/W%26B-FFBE00?logo=weightsandbiases&logoColor=black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A local-first toolkit that turns free-text clinical notes into **de-identified, terminology-coded,
schema-valid FHIR**. A locally fine-tuned small model runs inside a `plan → retrieve → act →
evaluate` agent loop, and **every output is validated against the FHIR schema and scored
field-by-field against gold** — because in a regulated domain, "looks right" isn't good enough.

**Local-first** — PHI never leaves the box (HIPAA-aligned); there is no hosted demo by design.

[![Specced — clinical note to FHIR eval](docs/specced-hero.svg)](docs/ARCHITECTURE.md)

## Features

- **De-identification** — surfaces PHI spans (HIPAA Safe Harbor categories); recall is gated ≥ 0.95 because a missed span is a leak.
- **FHIR extraction** — emits R4 resources (Condition, MedicationStatement, Observation, AllergyIntolerance, …) from messy clinical text.
- **Terminology coding** — binds concepts to SNOMED / RxNorm / ICD-10 / LOINC.
- **Eval harness** — the oracle: version-aware FHIR schema validation, resource & field-level precision/recall/F1 vs gold, de-id recall, and an optional clinical LLM judge.
- **Data pipeline** — PHI-safe synthetic `note → gold FHIR` generation with **reject-sampling**: a candidate enters training only by passing the eval harness.
- **Local & on-prem** — the fine-tuned model is served locally (Ollama); data never leaves the host.
- **Agent loop** *(planned)* — LangGraph `plan → retrieve → act → evaluate` with self-refinement on FHIR validation errors.

## Tech stack

| Concern        | Choice                                                        |
|----------------|---------------------------------------------------------------|
| Language       | Python                                                        |
| Base model     | Qwen2.5-Coder-7B-Instruct                                     |
| Fine-tuning    | Unsloth + TRL — QLoRA (NF4), fits a 16 GB GPU                 |
| Eval           | `fhir.resources` validation + a custom field-F1 / de-id harness |
| Orchestration  | LangGraph (plan → retrieve → act → evaluate)                 |
| Teacher / judge| Anthropic Claude (Sonnet 4.6 bulk · Opus 4.8 judge)          |
| Data           | Faker + a curated terminology knowledge base                 |
| Serving        | Ollama (GGUF); vLLM for batch eval                           |
| Tracking       | Weights & Biases                                             |

## Quick start

1. **Create a virtualenv**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

2. **Install** (the eval + data extras)
   ```bash
   pip install -e ".[evals,data]"
   ```

3. **Score an extraction** — run the harness on the example case (gold note + a sample prediction):
   ```bash
   python -m evals.cli specs/examples/cardio-visit.json
   ```

4. **Build a dataset** — generate PHI-safe synthetic training data offline (no API key needed):
   ```bash
   python -m data.build --n 8 --offline --out /tmp/curated --seed 0
   ```
   Set `ANTHROPIC_API_KEY` to switch on the Claude note-writer + teacher reject-sampling.

## Project structure

```
evals/      Eval harness (the oracle): FHIR validity, field-F1, de-id recall, judge
data/       Data pipeline: synthetic (note -> gold FHIR) generation + reject-sampling
agent/      LangGraph plan->retrieve->act->evaluate graph        (planned)
rag/        Terminology retrieval (SNOMED/RxNorm/ICD-10/LOINC)   (planned)
train/      Unsloth/TRL QLoRA fine-tuning                        (planned)
serve/      Ollama / vLLM local serving                          (planned)
specs/      Extraction-case JSON schema + example case/prediction
cli/        `specced` command line                               (planned)
tests/      pytest suite
docs/       SPEC, ARCHITECTURE, OPERATIONS
```

## Architecture

**Pattern:** `plan → retrieve → act → evaluate`. Read the note and target resource types → retrieve
candidate standard codes (`rag/`) → the fine-tuned model emits `{ phi_spans, resources }` (`serve/`)
→ the eval harness validates and scores it (`evals/run_eval.py`); on a FHIR-validation or low-score
failure, the errors are fed back to **act** and it retries (self-refine).

The same harness is the oracle everywhere — CLI, the data reject-sampler, and the agent's evaluate
step all call `evals/run_eval.py`. FHIR validity is a hard gate (`evals/fhir_validate.py`, pinned to
the case's version), and the **training target is always the gold** — the Claude teacher is a
consistency filter, never the label source. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for
the directory map, the eval-harness internals, and the data-pipeline flow.

## Evaluation

The eval harness scores every extraction on **FHIR schema validity** (hard gate) → **resource &
field-level P/R/F1** vs gold → **de-id recall** (safety-critical) → an optional clinical LLM judge,
then aggregates to a single pass/fail. The headline result is a reproducible **base vs fine-tuned vs
Claude** comparison on a held-out set.

```bash
python -m evals.cli specs/examples/cardio-visit.json
```

See **[evals/README.md](evals/README.md)** for the scoring rules.

## Testing

Tests live in `tests/` and run on [pytest](https://docs.pytest.org). Every feature ships a happy-path
test **plus at least one failure case** (a schema-invalid resource is flagged; a missed PHI span drops
de-id recall). Run the full **quality gate** before any commit:

```bash
make gate    # pytest + an offline data build
```

## Roadmap

Tracked user stories, in order (detail in [`STORIES.md`](STORIES.md)):

- [x] **US-1** — Spec-first scaffold + eval harness (FHIR validity · field-F1 · de-id recall)
- [x] **US-2** — Data pipeline: synthetic generator → Claude note-writer → reject-sampling → curated jsonl
- [ ] **US-3** — QLoRA fine-tune (Qwen2.5-Coder-7B) + local serving + baseline evals
- [ ] **US-4** — LangGraph agent: plan → retrieve → act → evaluate + self-refine
- [ ] **US-5** — Terminology RAG (SNOMED / RxNorm / ICD-10 / LOINC)
- [ ] **US-6** — Full eval pipeline + base-vs-fine-tuned-vs-Claude comparison + charts
- [ ] **US-7** — CLI + scaffold polish
- [ ] **US-8** — Write-up, blog series, demo

## Scripts

| Target | Purpose |
|--------|---------|
| `make eval` | Score the example case through the eval harness |
| `make data` | Build a curated dataset (Claude note-writer + teacher when a key is set) |
| `make data-offline` | Build a small dataset offline (template notes + cheap filter) |
| `make test` | Run the pytest suite |
| `make gate` | Quality gate — pytest + offline data build |
| `make hero` | Regenerate the README hero (`docs/specced-hero.svg`) |

## Running locally

Specced is local-first — there is no hosted endpoint. The fine-tuned model is served on the same
machine that holds the data, via **Ollama** (GGUF), with vLLM as an option for batch eval throughput.
Set the variables in `.env.example`; `ANTHROPIC_API_KEY` is optional (the data pipeline runs offline
without it). See **[docs/OPERATIONS.md](docs/OPERATIONS.md)** for the data-handling policy, environment,
and reproducibility.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** — agent steering guide: rules, directory map, output contract.
- **[docs/SPEC.md](docs/SPEC.md)** — the task, data model, output contract, and non-functional requirements.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the code fits together.
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)** — local serving, the no-PHI policy, and reproducibility.
- **[STORIES.md](STORIES.md)** — the user-story backlog.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — dev workflow, conventions, and the quality gate.
- **[SECURITY.md](SECURITY.md)** — vulnerability disclosure and the no-real-PHI policy.

## License

[MIT](LICENSE)
