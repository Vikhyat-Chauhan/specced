# evals/ — the eval harness

The heart of Specced. Given an extraction **case** (clinical note + gold FHIR) and a model **prediction**, it validates the FHIR, scores it field-by-field against gold, checks de-id recall, and writes a JSON report. The same `run_eval()` entry point is reused by the data reject-sampler (D3–4) and the agent's EVALUATE node (Week 4).

> In a regulated domain, "looks right" isn't good enough. Every output is validated against the FHIR schema and scored against gold.

## Pipeline

```
case (note + gold)  +  prediction { phi_spans, resources }
        │
        ▼
  ┌──────────────┐  Each resource validated against FHIR (version-pinned to the
  │ FHIR validity│  case). Invalid resources can't be trusted -> excluded from
  │   (gate)     │  matching and counted against precision.
  └──────┬───────┘
         ▼
  ┌──────────────┐  Predicted resources matched to gold by (resourceType,
  │ resource P/R/│  primary concept). Field accuracy = secondary fields
  │ F1 + fields  │  (dosage / value / status) correct on matched pairs.
  └──────┬───────┘
         ▼
  ┌──────────────┐  Gold PHI spans caught. Recall is the safety-critical metric
  │ de-id recall │  (a missed span is a leak); must be >= 0.95 to pass.
  └──────┬───────┘
         ▼
  ┌──────────────┐  Optional Claude judge (needs ANTHROPIC_API_KEY): flags
  │ clinical judge│ hallucinated/wrong content gold-matching can miss.
  └──────┬───────┘
         ▼
   aggregate score (0–1) + pass/fail  ->  reports/<case-id>/report.json
```

## Usage

```bash
uv pip install -e ".[evals]"          # fhir.resources (+ eval_type_backport on py<3.10)

# Score a prediction against a gold case (auto-finds <case>.pred.json if --pred omitted):
python -m evals.cli specs/examples/cardio-visit.json
python -m evals.cli specs/examples/cardio-visit.json --pred path/to/pred.json
python -m evals.cli specs/examples/cardio-visit.json --no-judge
```

## FHIR validation (the gate)

- **Strict** — uses `fhir.resources`, **pinned to the case's `fhir_version`** (R4/R4B → the `R4B` model set, R5 → top level). Real schema validation.
- **Fallback** — if `fhir.resources` can't import (e.g. Python < 3.10 without `eval_type_backport`), a lightweight required-field check runs instead so the harness still works. The report records which mode ran (`provenance.fhir_validator`).

## Scoring (`score.py`)

- Invalid FHIR resources are excluded from matching and counted against precision; `validity_rate` is reported separately.
- **Resource P/R/F1** — match predicted to gold by (resourceType, primary concept).
- **Field accuracy** — fraction of secondary fields (dosage/value/status) correct on matched pairs.
- **De-id recall** — gold PHI spans caught (recall-first; safety-critical).
- **Aggregate** — resource-F1 (0.5) + field accuracy (0.3) + de-id recall (0.2), renormalized over present components. **Pass** = score ≥ 0.7 **and** de-id recall ≥ 0.95 (when de-id is requested).

## Files

| Path | Role |
|---|---|
| `case.py` | Pydantic models (`Case`, `Gold`, `Prediction`); `load_case` / `load_prediction` |
| `fhir_validate.py` | Version-aware FHIR validity gate (strict + fallback) |
| `score.py` | Resource/field matching, de-id recall, aggregate |
| `judges/clinical.py` | Optional Claude clinical-correctness judge (+ `rubric.md`) |
| `report.py` | Build + write `report.json` (with provenance) |
| `run_eval.py` | Orchestrates one (case, prediction) → score + report. Reusable entry point. |
| `cli.py` | `python -m evals.cli <case.json> [--pred <pred.json>] [--no-judge]` |
| `reports/` | Per-case output |

## Worked example

`specs/examples/cardio-visit.json` + its `.pred.json` fixture exercise every signal: the prediction has a schema-invalid `AllergyIntolerance` (missing required `patient`), a dosage field miss (`10 mg` vs `10 mg daily`), and a missed `AGE` PHI span — so the harness reports 75% validity, 0.75 resource-F1, 0.667 field accuracy, 0.667 de-id recall, and **FAILs** on the de-id safety gate. That's the harness doing its job.

## Benchmarks

| Command | What it tests |
|---|---|
| `python -m evals.benchmark --data data/curated/held_out.jsonl --n 20` | Base vs FT on synthetic held-out set (with gold) |
| `python -m evals.compare --n 20` | Three-way: base vs FT vs FT+agent (with gold) |
| `python -m evals.mtsamples_bench --n 50` | Base vs FT on real MTSamples transcriptions (FHIR validity only, no gold) |

**Key results (RTX 5060 Ti, Qwen2.5-Coder-7B):**

*Synthetic held-out (20 cases, with gold):*

| Model | Passed | Resource F1 | De-id Recall |
|---|---|---|---|
| Base | 0/20 | 0.416 | 0.850 |
| Fine-tuned | 20/20 | 0.992 | 1.000 |
| FT + Agent | 20/20 | 1.000 | 1.000 |

*MTSamples OOD (50 real notes, FHIR validity only — no gold):*

| Model | Avg Validity | Resources/note | PHI spans/note | Empty outputs |
|---|---|---|---|---|
| Base | 0.951 | 3.06 | 5.9 | 13/50 |
| Fine-tuned | 0.842 | 5.08 | 7.06 | 6/50 |

OOD finding: the FT model extracts significantly more resources and PHI spans on real notes, but at lower precision — expected given 160-example synthetic training across a limited specialty set. The base model is more conservative (13 empty outputs) but what it does produce is mostly valid.

## Notes / limits

- **No real PHI** ever enters cases — synthetic (Faker) or public de-identified data only.
- The clinical judge is written + importable but runtime-verifying it needs an `ANTHROPIC_API_KEY`.
- MTSamples benchmark measures FHIR validity only (no gold annotations). For gold-annotated OOD eval, apply for i2b2 2014 (de-id) or n2c2 2018 (medication extraction).
