# Specced

> A spec-driven, eval-first toolkit for **clinical → FHIR extraction**. Turn free-text clinical notes into **de-identified, terminology-coded, schema-valid FHIR** — using a locally fine-tuned small model wrapped in a `plan → retrieve → act → evaluate` agent loop, and proven by a reproducible Python eval harness.

Healthcare data can't leave the building (HIPAA), so the model runs **locally**. Every output is **validated against the FHIR schema and scored field-by-field against gold** — because in a regulated domain, "looks right" isn't good enough.

## Why this exists

FHIR is the substrate nearly every health-AI product sits on (scribing, revenue-cycle, interoperability, analytics, agents). Getting structured, valid, coded FHIR out of messy clinical text is the hard, universal first step — and you **cannot** outsource it to a cloud API when the input is PHI. Specced is an open, local-first, **eval-driven** pipeline that does it and *measures* it.

## The pipeline

```
clinical note ──▶ PLAN ──▶ RETRIEVE ──▶ ACT ──▶ EVALUATE ──▶ de-identified, coded,
                          (terminology  (FT model)  (harness)     schema-valid FHIR + report
                            RAG)             ▲                        │
                                             └──── self-refine ◀──────┘
                                              (on FHIR validation errors)
```

The model's job, per note:
1. **De-identify** — surface PHI spans (HIPAA Safe Harbor categories).
2. **Extract** — emit FHIR resources (Condition, MedicationStatement, Observation, AllergyIntolerance, …).
3. **Code** — bind to standard terminologies (SNOMED / RxNorm / ICD-10 / LOINC) where possible.

## Eval harness (the crown jewel)

Every output is scored on:
1. **FHIR schema validity** — does each resource validate as FHIR? (hard gate)
2. **Field-level F1** — extracted resources/fields vs gold (precision / recall / F1).
3. **De-id recall** — PHI spans caught vs gold (recall is the safety-critical metric).
4. *(optional)* **Clinical-correctness LLM judge** — for borderline cases.

The money chart: run the held-out set across **base model vs fine-tuned vs Claude** → comparison table.

## Layout

| Dir | What |
|---|---|
| `specs/` | Extraction-case schema + example cases (note → gold FHIR) |
| `data/` | Synthea synthesis + teacher reject-sampling → train/val/held-out jsonl |
| `train/` | Unsloth/TRL QLoRA scripts + configs |
| `serve/` | Ollama / vLLM serving |
| `agent/` | LangGraph plan→retrieve→act→evaluate graph |
| `rag/` | Terminology retrieval (SNOMED/RxNorm/ICD-10/LOINC) + principles |
| `evals/` | Python harness: FHIR validity, field-F1, de-id recall, judge, reports |
| `cli/` | `specced extract ./case.json` |

## Status

🚧 Week 3–4 MVP in progress. `STORIES.md` is the backlog; `SPEC.md` explains the extraction-case format; `PROJECT.md` has the architecture and decisions.

## Quickstart (WIP)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[evals]"
# Score a prediction against a gold case:
python -m evals.cli specs/examples/cardio-visit.json --pred specs/examples/cardio-visit.pred.json
```

## License

MIT (intended — see LICENSE).
