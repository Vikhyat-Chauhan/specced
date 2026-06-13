# agent/ — LangGraph orchestration

⬜ Planned — **US-4** in `STORIES.md`.

The `plan → retrieve → act → evaluate` state machine with a self-refine loop.

```
note → PLAN ──▶ RETRIEVE ──▶ ACT ──▶ EVALUATE ──┐
                (terminology  (FT model)  (evals/) │ invalid FHIR / low score
                  RAG)           ▲                 │  (≤ N refine loops)
                                 └──── feed errors ─┘
```

- **plan** — read the note + target resource types; outline what to extract.
- **retrieve** — pull candidate standard codes from `rag/` (SNOMED/RxNorm/ICD-10/LOINC).
- **act** — fine-tuned model emits `{ phi_spans, resources }` (`serve/`).
- **evaluate** — run the eval harness (`evals/run_eval`); on FHIR validation errors or low score, feed the errors back into `act`.

## Planned layout
- `graph.py` — LangGraph graph definition + state.
- `nodes/` — `plan.py`, `retrieve.py`, `act.py`, `evaluate.py`.

## Notes
- State carries the note, retrieved codes, current `{phi_spans, resources}`, eval report, and refine count.
- The agent calls the *same* `evals/` harness used for data and benchmarking — one oracle everywhere.
- FHIR validation errors are precise and machine-readable, which makes the self-refine loop genuinely effective.
