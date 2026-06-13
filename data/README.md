# data/ — synthetic data + reject-sampling

⬜ Planned — **US-2** in `STORIES.md`.

The dataset philosophy: **PHI-safe data that earns its way in by passing the eval harness.** We never touch real patient data. Instead we generate synthetic patients, template notes from them, and keep only teacher extractions that pass `evals/` (FHIR-valid + field-consistent). Teacher accept-rate is itself a quality signal.

## Flow

```
Synthea ──▶ synthetic patients (ground-truth FHIR) ──▶ templated clinical notes
   │
   ▼  (Claude teacher extracts note -> FHIR; see PROJECT.md for the model)
note -> FHIR predictions ──▶ raw/teacher.jsonl
   │
   ▼  evals/ harness (FHIR validity · field-F1 · de-id recall)
keep score >= threshold ──▶ curated/{train,val,held_out}.jsonl
```

## Directories

- `templates/` — note templates + Synthea config (how synthetic FHIR becomes free-text notes).
- `raw/` — every teacher generation + its eval report (gitignored; large).
- `curated/` — schema-validated, deduped, split jsonl that feeds training (gitignored; large).

## Record format (curated jsonl)

```json
{
  "case": { "id": "...", "note": "...", "target_resources": ["..."], "deidentify": true,
            "gold": { "phi_spans": [...], "resources": [ ...FHIR... ] } },
  "prediction": { "phi_spans": [...], "resources": [ ...FHIR... ] },
  "eval": { "validity_rate": 1.0, "resource_f1": 0.93, "field_accuracy": 0.9, "deid_recall": 1.0, "score": 0.94 },
  "provenance": { "teacher_model": "...", "git_sha": "...", "ts": "..." }
}
```

> ⚠️ **No real PHI, ever.** Synthetic (Synthea) or public de-identified benchmarks only. ⚠️ **No copyrighted terminology files** committed — load SNOMED/RxNorm/etc. from a licensed local copy.

Status: 🚧 implemented in D3–4.
