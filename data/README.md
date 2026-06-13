# data/ — synthetic data + reject-sampling

✅ **US-2** in `STORIES.md`. PHI-safe `note → gold FHIR` generation.

The dataset philosophy: **PHI-safe data that earns its way in by passing the eval harness.** We never touch real patient data. A Python generator builds synthetic patients with ground-truth FHIR + synthetic PHI, a Claude note-writer turns them into realistic notes, and reject-sampling keeps only faithful, learnable pairs. The training target is the **gold**; the teacher is the consistency filter.

## Flow

```
generate.synth_case()  ─▶ gold FHIR (real codes) + synthetic PHI (Faker)
notes.write_note()     ─▶ realistic note (Claude / template) embedding the PHI, with exact spans
filter.accept()        ─▶ (1) concept-presence (cheap)  (2) teacher recovery via evals.run_eval
build.py               ─▶ dedup ─▶ split ─▶ data/curated/{train,val,held_out}.jsonl
```

Run it (offline: template notes + cheap filter, no API key; online: Claude note-writer + teacher when `ANTHROPIC_API_KEY` is set):

```bash
python -m data.build --n 8 --offline --out /tmp/curated --seed 0
python -m data.build --n 50            # online
```

## Modules

| File | Role |
|---|---|
| `knowledge.py` | curated clinical KB with real codes (RxNorm / ICD-10 / SNOMED / LOINC) + synonyms |
| `generate.py` | `synth_case()` → valid R4 gold FHIR + synthetic PHI + per-resource concept variants |
| `teacher.py` | Anthropic client — note-writer + extractor (model from `SPECCED_TEACHER_MODEL`) |
| `notes.py` | `write_note()` — Claude or template; PHI spans by locating the synthetic identifiers |
| `filter.py` | `accept()` — concept-presence + teacher recovery scored by the eval harness |
| `build.py` | orchestrator/CLI — dedup, split, write curated jsonl, log accept-rate |

## Directories

- `templates/` — note templates / generation config.
- `raw/` — raw teacher generations + reports (gitignored; large).
- `curated/` — schema-validated, deduped, split jsonl that feeds training (gitignored; large).

## Record format (curated jsonl)

```json
{
  "case": { "id": "...", "note": "...", "target_resources": ["..."], "deidentify": true,
            "gold": { "phi_spans": [...], "resources": [ ...FHIR... ] } },
  "prediction": { "phi_spans": [...], "resources": [ ...FHIR... ] },
  "eval": { "validity_rate": 1.0, "resource_f1": 0.93, "field_accuracy": 0.9, "deid_recall": 1.0, "score": 0.94 },
  "provenance": { "teacher_model": "...", "note_mode": "...", "git_sha": "...", "ts": "...", "seed": 0 }
}
```

> ⚠️ **No real PHI, ever** — synthetic or public de-identified only. ⚠️ **No copyrighted terminology files** committed — load SNOMED/RxNorm/etc. from a licensed local copy. See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
