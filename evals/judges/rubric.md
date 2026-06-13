# Judging rubric — clinical → FHIR extraction

The primary oracle is **objective**: FHIR schema validity + field-level F1 vs gold + de-id recall. The LLM clinical judge below is a **secondary** signal for what gold-matching can miss (hallucinated or clinically wrong content). Authored in our own words.

## Objective metrics (primary — see `score.py`)

| Metric | What it measures | Bar |
|---|---|---|
| **FHIR validity rate** | fraction of predicted resources that validate as FHIR (R4) | invalid resources don't count as correct |
| **Resource P/R/F1** | predicted resources matched to gold by (resourceType, primary concept) | higher is better |
| **Field accuracy** | over matched pairs, fraction of secondary fields correct (dosage, value, status) | higher is better |
| **De-id recall** | gold PHI spans caught (recall is safety-critical — a miss is a leak) | ≥ 0.95 to pass |

## Clinical-correctness judge (secondary — `clinical.py`)

The judge reads the note + extracted resources and scores `clinical_correctness` 0–1, flagging:

- **Hallucinations** — any med, dose, value, or code **not supported by the note** (heavily penalized).
- **Coding errors** — clinically wrong terminology binding (e.g., wrong RxNorm/SNOMED concept).
- **Completeness** — clinically salient facts in the note that were missed.

## Aggregate

`score.py` combines the present components — resource-F1 (0.5) + field accuracy (0.3) + de-id recall (0.2), renormalized — and requires score ≥ 0.7 **and** de-id recall ≥ 0.95 (when de-id is requested) to pass.
