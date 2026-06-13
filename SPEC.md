# SPEC.md — How to write an extraction case

A **case** is the unit of work and evaluation in Specced. It pairs an input clinical note with the extraction task and (for eval) the gold answer. Cases are JSON validated against [`specs/case.schema.json`](specs/case.schema.json).

## Anatomy of a case

```json
{
  "id": "cardio-visit-001",
  "note": "Mr. John Carter, 67yo, seen 03/14/2024. Started lisinopril 10 mg daily for hypertension. BP today 148/92. Allergic to penicillin (rash).",
  "fhir_version": "R4",
  "target_resources": ["Condition", "MedicationStatement", "Observation", "AllergyIntolerance"],
  "deidentify": true,
  "gold": {
    "phi_spans": [
      { "text": "John Carter", "type": "NAME", "start": 4, "end": 15 },
      { "text": "67", "type": "AGE", "start": 17, "end": 19 },
      { "text": "03/14/2024", "type": "DATE", "start": 31, "end": 41 }
    ],
    "resources": [
      { "resourceType": "Condition", "code": { "text": "hypertension" }, "clinicalStatus": { "text": "active" } },
      { "resourceType": "MedicationStatement", "status": "active",
        "medication": { "text": "lisinopril" }, "dosage": "10 mg daily" },
      { "resourceType": "Observation", "status": "final",
        "code": { "text": "blood pressure" }, "value": "148/92 mmHg" },
      { "resourceType": "AllergyIntolerance", "code": { "text": "penicillin" }, "reaction": "rash" }
    ]
  }
}
```

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | ✅ | Stable case identifier. |
| `note` | ✅ | The free-text clinical note (input). **Synthetic or de-identified only — never real PHI.** |
| `fhir_version` | — | Target FHIR version (default `R4`). |
| `target_resources` | — | Resource types the model should extract; omit to extract all it can find. |
| `deidentify` | — | If `true`, the model must also return `phi_spans` and the eval scores de-id recall. |
| `gold` | for eval | The reference answer: `phi_spans` + `resources`. Required to score; omit for an unlabeled inference case. |

## What the model returns (output contract)

```json
{ "phi_spans": [ {"text","type","start","end"} ],
  "resources": [ { "resourceType": "...", ...valid FHIR... } ] }
```

- Every resource must **validate as FHIR** (R4).
- **Code** clinical concepts to standard terminologies where possible (SNOMED / RxNorm / ICD-10 / LOINC); plain `text` is accepted when no code is grounded.
- **No values the note doesn't support** — never invent a med, dose, or code.
- `phi_spans` cover HIPAA Safe Harbor categories (NAME, DATE, AGE > 89, LOCATION, ID, PHONE, …). **Recall matters most** — a missed span is a leak.

## How a case is scored

See `evals/README.md`. In short: **FHIR schema validity** (hard gate) → **field-level precision/recall/F1** of resources vs `gold.resources` → **de-id recall** of `phi_spans` vs `gold.phi_spans` → optional clinical-correctness LLM judge → aggregate score + pass/fail.

## Cases at scale

For training/eval, cases are produced in bulk by the **data pipeline** (`data/`): Synthea generates synthetic patients (with ground-truth FHIR), notes are templated from them, and a Claude teacher's extractions are kept only when they pass the harness (reject-sampling). See `data/README.md`.
