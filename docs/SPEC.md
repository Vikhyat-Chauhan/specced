# Specced — Specification

The product spec: what Specced does, the task and data model, the output contract, and the non-functional requirements it is held to. For *how the code fits together*, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Goal

Turn free-text clinical notes into **de-identified, terminology-coded, schema-valid FHIR**, entirely **on-premise**, and prove the quality with an objective eval harness.

## Why this task

- **Privacy makes "local" a requirement, not a preference.** PHI cannot go to a hosted API, so a locally fine-tuned model is the only compliant option — not just the cheaper one.
- **FHIR is the universal substrate.** Scribing, revenue-cycle, interoperability, and agent platforms all need valid FHIR out of messy text, so the capability transfers across the sector.
- **The output is machine-verifiable.** FHIR schema validation + field-level F1 against gold give an objective oracle, so the eval harness (and reject-sampling) stay rigorous.
- **It subsumes the alternatives.** Producing good FHIR includes a **de-identification** stage (privacy) and a **terminology-coding** stage (SNOMED/RxNorm/ICD-10/LOINC) — one repo demonstrates all three.

## Non-goals (MVP)

- Longitudinal patient records / cross-document reconciliation.
- Beating Claude outright — a near-tie at far lower cost, latency, and on-prem privacy is the win.
- Hosted demo, DPO, 14B training — deferred.

## Data model — the extraction case

A **case** is the unit of work and evaluation: an input note + the extraction task + (for eval) the gold answer. Cases are JSON validated against [`specs/case.schema.json`](../specs/case.schema.json).

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
      { "text": "67", "type": "AGE", "start": 17, "end": 19 }
    ],
    "resources": [
      { "resourceType": "Condition", "code": { "text": "hypertension" } },
      { "resourceType": "MedicationStatement", "status": "active",
        "medicationCodeableConcept": { "text": "lisinopril" }, "dosage": [{ "text": "10 mg daily" }] }
    ]
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | ✅ | Stable case identifier. |
| `note` | ✅ | The free-text clinical note (input). **Synthetic or de-identified only — never real PHI.** |
| `fhir_version` | — | Target FHIR version (default `R4`). |
| `target_resources` | — | Resource types to extract; omit to extract all found. |
| `deidentify` | — | If `true`, the model must also return `phi_spans` and de-id recall is scored. |
| `gold` | for eval | Reference answer: `phi_spans` + `resources`. Required to score; omit for unlabeled inference. |

## Output contract

The model emits one JSON object per note:

```json
{ "phi_spans": [ {"text","type","start","end"} ],
  "resources": [ { "resourceType": "...", ...valid FHIR... } ] }
```

- Every resource must **validate as FHIR** (R4 by default).
- **Code** clinical concepts to standard terminologies where possible (SNOMED / RxNorm / ICD-10 / LOINC); plain `text` is accepted when no code is grounded.
- **No values the note doesn't support** — never invent a med, dose, or code.
- `phi_spans` cover HIPAA Safe Harbor categories (NAME, DATE, AGE, LOCATION, ID, PHONE, …). **Recall matters most** — a missed span is a leak.

## How a case is scored

FHIR schema validity (hard gate) → field-level precision/recall/F1 of resources vs `gold.resources` → de-id recall of `phi_spans` vs `gold.phi_spans` → optional clinical-correctness LLM judge → aggregate + pass/fail. Details in [`evals/README.md`](../evals/README.md).

## Non-functional requirements

| Requirement | Target |
|---|---|
| **Privacy** | PHI never leaves the host; data in the repo is synthetic or public de-identified only. |
| **FHIR validity** | Every emitted resource validates against the case's FHIR version. |
| **De-id recall** | ≥ 0.95 (safety gate) — missing PHI is treated as a failure. |
| **Faithfulness** | No values unsupported by the note (no hallucinated meds/doses/codes). |
| **Reproducibility** | Fixed seeds; every report records model/version/seed/git-SHA. |
| **Determinism of data** | An example enters training only by passing the eval harness (reject-sampling). |

## Cases at scale

For training/eval, cases are produced in bulk by the **data pipeline** (`data/`): a synthetic generator builds patients with ground-truth FHIR + synthetic PHI, a Claude note-writer turns them into realistic notes, and reject-sampling keeps only faithful, learnable pairs. See [`data/README.md`](../data/README.md).
