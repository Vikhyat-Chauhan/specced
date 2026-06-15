# How do you eval FHIR extraction?

Evaluating structured extraction is harder than evaluating text generation. You can't use BLEU or ROUGE — the output is a graph of typed objects, not a string. You need a domain-aware oracle that knows what "correct" means for a FHIR Condition vs a MedicationStatement.

Here's how Specced's eval harness works, and why each stage is necessary.

## The four-stage pipeline

```
Prediction
    ↓
[1] FHIR validity gate        (schema check per resource)
    ↓
[2] Resource matching         (P/R/F1 over typed concept pairs)
    ↓
[3] Field accuracy            (secondary fields on matched pairs)
    ↓
[4] De-id recall              (PHI span coverage, safety gate)
    ↓
Aggregate score → pass / fail
```

### Stage 1: FHIR validity

Each predicted resource is validated against the FHIR R4 schema using `fhir.resources`. Invalid resources are excluded from matching (you can't match a resource that isn't a valid resource) and counted against precision. The validity rate is reported separately.

Why not just flag invalid resources and keep going? Because an invalid FHIR resource is worse than a missing one — a downstream system that consumes the output may crash, or silently accept corrupted data. The validity gate is a hard prerequisite for resource matching.

The fallback (when `fhir.resources` isn't installed) checks only required fields per resource type. This is less strict but still useful in offline environments.

### Stage 2: Resource matching

Gold and predicted resources are matched by `(resourceType, primary_concept)`. The primary concept is extracted differently per type:

| Resource type | Primary concept |
|---|---|
| `Condition` | `code.text` or first coding display |
| `MedicationStatement` | `medicationCodeableConcept.text` |
| `Observation` | `code.text` |
| `AllergyIntolerance` | `code.text` or `substance.text` |

Both are normalized (lowercased, whitespace-collapsed) before comparison. This means "Type 2 Diabetes Mellitus" and "type 2 diabetes mellitus" match — surface form shouldn't penalize correct extractions.

From matched pairs, we compute precision, recall, and F1. Invalid predicted resources count against precision (they're false positives that happen to be broken).

### Stage 3: Field accuracy

On matched resource pairs, we check secondary fields:

| Resource type | Secondary fields |
|---|---|
| `MedicationStatement` | `dosage` (e.g., "10 mg daily") |
| `Observation` | `valueQuantity` or `valueString` |
| `Condition` | `clinicalStatus` (active/resolved/etc.) |

Field accuracy = fraction of secondary fields that match (normalized string comparison). A model that identifies the right medication but gets the dose wrong scores < 1.0 on field accuracy. This is important — a wrong dose is a clinical error, not just a labeling miss.

### Stage 4: De-id recall

For de-identification cases, we check that every gold PHI span appears in the prediction. The matching is by (type, normalized text) — not by exact position, since the model may detect the right PHI but report slightly different offsets.

**Recall is the only metric that matters here.** A false positive (flagging text that isn't PHI) is annoying; a false negative (missing a patient name or phone number) is a HIPAA violation. The gate is recall ≥ 0.95 — failing this gate fails the whole case, regardless of the FHIR score.

### Aggregate

```python
weights = [(0.5, resource_f1), (0.3, field_accuracy), (0.2, deid_recall)]
# renormalized over present components
agg = sum(w * c for w, c in weights) / sum(w for w, _ in weights)
passed = agg >= 0.7 AND deid_recall >= 0.95
```

The threshold weights encode the priority: getting the right resources is most important, getting field values right is next, de-id is last in the aggregate (but a hard gate independently).

## Why one oracle everywhere

The same `run_eval()` function is called in three places:
1. **Data pipeline** — `filter.accept()` calls it to reject-sample training pairs
2. **Benchmark** — `evals.benchmark` calls it to score base vs fine-tuned
3. **Agent** — the `evaluate` node calls it to decide whether to self-refine

This isn't an accident. When the training filter and the deployment gate are the same function, there's no semantic gap between "passed during data generation" and "passed in production". A model that learns to pass the training filter is learning to pass the deployment gate.

## What the harness doesn't check

- **Clinical correctness** — the harness can't know if "hypertension, active" is wrong because the note says "hypertension resolved". That requires the optional LLM judge (Claude Opus), which reads the note and checks for factual consistency.
- **Code system correctness** — a Condition coded with ICD-10 I10 and one coded with SNOMED 38341003 both match the same gold, since matching is on display text, not on codes. Coding accuracy is partially covered by field accuracy but not fully.
- **Cross-resource consistency** — a patient can't be allergic to penicillin in AllergyIntolerance but have it listed in MedicationStatement. The harness scores each resource independently.

These are known limitations. For the MVP (note-level extraction), they're acceptable. Cross-resource consistency and full code validation are deferred.

## Usage

```bash
# Score a single case
python -m evals.cli my_case.json

# Base vs fine-tuned on held-out set
python -m evals.benchmark --data data/curated/held_out.jsonl --n 20

# Three-way comparison with chart
make compare
```
