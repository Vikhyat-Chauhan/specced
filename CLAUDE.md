# CLAUDE.md — agent guardrails & context priming

You are working in **Specced**, a spec-driven toolkit for **clinical → FHIR extraction**. Read `docs/SPEC.md` for the task/data model and `docs/ARCHITECTURE.md` for how the code fits together before non-trivial work.

## What we build here

A locally fine-tuned model that turns a free-text clinical note into **de-identified, terminology-coded, schema-valid FHIR**, driven by a LangGraph agent loop and scored by a Python eval harness. Note-level extraction only (no cross-document reconciliation).

## Hard rules

- **Output contract:** the model emits one JSON object per note — `{ "phi_spans": [...], "resources": [ ...FHIR resources... ] }`. Every resource MUST validate as FHIR (R4), bind to standard terminologies (SNOMED/RxNorm/ICD-10/LOINC) where possible, and contain **no values not supported by the note** (no hallucinated meds, doses, or codes). De-id spans cover HIPAA Safe Harbor categories.
- **No real PHI, ever.** Training/eval data is synthetic (Synthea) or public de-identified benchmarks. Never commit, log, or transmit real patient data. The whole point is that PHI stays local.
- **No copyrighted/licensed terminology files in the repo.** Load SNOMED/RxNorm/etc. from the user's licensed local copy at runtime; commit only loaders + principle notes written in our own words.
- **Claude model IDs:** never hardcode from memory — consult the `claude-api` skill for current IDs/pricing before wiring a teacher/judge call.
- **One language — Python.** Data, training, agent, and the eval harness are all Python. Don't reintroduce a Node toolchain.
- **Every eval/generation records provenance:** model, version, temperature, seed, timestamp, git SHA.
- **Reject-sampling is the data philosophy:** an example earns its way into training only by passing the eval harness (FHIR-valid + field-consistent). Don't relax thresholds to inflate dataset size.

## Style

- Match existing file conventions. Small, composable, typed modules (Pydantic models for cases/outputs).
- Prefer reusing harness utilities over re-implementing.
- Recall is safety-critical for de-id: when in doubt, over-flag PHI rather than miss it.

## Definition of done for one extracted note

Every resource validates as FHIR ✓ · field-level F1 ≥ threshold vs gold ✓ · de-id recall ≥ threshold (no missed PHI) ✓ · no hallucinated values ✓.
