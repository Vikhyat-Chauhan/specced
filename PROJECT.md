# PROJECT.md — Specced

Single source of truth for goals, architecture, conventions, and the build plan. (The agent reads this; humans read this; keep it current.)

## Goal

Ship an MVP that demonstrates the full AI-Engineering stack on one narrow, high-value, privacy-bound task: **clinical free-text → de-identified, terminology-coded, schema-valid FHIR.** Optimize for a defensible, reproducible story that gets the attention of AI-Engineer recruiters at health-AI companies.

## Why this task (the thesis)

- **Privacy makes "local" a requirement, not a preference.** PHI cannot go to a hosted API, so a locally fine-tuned model is the *only* compliant option — not just the cheaper one.
- **FHIR is the universal substrate.** Scribing, revenue-cycle, interoperability, and agent platforms all need valid FHIR out of messy text — so the skill transfers across the whole sector.
- **The output is machine-verifiable.** FHIR schema validation + field-level F1 against gold give an objective oracle, so the eval harness (and reject-sampling) stay rigorous.
- **It subsumes the alternatives.** Producing good FHIR includes a **de-identification** stage (privacy) and a **terminology-coding** stage (SNOMED/RxNorm/ICD-10/LOINC) — so one repo demonstrates all three.

## Non-goals (MVP)

- Full longitudinal patient records / cross-document reconciliation.
- Beating Claude outright (a near-tie at far lower cost, lower latency, and on-prem privacy is the win).
- Hosted demo, DPO, 14B training — deferred to polish.

## Architecture

`plan → retrieve → act → evaluate` (LangGraph), with a self-refine loop feeding FHIR validation/eval failures back into `act` (≤ N iterations).

- **PLAN** — read the note + target resource types; outline what to extract.
- **RETRIEVE** — RAG over medical terminologies (SNOMED/RxNorm/ICD-10/LOINC) + extraction principles to ground codes.
- **ACT** — fine-tuned `Qwen2.5-Coder-7B` emits `{ phi_spans, resources }` (de-id + FHIR).
- **EVALUATE** — the harness: FHIR schema-validity (gate), field-level F1, de-id recall; on failure, feed the validation errors back to `act`.

## Output contract

The model emits one JSON object per note:
```json
{ "phi_spans": [ {"text","type","start","end"} ],
  "resources": [ { "resourceType": "...", ...valid FHIR... } ] }
```
Resources MUST validate as FHIR (R4 by default), bind to standard terminologies where possible, and contain **no values not supported by the note** (no hallucinated meds/doses/codes).

## Tech decisions (locked for MVP)

| Concern | Choice | Rationale |
|---|---|---|
| Task | clinical note → de-identified, coded FHIR | privacy-bound, verifiable, universal |
| FHIR version | R4 | most widely deployed |
| Base model | Qwen2.5-Coder-7B-Instruct | strong structured-JSON output; QLoRA fits 16GB |
| Fine-tune | Unsloth + TRL, QLoRA NF4 | most-supported recipe, 16GB-friendly |
| Teacher + judges | Claude Opus 4.8 (quality) / Sonnet 4.6 (bulk) | reject-sampling teacher + clinical judge |
| Serving | Ollama (GGUF); vLLM for batch eval | simple local demo + throughput |
| Orchestration | LangGraph (Python) | stateful control + self-refine |
| RAG | LanceDB/Chroma + bge-small over terminologies | ground codes (SNOMED/RxNorm/ICD-10/LOINC) |
| Eval harness | **Python**: fhir.resources validation + field-F1 + de-id recall | one-language stack; objective oracle |
| Tracking | Weights & Biases | reproducible experiment logs |
| Synthetic data | Synthea (synthetic FHIR + templated notes) | PHI-safe training data |

> ⚠️ Before wiring any Claude call, read the `claude-api` skill for current model IDs/pricing. Do not hardcode model strings from memory.

## Conventions

- **One language: Python.** Data, training, agent, and the eval harness are all Python. (No Node — the frontend harness was retired in the Clinical→FHIR retarget.)
- **Extraction cases** validate against `specs/case.schema.json` (mirrored as Pydantic in `evals/`).
- **No real PHI in the repo.** Training/eval data is synthetic (Synthea) or public de-identified benchmarks; never commit real patient data.
- **No copyrighted terminology files** committed — load SNOMED/RxNorm/etc. from the user's licensed local copy at runtime; ship only loaders + our own principle notes.
- **Determinism for evals:** fixed seeds where possible; record model/version/temperature/seed/git-SHA in every report.
- **Reject-sampling is the data philosophy:** an example earns its way into training only by passing the eval harness (schema-valid + field-consistent). Don't relax thresholds to inflate dataset size.

## Build plan (Weeks 3–4)

- **D1–2** ✅ scaffold + spec-first framework + eval-harness skeleton (FHIR validity + field-F1 + de-id recall) + rubric
- **D3–4** data pipeline: Synthea + teacher reject-sampling → train/val/held-out
- **D5** QLoRA train + GGUF export + Ollama + baseline evals (base vs FT)
- **D6–7** LangGraph graph + self-refine + terminology RAG
- **D8** clinical judge + base-vs-FT-vs-Claude comparison + charts
- **D9** serving + CLI + scaffold polish
- **D10** write-up + charts + README + demo (buffer)

## Metrics

FHIR validity rate · field-level precision/recall/F1 (per resource type + overall) · **de-id recall** (safety-critical) · self-refine lift · tokens/latency/$ per note (local FT vs Claude) · training loss/mem/wall-clock · teacher accept-rate (reject-sampling signal).
