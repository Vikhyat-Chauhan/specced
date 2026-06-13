# STORIES.md — user-story backlog

The project as a sequence of user stories. Each has a goal, scope, acceptance criteria, and status. Maps to the Weeks 3–4 build plan in `PROJECT.md`. Status: ✅ done · 🚧 in progress · ⬜ planned.

> Scope boundary for the MVP: **note-level** clinical → FHIR extraction (one note → de-identified, coded, schema-valid FHIR). Cross-document reconciliation, full patient records, DPO, and a hosted demo are deferred.

---

## US-1 — Spec-first scaffold + eval harness ✅
**As** an ML engineer, **I want** a cloneable scaffold and an eval harness that scores a clinical→FHIR extraction, **so that** every later story has a reproducible, objective oracle.

- **Scope:** `CLAUDE.md`/`PROJECT.md`/`SPEC.md`, extraction-case JSON schema + Pydantic, example case + prediction fixture; Python harness: FHIR validity (version-pinned) → resource P/R/F1 + field accuracy → de-id recall → optional clinical judge → aggregate → JSON report.
- **Acceptance:**
  - [x] `python -m evals.cli <case> [--pred <pred>]` produces a scored `report.json`.
  - [x] FHIR validity is a real gate (strict via `fhir.resources`, graceful fallback); de-id recall is a safety gate (≥ 0.95).
  - [x] Runs end-to-end on the fixture in both validator modes; catches an invalid resource, a field miss, and a missed PHI span.
  - [ ] Clinical judge runtime-verified against a real `ANTHROPIC_API_KEY` (blocked on key).
- **Docs:** `evals/README.md`, `evals/judges/rubric.md`.

---

## US-2 — Data pipeline: Synthea + teacher reject-sampling ⬜
**As** an ML engineer, **I want** a PHI-safe `note → FHIR` dataset built by filtering a teacher through the eval harness, **so that** the fine-tune learns from outputs that pass our bar.

- **Scope:** Synthea generates synthetic patients (ground-truth FHIR); template clinical notes from them; Claude teacher extracts; keep only extractions that pass the harness (FHIR-valid + field-consistent); schema-validate, dedup, split into `train`/`val`/`held_out` jsonl.
- **Acceptance:**
  - [ ] Synthea → notes + gold FHIR cases across resource types.
  - [ ] Teacher generation + reject-sampling → `data/curated/{train,val,held_out}.jsonl`.
  - [ ] Each record carries case, prediction, eval result, provenance; teacher accept-rate logged.
  - [ ] **No real PHI** anywhere.
- **Docs:** `data/README.md`.

---

## US-3 — QLoRA fine-tune + serving + baseline evals ⬜
**As** an ML engineer, **I want** to QLoRA-fine-tune Qwen2.5-Coder-7B on `note → FHIR` pairs, serve it locally, and benchmark base-vs-fine-tuned, **so that** I demonstrate a compliant on-prem extraction model.

- **Scope:** Unsloth + TRL QLoRA (NF4) on 16GB; W&B; export GGUF; Ollama; run the held-out set on base vs fine-tuned.
- **Acceptance:**
  - [ ] Training run with decreasing loss logged to W&B; GGUF exported + served.
  - [ ] Served model extracts FHIR for a sample note.
  - [ ] Comparison table: base vs fine-tuned on validity / resource-F1 / field accuracy / de-id recall.
- **Docs:** `train/README.md`, `serve/README.md`.

---

## US-4 — LangGraph agent: plan → retrieve → act → evaluate + self-refine ⬜
**As** a user, **I want** an agent that extracts, validates, and refines on FHIR errors, **so that** output validity and accuracy beat a single shot.

- **Scope:** LangGraph graph; `act` = fine-tuned model; `evaluate` = the harness; on FHIR validation/eval failure, feed errors back to `act` (≤ N iterations).
- **Acceptance:**
  - [ ] One note runs through the full graph → de-identified, coded, schema-valid FHIR + report.
  - [ ] A failing first attempt (invalid resource / missed field) is measurably improved by self-refine (before/after logged).
- **Docs:** `agent/README.md`.

---

## US-5 — Terminology RAG ⬜
**As** the agent, **I want** to retrieve standard codes during `retrieve`, **so that** extracted concepts are correctly coded (SNOMED/RxNorm/ICD-10/LOINC).

- **Scope:** embed + vector store over terminology concepts (loaded from the user's licensed local copy); wire into the `retrieve` node so `act` can bind codes.
- **Acceptance:**
  - [ ] Retrieval returns plausible codes for concepts in a note (e.g. "lisinopril" → RxNorm).
  - [ ] Ablation: agent with vs without terminology RAG, scored on the held-out set.
  - [ ] No copyrighted terminology files committed (loaders only).
- **Docs:** `rag/README.md`.

---

## US-6 — Full eval pipeline + comparison + charts ⬜
**As** an AI engineer, **I want** the base-vs-FT-vs-Claude comparison with charts, **so that** the project has its centerpiece result.

- **Scope:** clinical judge across the held-out set for all three systems; efficiency metrics (tokens/latency/$ per note); comparison table + charts.
- **Acceptance:**
  - [ ] One command reproduces the comparison from the held-out set.
  - [ ] Charts: validity & F1 by resource type, **de-id recall**, self-refine lift, cost/latency.
- **Docs:** extend `evals/README.md` with the benchmark runner.

---

## US-7 — CLI + scaffold polish ⬜
**As** a cloner, **I want** `specced extract ./note.txt` and a clean quickstart, **so that** I can run extraction end-to-end from a fresh checkout.

- **Scope:** `cli/` (`specced extract` / `specced eval`); tighten the cloneable scaffold.
- **Acceptance:**
  - [ ] Fresh clone → follow README → extract FHIR from a sample note via the CLI, no manual hacks.
- **Docs:** `cli/README.md`, top-level `README.md`.

---

## US-8 — Write-up, blog series, demo ⬜
**As** a job-seeker, **I want** metrics, charts, blog posts, and a recorded demo, **so that** the project is interview- and portfolio-ready for health-AI AI-Engineer roles.

- **Scope:** blog angles (reject-sampling; QLoRA on 16GB; "how do you eval FHIR extraction?"; self-refine on schema errors; local-vs-Claude on cost/latency/privacy; de-id recall as a safety metric); charts; demo.
- **Acceptance:**
  - [ ] README links the comparison results + headline write-up.
  - [ ] Demo shows note → agent loop → de-identified, coded, validated FHIR.

---

## Deferred (post-MVP)
DPO/preference tuning · 14B inference comparison · cross-document / longitudinal reconciliation · full FHIR profile/US-Core conformance · hosted demo · additional verticals (the same harness retargets to contracts/insurance/finance by swapping the schema + gold).

## Open design decisions
- **FHIR version & profile depth** — R4B base validation for the MVP; US-Core profile conformance deferred.
- **How much terminology coding to require** in the MVP (text-only acceptable vs. coded-required) — currently text accepted, coding rewarded.
