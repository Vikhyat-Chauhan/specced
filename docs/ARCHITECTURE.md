# Architecture

How Specced fits together: the directory map, the processing lifecycle, and the internals of the two systems that are built today — the eval harness and the data pipeline. For *what* it does and *why*, see [SPEC.md](SPEC.md).

## Directory map

```
specced/
  evals/          Eval harness (the oracle): FHIR validity, field-F1, de-id recall, judge
    case.py         Pydantic models — Case / Gold / Prediction / PhiSpan
    fhir_validate.py Version-aware FHIR validity gate (strict fhir.resources + fallback)
    score.py        Resource/field matching, de-id recall, aggregate
    run_eval.py     One (case, prediction) -> score + report  (the reusable entry point)
    cli.py          `python -m evals.cli <case> [--pred <pred>]`
    judges/         Optional Claude clinical-correctness judge + rubric
  data/           Data pipeline: synthetic (note -> gold FHIR) generation
    knowledge.py    Curated clinical KB with real codes (RxNorm/ICD-10/SNOMED/LOINC)
    generate.py     Synthetic patient -> valid R4 gold FHIR + synthetic PHI (Faker)
    teacher.py      Anthropic client (note-writer + extractor)
    notes.py        Note rendering (Claude / template) + exact PHI spans
    filter.py       Reject-sampling: concept-presence + teacher recovery via evals/
    build.py        `python -m data.build` -> dedup, split, curated jsonl
  agent/          LangGraph plan->retrieve->act->evaluate graph        (planned, US-4)
  rag/            Terminology retrieval (SNOMED/RxNorm/ICD-10/LOINC)   (planned, US-5)
  train/          Unsloth/TRL QLoRA fine-tuning                        (planned, US-3)
  serve/          Ollama / vLLM local serving                         (planned, US-3)
  specs/          Extraction-case JSON schema + example case/prediction
  cli/            `specced` command line                              (planned, US-7)
  docs/           SPEC, ARCHITECTURE, OPERATIONS
```

## Processing lifecycle

The runtime pattern, applied per note:

**`plan → retrieve → act → evaluate`** — read the note and target resource types → retrieve candidate standard codes (`rag/`) → the fine-tuned model emits `{ phi_spans, resources }` (`serve/`) → the eval harness validates and scores it (`evals/run_eval.py`); on FHIR-validation or low-score failure, the errors are fed back to **act** and it retries (self-refine, ≤ N loops). The agent (`agent/`) is US-4; today the harness and data pipeline are wired and tested.

The **training target is always the gold** (ground truth). The Claude teacher is a *consistency filter*, never the label source — "consistency-filtered synthetic supervision."

## Eval harness internals (`evals/`)

The single oracle reused everywhere (CLI, data reject-sampler, agent EVALUATE):

1. **FHIR validity gate** — `fhir_validate.validate_resource(resource, fhir_version)`. Strict validation via `fhir.resources`, pinned to the case's FHIR version (`R4`/`R4B` → the R4B model set, `R5` → top level); degrades to a lightweight required-field check when the library can't import. Invalid resources can't be trusted, so they're excluded from matching and counted against precision.
2. **Resource & field scoring** — `score.score(case, pred)`. Predicted resources are matched to gold by `(resourceType, primary concept)`; field accuracy is the fraction of secondary fields (dosage / value / status) correct on matched pairs.
3. **De-id recall** — gold PHI spans caught; recall is the safety-critical metric (gate ≥ 0.95).
4. **Aggregate** — resource-F1 (0.5) + field accuracy (0.3) + de-id recall (0.2), renormalized; **pass** requires the aggregate ≥ 0.7 and de-id recall ≥ 0.95.
5. **Provenance** — every report (`report.py`) records git SHA, timestamp, validator mode.

See [`evals/README.md`](../evals/README.md).

## Data pipeline internals (`data/`)

```
generate.synth_case()  ->  gold FHIR (real codes) + synthetic PHI (Faker)
notes.write_note()     ->  realistic note (Claude / template) embedding the PHI, exact spans
filter.accept()        ->  (1) concept-presence (cheap)  (2) teacher recovery via evals.run_eval
build.py               ->  dedup -> split -> data/curated/{train,val,held_out}.jsonl
```

`data/build.py` runs fully offline (template notes + cheap filter) with no API key, and switches to the Claude note-writer + teacher reject-sampling when `ANTHROPIC_API_KEY` is set. See [`data/README.md`](../data/README.md).

## Key conventions

- **One language — Python.** Data, training, agent, and the eval harness are all Python.
- **No real PHI.** Synthetic (Faker + curated KB) or public de-identified benchmarks only.
- **No licensed terminology files committed** — load SNOMED/RxNorm/etc. from a local licensed copy at runtime; the repo ships loaders + our own notes.
- **Claude model IDs** are never hardcoded from memory — see the `claude-api` skill (defaults: `claude-sonnet-4-6` bulk teacher, `claude-opus-4-8` judge).

## Tech decisions

| Concern | Choice | Rationale |
|---|---|---|
| Task | clinical note → de-identified, coded FHIR | privacy-bound, verifiable, universal |
| FHIR version | R4 (validated via R4B model set) | most widely deployed |
| Base model | Qwen2.5-Coder-7B-Instruct | strong structured JSON; QLoRA fits 16 GB |
| Fine-tune | Unsloth + TRL, QLoRA NF4 | most-supported recipe, 16 GB-friendly |
| Teacher / judge | Claude Sonnet 4.6 (bulk) / Opus 4.8 (judge) | note-writer + reject-sampling + clinical judge |
| Eval | `fhir.resources` + custom harness | objective, version-aware oracle |
| Orchestration | LangGraph | stateful self-refine |
| Serving | Ollama (GGUF); vLLM for batch | local demo + throughput |
| Tracking | Weights & Biases | reproducible experiment logs |
