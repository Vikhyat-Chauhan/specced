# Reject-sampling as a data philosophy

When you're building a supervised fine-tuning dataset for structured extraction, the naive approach is to generate examples and call it done. The problem: if your generator produces invalid FHIR, or your note-writer drops a medication, the model learns to repeat those mistakes. Garbage in, garbage out — at 7B parameters.

Specced takes a different approach: **an example earns its way into training only by passing the eval harness**.

## The pipeline

```
synth_case()          →  gold FHIR + synthetic PHI (Faker)
write_note()          →  realistic note embedding PHI (Claude / template)
accept()              →  concept-presence filter  +  teacher recovery
                                                          ↓
                                              evals.run_eval ≥ 0.8 F1
                                                          ↓
                                              training example admitted
```

The accept filter has two stages:

**Stage 1 — cheap concept-presence check.** Every gold concept (medication, condition, lab) must appear somewhere in the note text. This catches note-writer hallucinations and template gaps before spending API tokens. Reject rate here: ~5%.

**Stage 2 — teacher recovery.** Claude Sonnet extracts FHIR from the note as if it were the model being trained. The extraction is scored against the gold using the same harness used at eval time. If resource F1 ≥ 0.8, we admit the pair. If not — if even a strong extractor can't recover the gold from the note — we throw it away.

This is the key insight: **the training target is always the gold**, not the teacher's output. Claude is a *consistency filter*, not a label source. We're asking: "Is this note learnable?" not "What did Claude think?".

## Why it matters

The alternative — filter-free generation — produces examples where the note says "aspirin 81 mg" but the gold FHIR has a lisinopril resource. The model learns a confused mapping. Reject-sampling enforces a simple invariant: every training pair is faithful, and every faithful pair is learnable.

In practice, the accept rate with both filters and the Claude note-writer is around 70–80%. That means 20–30% of generated pairs get discarded. That's the right trade-off: a smaller, cleaner dataset outperforms a larger, noisy one.

## The offline path

Reject-sampling requires an API key for the teacher filter. When no key is set, the pipeline falls back to template notes + cheap filter only — no teacher, no Claude, fully deterministic. This is useful for CI, for quick iteration, and for environments where no external calls are allowed.

```bash
python -m data.build --n 50 --offline --seed 42
```

With a key:
```bash
ANTHROPIC_API_KEY=sk-... python -m data.build --n 200 --seed 42
```

The `--offline` flag produces lower-quality notes (template text vs. Claude-written prose) but the gold FHIR is identical, and the cheap filter still catches the worst cases.

## The dataset record format

Each accepted example is stored as:

```json
{
  "case": { "id": "case-00042", "note": "...", "gold": { "phi_spans": [...], "resources": [...] } },
  "prediction": { ... },   // teacher's extraction (for analysis, not training)
  "eval": { "score": 0.91, "resource_f1": 0.88, ... },
  "provenance": { "teacher_model": "claude-sonnet-4-6", "note_mode": "claude", "git_sha": "a8fb38c", "seed": 42 }
}
```

The teacher prediction and eval score are stored alongside the gold — not as training targets, but for auditing and debugging. Every record knows which model wrote it, which seed generated it, and which git commit produced it.

## Connection to eval

The reject-sampling threshold (F1 ≥ 0.8) uses `evals.run_eval` — the same function used at eval time, at benchmark time, and inside the agent's evaluate node. This isn't coincidental. One oracle everywhere means the training filter and the deployment gate are semantically identical. A model that learns to pass the training filter is learning to pass the deployment gate.
