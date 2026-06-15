# Self-refine on schema errors

The most underrated thing about using a structured output format like FHIR is that **validation errors are machine-readable**. When the model emits an invalid resource, we know exactly what's wrong: "Condition: missing required field 'subject'", "MedicationStatement: medicationCodeableConcept must have at least one coding". That's a precise, actionable error message — not a vague "output looks bad".

Self-refine exploits this. Instead of accepting the first output, we validate it, format the errors into natural language, and feed them back into the prompt for a second attempt.

## The loop

```
note + target resources
    ↓
PLAN ──▶ RETRIEVE ──▶ ACT ──▶ EVALUATE ──▶ DONE (passed)
                       ↑            │
                       └── errors ──┘ (refine, if not passed and count < max)
```

The LangGraph `AgentState` carries:
- `prediction` — current best extraction
- `eval_score` — the last EvalScore from the harness
- `error_context` — formatted error string
- `refine_count` — how many iterations we've done
- `done` — True when passed or ceiling hit

## How errors become prompt context

After each failed evaluation, the evaluate node formats the errors:

```python
errors = list(score.invalid_resources) + list(score.reasons)
error_ctx = "\n".join(f"- {e}" for e in errors)
```

`score.invalid_resources` contains things like:
```
Condition: missing required field 'subject'
MedicationStatement: 'status' is a required property
```

`score.reasons` contains aggregate failures like:
```
2 invalid FHIR resource(s)
de-id recall 0.67 < 0.95 (missed PHI)
```

On the next iteration, the act node appends this to the user message:

```
Clinical note:
[original note]

Target resource types: Condition, MedicationStatement, ...

Relevant standard codes to use where they match:
- lisinopril → RxNorm 29046 (...)
- hypertension → ICD-10 I10 (...)

Your previous attempt had these issues — please fix them:
- Condition: missing required field 'subject'
- MedicationStatement: 'status' is a required property
- 2 invalid FHIR resource(s)
```

The model sees the original note (unchanged), the standard codes (from RAG), and the specific errors it needs to fix. It doesn't need to regenerate everything — just fix the broken parts. In practice, when the base model produces invalid FHIR, a single refine iteration almost always fixes it (the issues are usually missing required fields, not conceptual errors).

## Results

On the 20 held-out cases:

| Variant | Passed | F1 | De-id |
|---|---|---|---|
| Fine-tuned, single shot | 20/20 | 0.992 | 1.000 |
| Fine-tuned + agent (max 3 refines) | 20/20 | **1.000** | 1.000 |

The fine-tuned model was already excellent — 20/20 passing, F1=0.992. The agent's self-refine closed the remaining 0.008 F1 gap. On cases where the fine-tuned model's first attempt has minor field accuracy issues, the agent catches and fixes them.

The agent's overhead: +0.5 s/note on average. On the 20 held-out cases with a well-trained adapter, most cases passed on the first attempt and the conditional edge routed directly to END without a refine iteration.

## When self-refine is most valuable

Self-refine matters most at inference time when:
1. The note is out-of-distribution (unusual phrasing, rare conditions, abbreviations not in training data)
2. The first attempt produces structurally valid JSON but with missing required FHIR fields
3. De-id recall fails (a PHI span was missed) — the model can be told explicitly which type of PHI it missed

For cases where the first attempt is completely unparseable (no JSON at all), self-refine with the same prompt won't help much — you need a different strategy (e.g., temperature perturbation). In Specced, the fine-tuned model almost never produces unparseable output, so self-refine on structural errors is effective.

## Implementation note: one model load per run

Reloading a 4-bit 7B model for every refine iteration would add ~5 seconds per iteration (IO + deserialization). Instead, Specced loads the model once before the graph runs, shares it across all iterations via a module-level singleton in `agent/nodes/act.py`, and unloads it when `agent.run` completes:

```python
# agent/run.py
act_node.load_backend(adapter_path)
try:
    graph = build_graph()
    final = graph.invoke(initial_state(case, max_refines=3))
finally:
    act_node.unload_backend()
```

This keeps the model warm in VRAM across iterations — the refine loop has no GPU IO overhead.
