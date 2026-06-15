"""Evaluate node — score the current prediction and decide whether to refine.

Uses the same evals.run_eval oracle as the data pipeline and benchmark — one
consistent gate everywhere. Sets state["done"] = True when the extraction passes
or the refine ceiling is hit.
"""

from __future__ import annotations

from agent.state import AgentState


def run(state: AgentState) -> AgentState:
    from evals.run_eval import run_eval

    case = state["case"]
    pred = state["prediction"]
    refine_count = state["refine_count"] + 1

    score, report = run_eval(case, pred, use_judge=False)

    # Build error context for the next act iteration.
    errors = list(score.invalid_resources) + list(score.reasons)
    error_ctx = "\n".join(f"- {e}" for e in errors) if errors else ""

    done = score.passed or refine_count >= state["max_refines"]

    print(
        f"[eval] score={score.score:.3f} passed={score.passed} "
        f"validity={score.validity_rate} f1={score.resource_f1} "
        f"deid={score.deid_recall} refines={refine_count}/{state['max_refines']}"
    )

    return {
        **state,
        "eval_score": score,
        "eval_report": report,
        "error_context": error_ctx,
        "refine_count": refine_count,
        "done": done,
    }
