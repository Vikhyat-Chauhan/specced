"""AgentState — the single dict threaded through every LangGraph node."""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from evals.case import Case, Prediction
from evals.score import EvalScore


class AgentState(TypedDict):
    case: Case
    prediction: Optional[Prediction]
    eval_score: Optional[EvalScore]
    eval_report: dict[str, Any]
    refine_count: int
    max_refines: int
    done: bool
    retrieved_hints: str   # formatted code hints from RAG (empty until US-5 wired)
    error_context: str     # human-readable FHIR errors fed back into act on refines


def initial_state(case: Case, *, max_refines: int = 3) -> AgentState:
    return AgentState(
        case=case,
        prediction=None,
        eval_score=None,
        eval_report={},
        refine_count=0,
        max_refines=max_refines,
        done=False,
        retrieved_hints="",
        error_context="",
    )
