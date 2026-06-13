"""Single entry point: evaluate one (case, prediction) through the harness.

Reused by the CLI, the data reject-sampler, and the agent's EVALUATE node:
FHIR validity -> field-F1 -> de-id recall -> optional clinical judge -> report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .case import Case, Prediction
from .score import score as score_fn, EvalScore
from .judges import load_clinical_judge
from .report import build_report


def run_eval(
    case: Case,
    pred: Prediction,
    *,
    use_judge: bool = True,
    ts: Optional[str] = None,
) -> tuple[EvalScore, dict[str, Any]]:
    ts = ts or datetime.now(timezone.utc).isoformat()
    score = score_fn(case, pred)

    judge_out: Optional[dict[str, Any]] = None
    if use_judge:
        judge = load_clinical_judge()
        if judge is not None:
            judge_out = judge(case, pred)

    report = build_report(case, pred, score, ts, judge=judge_out)
    return score, report
