"""Reject-sampling: keep only (note, gold) pairs that are faithful and learnable.

Two stages:
1. concept-presence (cheap, no API) — every gold concept's text/synonym must
   appear in the note, else the note dropped a fact.
2. teacher recovery (Claude) — a strong extractor reads the note and the eval
   harness scores its output against gold; keep the pair only if recovery is high.

The teacher's prediction + eval are returned for storage/analysis; the training
target remains the gold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from evals.case import Case, Prediction
from evals.run_eval import run_eval
from . import teacher

ACCEPT_F1 = 0.8  # min teacher resource-F1 to accept a pair


@dataclass
class AcceptResult:
    accepted: bool
    reason: str
    teacher_pred: Optional[Prediction] = None
    eval: Optional[dict[str, Any]] = None


def concept_presence(note: str, concept_variants: list[list[str]]) -> list[int]:
    """Return indices of gold resources whose concept text is absent from the note."""
    low = note.lower()
    missing = []
    for i, variants in enumerate(concept_variants):
        if not any(v.lower() in low for v in variants if v):
            missing.append(i)
    return missing


def accept(
    case: Case, concept_variants: list[list[str]], *, use_teacher: bool
) -> AcceptResult:
    missing = concept_presence(case.note, concept_variants)
    if missing:
        return AcceptResult(False, f"note missing {len(missing)} concept(s)")

    if not use_teacher:
        return AcceptResult(True, "concept-presence")

    pred = teacher.extract_via_claude(case.note, case.target_resources)
    score, report = run_eval(case, pred, use_judge=False)
    if (score.resource_f1 or 0.0) >= ACCEPT_F1:
        return AcceptResult(True, f"teacher f1={score.resource_f1}", pred, report["score"])
    return AcceptResult(False, f"teacher recovery low (f1={score.resource_f1})", pred, report["score"])
