"""Claude-backed clinical-correctness judge (secondary signal).

Checks the extracted FHIR against the source note for hallucinated or incorrect
clinical facts (meds, doses, values, conditions) that field-F1 against gold can
miss. Model IDs per the claude-api skill: default Opus 4.8; override with
SPECCED_JUDGE_MODEL. Only invoked when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..case import Case, Prediction

MODEL = os.environ.get("SPECCED_JUDGE_MODEL", "claude-opus-4-8")

_SYSTEM = (
    "You are a clinical informatics reviewer. Given a clinical note and FHIR "
    "resources extracted from it, judge whether the extraction is clinically "
    "faithful to the note. Penalize any value NOT supported by the note "
    "(hallucinated meds, doses, codes, observations) and any clinically wrong "
    "coding. Reward complete, accurate, well-coded extraction.\n\n"
    'Respond with ONLY JSON: {"clinical_correctness": <0..1>, '
    '"hallucinations": [<strings>], "rationale": "<one sentence>"}'
)


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def clinical_judge(case: Case, pred: Prediction) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    user = (
        f"NOTE:\n{case.note}\n\n"
        f"EXTRACTED RESOURCES:\n{json.dumps(pred.resources, indent=2)}\n\n"
        "Judge clinical faithfulness against the rubric."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    out = _extract_json(text)
    return {
        "clinical_correctness": float(out.get("clinical_correctness", 0.0)),
        "hallucinations": out.get("hallucinations", []),
        "rationale": out.get("rationale", ""),
    }
