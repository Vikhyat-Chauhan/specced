"""Anthropic teacher used by the data pipeline for two jobs:

1. note_via_claude  — write a realistic clinical note from structured facts.
2. extract_via_claude — extract FHIR + PHI spans from a note (the reject-sampling
   "can a strong extractor recover the gold?" check).

Model from SPECCED_TEACHER_MODEL (default claude-sonnet-4-6 for bulk; per the
claude-api skill). Adaptive thinking. No key -> available() is False and the
caller uses the offline/template path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from evals.case import Prediction

MODEL = os.environ.get("SPECCED_TEACHER_MODEL", "claude-sonnet-4-6")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    import anthropic

    return anthropic.Anthropic()


def _complete(system: str, user: str, max_tokens: int = 2048) -> str:
    resp = _client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


_NOTE_SYSTEM = (
    "You write short, realistic free-text clinical notes (1 short paragraph). "
    "Write the way a busy clinician would — natural phrasing, light abbreviation. "
    "You MUST include every provided identifier VERBATIM (exact characters) and "
    "mention every clinical fact (conditions, medications with doses, vitals/labs "
    "with values, allergies). Do not invent any clinical facts not provided. "
    "Output ONLY the note text."
)


def note_via_claude(facts: list[dict[str, Any]], pii: dict[str, str]) -> str:
    user = (
        "Identifiers to include verbatim:\n"
        + "\n".join(f"- {k}: {v}" for k, v in pii.items())
        + "\n\nClinical facts to convey:\n"
        + json.dumps(facts, indent=2)
        + "\n\nWrite the note."
    )
    return _complete(_NOTE_SYSTEM, user).strip()


_EXTRACT_SYSTEM = (
    "Extract structured FHIR (R4) from the clinical note. Return ONLY JSON: "
    '{"phi_spans": [{"text": "...", "type": "NAME|DATE|AGE|ID|PHONE|LOCATION|OTHER"}], '
    '"resources": [ <FHIR resources> ]}. '
    "Use resourceType Condition, MedicationStatement, Observation, AllergyIntolerance. "
    "Code with standard terminologies where possible (RxNorm/ICD-10/SNOMED/LOINC). "
    "Extract only what the note supports — do not invent values."
)


def extract_via_claude(note: str, target_resources: list[str]) -> Prediction:
    user = (
        (f"Target resource types: {', '.join(target_resources)}\n\n" if target_resources else "")
        + f"NOTE:\n{note}\n\nReturn the JSON."
    )
    out = _extract_json(_complete(_EXTRACT_SYSTEM, user))
    return Prediction.model_validate(
        {"phi_spans": out.get("phi_spans", []), "resources": out.get("resources", [])}
    )
