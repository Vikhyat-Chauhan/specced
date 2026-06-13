"""Turn a SynthCase into a clinical note + gold PHI spans.

Two modes:
- "claude"   — Claude writes a realistic note from the facts (default online).
- "template" — deterministic fill-in note (offline / no-key, always PHI-faithful).

In both modes, PHI spans are computed by locating each synthetic identifier in the
final note text, so gold spans are exact. Claude mode verifies every identifier is
present (else raises NoteError so the case is rejected/retried).
"""

from __future__ import annotations

from typing import Any

from evals.case import PhiSpan
from . import teacher
from .generate import SynthCase


class NoteError(Exception):
    """Raised when a generated note dropped a required identifier."""


def _find_spans(note: str, pii: dict[str, str]) -> list[PhiSpan]:
    spans: list[PhiSpan] = []
    for ptype, value in pii.items():
        idx = note.find(value)
        if idx >= 0:
            spans.append(PhiSpan(text=value, type=ptype, start=idx, end=idx + len(value)))
    return spans


def _template_note(sc: SynthCase) -> str:
    p = sc.pii
    lines = [
        f"{p['NAME']}, a {p['AGE']}-year-old patient (MRN {p['ID']}), was seen on "
        f"{p['DATE']}. Contact: {p['PHONE']}."
    ]
    for f in sc.note_facts:
        kind = f["kind"]
        if kind == "condition":
            lines.append(f"History of {f['name']}.")
        elif kind == "medication":
            lines.append(f"Currently taking {f['name']} {f['dose']}.")
        elif kind == "lab":
            if f["name"] == "blood pressure":
                lines.append(f"BP {f['value']} {f['unit']}.")
            else:
                lines.append(f"{f['name'].capitalize()} {f['value']} {f['unit']}.")
        elif kind == "allergy":
            lines.append(f"Allergic to {f['substance']} ({f['reaction']}).")
    return " ".join(lines)


def write_note(sc: SynthCase, mode: str = "claude") -> tuple[str, list[PhiSpan]]:
    if mode == "claude":
        note = teacher.note_via_claude(sc.note_facts, sc.pii)
        missing = [v for v in sc.pii.values() if v not in note]
        if missing:
            raise NoteError(f"note dropped identifiers: {missing}")
    else:
        note = _template_note(sc)
    return note, _find_spans(note, sc.pii)
