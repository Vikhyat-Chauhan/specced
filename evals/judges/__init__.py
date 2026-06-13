"""Optional LLM judges. The objective oracle (FHIR validity + field-F1 + de-id
recall) is primary; the clinical judge is a secondary signal for borderline cases.

Returns None when ANTHROPIC_API_KEY is unset, so the harness still runs.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional


def load_clinical_judge() -> Optional[Callable[..., dict[str, Any]]]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from .clinical import clinical_judge

        return clinical_judge
    except Exception:
        return None
