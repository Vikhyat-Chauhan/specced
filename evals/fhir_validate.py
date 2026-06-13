"""FHIR schema-validity gate (version-aware).

Uses the `fhir.resources` library when it imports on this interpreter (real FHIR
validation, pinned to the case's FHIR version); otherwise falls back to a
lightweight required-field check so the harness is runnable without the dep.

Notes:
- `fhir.resources` >= 8 ships R5 models at the top level and R4/R4B models under
  the `fhir.resources.R4B` subpackage; we map "R4"/"R4B" -> R4B.
- On Python < 3.10 the model modules need the `eval_type_backport` package to
  import; if anything fails we degrade to the fallback rules (MODE == "fallback").
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Optional

# Minimal required-field rules (R4) for the fallback validator.
_REQUIRED: dict[str, list[str]] = {
    "Patient": [],
    "Condition": ["subject"],
    "MedicationStatement": ["status", "subject", "medication[x]"],
    "MedicationRequest": ["status", "intent", "subject", "medication[x]"],
    "Observation": ["status", "code"],
    "AllergyIntolerance": ["patient"],
    "Procedure": ["status", "subject"],
    "Immunization": ["status", "vaccineCode", "patient", "occurrence[x]"],
    "DiagnosticReport": ["status", "code"],
    "Encounter": ["status", "class"],
    "FamilyMemberHistory": ["status", "patient", "relationship"],
}
KNOWN_TYPES = set(_REQUIRED)


def _fallback_validate(resource: dict[str, Any]) -> Optional[str]:
    rt = resource.get("resourceType")
    if not rt:
        return "missing resourceType"
    if rt not in KNOWN_TYPES:
        return f"unknown/unsupported resourceType: {rt}"
    for req in _REQUIRED[rt]:
        if req.endswith("[x]"):
            prefix = req[:-3]
            if not any(k.startswith(prefix) for k in resource):
                return f"{rt}: missing required choice element {req}"
        elif req not in resource:
            return f"{rt}: missing required field '{req}'"
    return None


def _release_pkg(fhir_version: str) -> Optional[str]:
    """Map a case FHIR version to a fhir.resources subpackage ('' = top level)."""
    v = (fhir_version or "R4").upper()
    if v in ("R4", "R4B"):
        return "R4B"
    if v == "R5":
        return ""  # R5 is the top-level package
    return None  # unknown -> not strict-validatable


@lru_cache(maxsize=None)
def _model_class(fhir_version: str, resource_type: str):
    pkg = _release_pkg(fhir_version)
    if pkg is None:
        return None
    base = f"fhir.resources.{pkg}." if pkg else "fhir.resources."
    try:
        mod = importlib.import_module(base + resource_type.lower())
        return getattr(mod, resource_type)
    except Exception:
        return None


def _strict_importable() -> bool:
    try:
        import fhir.resources  # noqa: F401

        # Smoke-test that versioned model modules actually import here.
        return _model_class("R4", "Condition") is not None
    except Exception:
        return False


MODE = "fhir.resources" if _strict_importable() else "fallback"


def validate_resource(
    resource: dict[str, Any], fhir_version: str = "R4"
) -> tuple[bool, Optional[str]]:
    """(ok, error). Strict (version-pinned) when fhir.resources is usable, else fallback."""
    if MODE == "fhir.resources":
        rt = resource.get("resourceType")
        if not rt:
            return False, "missing resourceType"
        cls = _model_class(fhir_version, rt)
        if cls is None:
            err = _fallback_validate(resource)  # unknown type / version -> fallback rule
            return err is None, err
        try:
            cls.model_validate(resource)
            return True, None
        except Exception as e:
            return False, f"{rt}: {str(e).splitlines()[0]}"
    err = _fallback_validate(resource)
    return err is None, err
