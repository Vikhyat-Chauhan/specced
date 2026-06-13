"""Pydantic models for extraction cases and model predictions.

The JSON Schema in specs/case.schema.json is the canonical contract; these
models are the runtime validators (and what the harness/agent pass around).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

PHI_TYPES = {
    "NAME", "DATE", "AGE", "LOCATION", "ID", "PHONE", "EMAIL", "ORG",
    "PROFESSION", "OTHER",
}


class PhiSpan(BaseModel):
    text: str
    type: str
    start: Optional[int] = None
    end: Optional[int] = None


class Prediction(BaseModel):
    """What the model returns for one note."""

    phi_spans: list[PhiSpan] = Field(default_factory=list)
    resources: list[dict[str, Any]] = Field(default_factory=list)


class Gold(Prediction):
    """Reference answer — same shape as a prediction."""


class Case(BaseModel):
    id: str
    note: str
    fhir_version: str = "R4"
    target_resources: list[str] = Field(default_factory=list)
    deidentify: bool = False
    gold: Optional[Gold] = None


def load_case(path: str | Path) -> Case:
    return Case.model_validate(json.loads(Path(path).read_text()))


def load_prediction(path: str | Path) -> Prediction:
    return Prediction.model_validate(json.loads(Path(path).read_text()))
