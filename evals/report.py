"""Build and persist an eval report (JSON) under evals/reports/<case-id>/."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from .case import Case, Prediction
from .fhir_validate import MODE as FHIR_MODE
from .score import EvalScore

HARNESS_VERSION = "0.2.0"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_report(
    case: Case,
    pred: Prediction,
    score: EvalScore,
    ts: str,
    judge: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "score": score.to_dict(),
        "judge": judge,
        "prediction": pred.model_dump(),
        "provenance": {
            "git_sha": _git_sha(),
            "ts": ts,
            "harness_version": HARNESS_VERSION,
            "fhir_validator": FHIR_MODE,
        },
    }


def write_report(report: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> Path:
    out = reports_dir / report["case_id"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
