"""Scoring: FHIR validity, resource/field matching, de-id recall, aggregate.

Transparent and gate-aware:
- Invalid FHIR resources can't be trusted -> excluded from matching and counted
  against precision (reported separately as validity_rate).
- Resource P/R/F1: match predicted resources to gold by (resourceType, primary
  concept). Field accuracy: over matched pairs, fraction of secondary fields
  (dosage/value/status) correct.
- De-id recall: gold PHI spans caught (recall is the safety-critical metric).
- Aggregate weights present components: resource-F1 0.5, field-acc 0.3, de-id 0.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .case import Case, Prediction
from .fhir_validate import validate_resource

PASS_THRESHOLD = 0.7
DEID_MIN_RECALL = 0.95  # safety gate when deidentify is requested


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _text_of(node: Any) -> str:
    """Best-effort human text from a FHIR CodeableConcept / string / reference."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("text"):
            return str(node["text"])
        for coding in node.get("coding", []) or []:
            if coding.get("display"):
                return str(coding["display"])
            if coding.get("code"):
                return str(coding["code"])
        if node.get("display"):
            return str(node["display"])
        if node.get("reference"):
            return str(node["reference"])
    return ""


def _medication_text(r: dict[str, Any]) -> str:
    for k, v in r.items():
        if k.startswith("medication"):
            return _text_of(v)
    return ""


def _value_text(r: dict[str, Any]) -> str:
    for k, v in r.items():
        if k == "valueString":
            return str(v)
        if k == "valueQuantity" and isinstance(v, dict):
            return _norm(f"{v.get('value', '')} {v.get('unit', '')}")
        if k.startswith("value"):
            return _text_of(v)
    return ""


def _dosage_text(r: dict[str, Any]) -> str:
    dosage = r.get("dosage")
    if isinstance(dosage, list):
        return " ".join(_text_of(d) for d in dosage)
    return _text_of(dosage)


def project(r: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    """(resourceType, primary_concept, secondary_fields) — all normalized."""
    rt = r.get("resourceType", "")
    if rt in ("MedicationStatement", "MedicationRequest"):
        key, sec = _medication_text(r), {"dosage": _dosage_text(r)}
    elif rt == "Observation":
        key, sec = _text_of(r.get("code")), {"value": _value_text(r)}
    elif rt == "Condition":
        key, sec = _text_of(r.get("code")), {"status": _text_of(r.get("clinicalStatus"))}
    elif rt == "AllergyIntolerance":
        key, sec = _text_of(r.get("code") or r.get("substance")), {}
    else:
        key, sec = _text_of(r.get("code")), {}
    return rt, _norm(key), {k: _norm(v) for k, v in sec.items() if _norm(v)}


@dataclass
class EvalScore:
    score: float
    passed: bool
    validity_rate: Optional[float]
    resource_precision: Optional[float]
    resource_recall: Optional[float]
    resource_f1: Optional[float]
    field_accuracy: Optional[float]
    deid_recall: Optional[float]
    invalid_resources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score(case: Case, pred: Prediction) -> EvalScore:
    reasons: list[str] = []
    gold = case.gold

    # --- FHIR validity gate ---
    valid_pred: list[dict[str, Any]] = []
    invalid: list[str] = []
    for r in pred.resources:
        ok, err = validate_resource(r, case.fhir_version)
        if ok:
            valid_pred.append(r)
        else:
            invalid.append(f"{r.get('resourceType', '?')}: {err}")
    validity_rate = (len(valid_pred) / len(pred.resources)) if pred.resources else None
    if invalid:
        reasons.append(f"{len(invalid)} invalid FHIR resource(s)")

    if gold is None or gold.resources is None:
        # Unlabeled inference case: only validity is computable.
        reasons.append("no gold; only FHIR validity scored")
        return EvalScore(
            score=validity_rate or 0.0, passed=False, validity_rate=validity_rate,
            resource_precision=None, resource_recall=None, resource_f1=None,
            field_accuracy=None, deid_recall=None, invalid_resources=invalid, reasons=reasons,
        )

    # --- Resource matching (only valid predictions are eligible) ---
    gold_proj = [project(r) for r in gold.resources]
    pred_proj = [project(r) for r in valid_pred]
    used = [False] * len(pred_proj)
    tp = 0
    field_correct = field_total = 0
    for grt, gkey, gsec in gold_proj:
        match_idx = next(
            (i for i, (prt, pkey, _) in enumerate(pred_proj)
             if not used[i] and prt == grt and pkey == gkey and gkey),
            None,
        )
        if match_idx is None:
            continue
        used[match_idx] = True
        tp += 1
        _, _, psec = pred_proj[match_idx]
        for k, gv in gsec.items():
            field_total += 1
            if psec.get(k) == gv:
                field_correct += 1

    fp = len(valid_pred) - tp + len(invalid)  # invalid preds count against precision
    fn = len(gold.resources) - tp
    precision, recall, f1 = _prf(tp, fp, fn)
    field_accuracy = (field_correct / field_total) if field_total else None

    # --- De-id recall ---
    deid_recall: Optional[float] = None
    if case.deidentify and gold.phi_spans:
        gold_spans = {(s.type, _norm(s.text)) for s in gold.phi_spans}
        pred_spans = {(s.type, _norm(s.text)) for s in pred.phi_spans}
        caught = len(gold_spans & pred_spans)
        deid_recall = caught / len(gold_spans)
        if deid_recall < DEID_MIN_RECALL:
            reasons.append(f"de-id recall {deid_recall:.2f} < {DEID_MIN_RECALL} (missed PHI)")

    # --- Aggregate (renormalize over present components) ---
    comps: list[tuple[float, float]] = [(0.5, f1)]
    if field_accuracy is not None:
        comps.append((0.3, field_accuracy))
    if deid_recall is not None:
        comps.append((0.2, deid_recall))
    wsum = sum(w for w, _ in comps)
    agg = sum(w * c for w, c in comps) / wsum if wsum else 0.0

    passed = agg >= PASS_THRESHOLD and (deid_recall is None or deid_recall >= DEID_MIN_RECALL)

    return EvalScore(
        score=round(agg, 3), passed=passed, validity_rate=validity_rate,
        resource_precision=round(precision, 3), resource_recall=round(recall, 3),
        resource_f1=round(f1, 3),
        field_accuracy=None if field_accuracy is None else round(field_accuracy, 3),
        deid_recall=None if deid_recall is None else round(deid_recall, 3),
        invalid_resources=invalid, reasons=reasons,
    )
