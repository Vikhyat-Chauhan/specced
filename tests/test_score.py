"""Eval-harness scoring: a happy path plus failure cases."""

import json
from pathlib import Path

from evals.case import Case, Prediction
from evals.score import score, DEID_MIN_RECALL

ROOT = Path(__file__).resolve().parents[1]
CASE = Case.model_validate(json.loads((ROOT / "specs/examples/cardio-visit.json").read_text()))


def test_gold_as_prediction_is_perfect():
    pred = Prediction(phi_spans=CASE.gold.phi_spans, resources=CASE.gold.resources)
    s = score(CASE, pred)
    assert s.passed
    assert s.score == 1.0
    assert s.validity_rate == 1.0
    assert s.deid_recall == 1.0


def test_invalid_resource_is_flagged():
    # AllergyIntolerance is missing the required `patient` field -> not valid FHIR.
    pred = Prediction(
        phi_spans=CASE.gold.phi_spans,
        resources=[{"resourceType": "AllergyIntolerance", "code": {"text": "penicillin"}}],
    )
    s = score(CASE, pred)
    assert s.invalid_resources, "expected the invalid resource to be reported"
    assert s.validity_rate is not None and s.validity_rate < 1.0


def test_missed_phi_fails_deid_gate():
    # Drop one gold PHI span -> recall below the safety threshold -> fail.
    pred = Prediction(phi_spans=CASE.gold.phi_spans[:-1], resources=CASE.gold.resources)
    s = score(CASE, pred)
    assert s.deid_recall is not None and s.deid_recall < DEID_MIN_RECALL
    assert not s.passed
