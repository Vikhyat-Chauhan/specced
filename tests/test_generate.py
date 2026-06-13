"""Generator self-consistency: synthetic gold is valid FHIR and scores perfectly."""

import random

from faker import Faker

from evals.case import Case, Gold, Prediction
from evals.fhir_validate import validate_resource
from evals.score import score
from data.generate import synth_case
from data.notes import write_note


def _build_case(seed: int) -> Case:
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)
    sc = synth_case(rng, fake)
    note, spans = write_note(sc, mode="template")  # offline, no API key
    return Case(
        id=f"t{seed}",
        note=note,
        fhir_version="R4",
        target_resources=sc.target_resources,
        deidentify=True,
        gold=Gold(phi_spans=spans, resources=sc.resources),
    )


def test_generated_gold_is_valid_and_self_consistent():
    for seed in range(5):
        case = _build_case(seed)
        for r in case.gold.resources:
            ok, err = validate_resource(r, "R4")
            assert ok, f"seed {seed}: invalid gold resource: {err}"
        pred = Prediction(phi_spans=case.gold.phi_spans, resources=case.gold.resources)
        s = score(case, pred)
        assert s.passed and s.score == 1.0, f"seed {seed}: gold did not score perfectly"
