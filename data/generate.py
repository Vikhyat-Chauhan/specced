"""Synthetic case generator: gold FHIR (real codes) + synthetic PHI.

`synth_case(rng, fake)` samples a patient slice from the curated KB, builds valid
R4 FHIR resources (the gold), and draws synthetic PHI (Faker). The note is written
later (notes.py); here we only produce the structured truth + the PHI strings to
embed and the per-resource concept variants used by the cheap filter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from faker import Faker

from . import knowledge as kb

PATIENT_REF = {"reference": "Patient/example"}


@dataclass
class SynthCase:
    resources: list[dict[str, Any]]
    pii: dict[str, str]                       # PHI type -> value to embed in the note
    target_resources: list[str]
    concept_variants: list[list[str]]         # aligned with resources (acceptable surface texts)
    note_facts: list[dict[str, Any]]          # structured facts for the note-writer/template


def _codeable(system: str, code: str, display: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


def _condition(c: kb.Condition) -> tuple[dict, list[str], dict]:
    res = {
        "resourceType": "Condition",
        "subject": PATIENT_REF,
        "clinicalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
        },
        "code": {
            "coding": [
                {"system": kb.ICD10, "code": c.icd10, "display": c.display},
                {"system": kb.SNOMED, "code": c.snomed, "display": c.display},
            ],
            "text": c.display,
        },
    }
    fact = {"kind": "condition", "name": c.display}
    return res, kb.all_concept_texts(c), fact


def _medication(m: kb.Medication, dose: str) -> tuple[dict, list[str], dict]:
    res = {
        "resourceType": "MedicationStatement",
        "status": "active",
        "subject": PATIENT_REF,
        "medicationCodeableConcept": _codeable(kb.RXNORM, m.rxnorm, m.display),
        "dosage": [{"text": dose}],
    }
    fact = {"kind": "medication", "name": m.display, "dose": dose}
    return res, kb.all_concept_texts(m), fact


def _observation(lab: kb.Lab, value: float) -> tuple[dict, list[str], dict]:
    num: Any = int(round(value)) if lab.decimals == 0 else round(value, lab.decimals)
    res = {
        "resourceType": "Observation",
        "status": "final",
        "code": _codeable(kb.LOINC, lab.loinc, lab.display),
        "valueQuantity": {"value": num, "unit": lab.unit, "system": kb.UCUM, "code": lab.ucum},
    }
    fact = {"kind": "lab", "name": lab.display, "value": str(num), "unit": lab.unit}
    return res, kb.all_concept_texts(lab), fact


def _blood_pressure(systolic: int, diastolic: int) -> tuple[dict, list[str], dict]:
    res = {
        "resourceType": "Observation",
        "status": "final",
        "code": _codeable(kb.LOINC, "85354-9", "blood pressure"),
        "valueString": f"{systolic}/{diastolic} mmHg",
    }
    return res, ["blood pressure", "BP"], {"kind": "lab", "name": "blood pressure", "value": f"{systolic}/{diastolic}", "unit": "mmHg"}


def _allergy(a: kb.Allergen, reaction: str) -> tuple[dict, list[str], dict]:
    res = {
        "resourceType": "AllergyIntolerance",
        "patient": PATIENT_REF,
        "code": {"text": a.substance},
    }
    fact = {"kind": "allergy", "substance": a.substance, "reaction": reaction}
    return res, kb.all_concept_texts(a), fact


def _pii(rng: random.Random, fake: Faker) -> dict[str, str]:
    age = rng.randint(35, 92)
    return {
        "NAME": fake.name(),
        "AGE": str(age),
        "DATE": fake.date_between(start_date="-2y", end_date="today").strftime("%m/%d/%Y"),
        "ID": fake.numerify("########"),
        "PHONE": fake.numerify("###-###-####"),
    }


def synth_case(rng: random.Random, fake: Faker) -> SynthCase:
    resources: list[dict] = []
    variants: list[list[str]] = []
    facts: list[dict] = []

    def add(triple: tuple[dict, list[str], dict]) -> None:
        res, var, fact = triple
        resources.append(res)
        variants.append(var)
        facts.append(fact)

    for c in rng.sample(kb.CONDITIONS, rng.randint(1, 3)):
        add(_condition(c))
    for m in rng.sample(kb.MEDICATIONS, rng.randint(1, 3)):
        add(_medication(m, rng.choice(m.doses)))
    for lab in rng.sample(kb.LABS, rng.randint(1, 3)):
        add(_observation(lab, rng.uniform(lab.low, lab.high)))
    if rng.random() < 0.6:
        add(_blood_pressure(rng.randint(110, 165), rng.randint(65, 100)))
    for a in rng.sample(kb.ALLERGENS, rng.randint(0, 2)):
        add(_allergy(a, rng.choice(a.reactions)))

    target = sorted({r["resourceType"] for r in resources})
    return SynthCase(
        resources=resources,
        pii=_pii(rng, fake),
        target_resources=target,
        concept_variants=variants,
        note_facts=facts,
    )
