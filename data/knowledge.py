"""A small, curated clinical knowledge base with real terminology codes.

Authored by us (not bulk licensed terminology) so synthetic gold FHIR carries
correct, recognizable codes: medications -> RxNorm, conditions -> ICD-10-CM +
SNOMED CT, labs/vitals -> LOINC (+ UCUM units and plausible value ranges),
allergies -> substance + reactions. Each concept lists synonyms so the cheap
concept-presence filter and the note-writer can vary surface text.

Systems:
  RxNorm  http://www.nlm.nih.gov/research/umls/rxnorm
  ICD-10  http://hl7.org/fhir/sid/icd-10-cm
  SNOMED  http://snomed.info/sct
  LOINC   http://loinc.org
  UCUM    http://unitsofmeasure.org
"""

from __future__ import annotations

from dataclasses import dataclass, field

RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"


@dataclass(frozen=True)
class Medication:
    display: str
    rxnorm: str
    doses: tuple[str, ...]
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Condition:
    display: str
    icd10: str
    snomed: str
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Lab:
    display: str
    loinc: str
    unit: str           # human unit shown in the note
    ucum: str           # UCUM code for valueQuantity
    low: float
    high: float
    decimals: int = 0
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Allergen:
    substance: str
    reactions: tuple[str, ...]
    synonyms: tuple[str, ...] = ()


MEDICATIONS: list[Medication] = [
    Medication("lisinopril", "29046", ("10 mg daily", "20 mg daily", "5 mg daily")),
    Medication("metformin", "6809", ("500 mg twice daily", "1000 mg twice daily", "850 mg daily")),
    Medication("atorvastatin", "83367", ("40 mg daily", "20 mg daily", "80 mg nightly"), ("Lipitor",)),
    Medication("amlodipine", "17767", ("5 mg daily", "10 mg daily")),
    Medication("omeprazole", "7646", ("20 mg daily", "40 mg daily"), ("Prilosec",)),
    Medication("levothyroxine", "10582", ("50 mcg daily", "75 mcg daily", "100 mcg daily"), ("Synthroid",)),
    Medication("albuterol", "435", ("2 puffs every 4-6 hours as needed",), ("salbutamol",)),
    Medication("metoprolol", "6918", ("25 mg twice daily", "50 mg twice daily")),
    Medication("losartan", "52175", ("50 mg daily", "100 mg daily")),
    Medication("gabapentin", "25480", ("300 mg three times daily", "600 mg at bedtime")),
    Medication("sertraline", "36437", ("50 mg daily", "100 mg daily"), ("Zoloft",)),
    Medication("furosemide", "4603", ("20 mg daily", "40 mg twice daily"), ("Lasix",)),
    Medication("hydrochlorothiazide", "5487", ("25 mg daily", "12.5 mg daily"), ("HCTZ",)),
    Medication("aspirin", "1191", ("81 mg daily",), ("ASA",)),
    Medication("insulin glargine", "274783", ("20 units at bedtime", "30 units at bedtime"), ("Lantus",)),
]

CONDITIONS: list[Condition] = [
    Condition("hypertension", "I10", "38341003", ("high blood pressure", "HTN")),
    Condition("type 2 diabetes mellitus", "E11.9", "44054006", ("type 2 diabetes", "T2DM", "DM2")),
    Condition("hyperlipidemia", "E78.5", "55822004", ("high cholesterol",)),
    Condition("asthma", "J45.909", "195967001", ()),
    Condition("gastroesophageal reflux disease", "K21.9", "235595009", ("GERD", "acid reflux")),
    Condition("hypothyroidism", "E03.9", "40930008", ("underactive thyroid",)),
    Condition("major depressive disorder", "F32.9", "370143000", ("depression",)),
    Condition("chronic obstructive pulmonary disease", "J44.9", "13645005", ("COPD",)),
    Condition("osteoarthritis", "M19.90", "396275006", ("OA", "degenerative joint disease")),
    Condition("atrial fibrillation", "I48.91", "49436004", ("AFib", "A-fib")),
    Condition("chronic kidney disease", "N18.9", "709044004", ("CKD",)),
    Condition("generalized anxiety disorder", "F41.1", "21897009", ("anxiety", "GAD")),
]

# Scalar labs/vitals (single valueQuantity). Blood pressure is handled specially
# in generate.py (two numbers -> one Observation valueString "148/92 mmHg").
LABS: list[Lab] = [
    Lab("heart rate", "8867-4", "bpm", "/min", 55, 105, 0, ("pulse", "HR")),
    Lab("body temperature", "8310-5", "°F", "[degF]", 97, 101, 1, ("temp",)),
    Lab("hemoglobin A1c", "4548-4", "%", "%", 5.2, 10.5, 1, ("A1c", "HbA1c")),
    Lab("glucose", "2339-0", "mg/dL", "mg/dL", 80, 240, 0, ("blood sugar", "blood glucose")),
    Lab("LDL cholesterol", "13457-7", "mg/dL", "mg/dL", 70, 190, 0, ("LDL",)),
    Lab("body weight", "29463-7", "kg", "kg", 55, 120, 1, ("weight",)),
    Lab("creatinine", "2160-0", "mg/dL", "mg/dL", 0.6, 2.4, 1, ("Cr",)),
    Lab("thyroid stimulating hormone", "3016-3", "mIU/L", "m[IU]/L", 0.4, 8.0, 1, ("TSH",)),
]

ALLERGENS: list[Allergen] = [
    Allergen("penicillin", ("rash", "hives", "anaphylaxis"), ("PCN",)),
    Allergen("sulfa", ("rash", "itching"), ("sulfonamides",)),
    Allergen("peanut", ("anaphylaxis", "swelling")),
    Allergen("latex", ("contact dermatitis", "rash")),
    Allergen("codeine", ("nausea", "hives")),
    Allergen("shellfish", ("hives", "swelling")),
    Allergen("ibuprofen", ("hives", "wheezing"), ("NSAIDs", "Advil")),
]


def all_concept_texts(concept) -> list[str]:
    """Display + synonyms (for the cheap concept-presence filter / note variety)."""
    if isinstance(concept, Allergen):
        return [concept.substance, *concept.synonyms]
    return [concept.display, *getattr(concept, "synonyms", ())]
