"""Prompt formatting for note → {phi_spans, resources} fine-tuning and inference.

Uses Qwen2.5's ChatML token format. Shared by training (train_qlora.py) and
serving (serve/client.py) so the inference prefix exactly matches training.
"""

from __future__ import annotations

import json

from evals.case import Case, Gold

SYSTEM_PROMPT = (
    "You are a clinical FHIR extraction model. "
    "Given a free-text clinical note and a list of target FHIR resource types, "
    "extract all PHI spans and FHIR R4 resources. "
    "Return ONLY a single JSON object:\n"
    '{"phi_spans": [{"text": "...", "type": "NAME|DATE|AGE|LOCATION|ID|PHONE|EMAIL|ORG|PROFESSION|OTHER", '
    '"start": <int>, "end": <int>}], '
    '"resources": [<valid FHIR R4 resource>, ...]}\n'
    "Rules:\n"
    "- Every resource must be valid FHIR R4 (include required fields).\n"
    "- Bind to standard terminologies (SNOMED/RxNorm/ICD-10/LOINC) where possible.\n"
    "- De-id recall is safety-critical: flag ALL HIPAA Safe Harbor PHI — when in doubt, over-flag.\n"
    "- Output only values the note explicitly supports. No hallucinated meds, doses, or codes."
)

_DEFAULT_RESOURCES = "Condition, MedicationStatement, Observation, AllergyIntolerance"


def _user_message(note: str, target_resources: list[str]) -> str:
    types = ", ".join(target_resources) if target_resources else _DEFAULT_RESOURCES
    return f"Clinical note:\n{note}\n\nTarget resource types: {types}"


def format_prompt(case: Case, gold: Gold) -> str:
    """Full ChatML string for training — includes the gold completion."""
    gold_json = json.dumps(
        {
            "phi_spans": [s.model_dump() for s in gold.phi_spans],
            "resources": gold.resources,
        },
        indent=2,
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{_user_message(case.note, case.target_resources)}<|im_end|>\n"
        f"<|im_start|>assistant\n{gold_json}<|im_end|>"
    )


def format_inference_prompt(note: str, target_resources: list[str]) -> str:
    """ChatML prefix for generation — model fills in the assistant turn."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{_user_message(note, target_resources)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
