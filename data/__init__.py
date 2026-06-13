"""Specced data pipeline — PHI-safe synthetic (note -> gold FHIR) generation.

Flow: generate (synthetic patient + gold FHIR + synthetic PHI) -> notes (Claude
note-writer / template) -> filter (concept-presence + teacher reject-sampling via
the eval harness) -> build (curated train/val/held_out jsonl). No real PHI, ever.
"""
