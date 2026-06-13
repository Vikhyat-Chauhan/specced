"""Specced eval harness — clinical -> FHIR extraction scoring.

Pipeline: FHIR schema validity (gate) -> field-level F1 -> de-id recall ->
optional clinical judge -> aggregate score + report. The same `run_eval`
entry point is reused by the data reject-sampler and the agent's EVALUATE node.
"""
