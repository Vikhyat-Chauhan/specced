# Contributing to Specced

Thanks for helping improve Specced. This guide covers local setup, conventions, and the checks every change must pass.

## Getting started

Follow the **Quick start** in the [README](README.md) to create a virtualenv and install the `evals` + `data` extras. For how the code is organized, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); for the task and data model, [docs/SPEC.md](docs/SPEC.md).

## Branching & commits

- Branch off `main`; open a pull request back into `main`.
- Write imperative, scoped commit subjects matching the existing history, e.g.
  `evals: pin FHIR validation to the case version`, `data: two-stage reject-sampling filter`.
- Keep each PR focused on one feature or fix.

## Quality gate

Run the full gate before every commit and PR:

```bash
make gate    # pytest + offline data build + eval CLI on the example case
```

A change is not ready to merge until the gate passes.

## Tests

- Tests live in `tests/` and run on [pytest](https://docs.pytest.org) (`make test`).
- Every feature ships a test for its core path **plus at least one failure case** (e.g. a schema-invalid resource is flagged, a missed PHI span drops de-id recall).

## Pull request checklist

Mirror the project's Definition of Done before requesting review:

- [ ] Every emitted FHIR resource **validates** against the case's FHIR version.
- [ ] No values unsupported by the note (no hallucinated meds, doses, or codes).
- [ ] When de-id is requested, **de-id recall ≥ 0.95** (a missed PHI span is a failure).
- [ ] **No real PHI** anywhere — data is synthetic or public de-identified only.
- [ ] A test covers the happy path and at least one failure case.
- [ ] `make gate` passes.

## Dependencies

Avoid adding new dependencies — exhaust the existing packages first. If one is truly necessary, call it out in the PR description and get sign-off. Keep heavy ML deps in the `train` / `agent` extras so the core install stays light.

## Reporting security issues

Do **not** open a public issue for a vulnerability. Follow the [Security Policy](SECURITY.md).
