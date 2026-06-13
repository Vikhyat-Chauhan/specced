# cli/ — `specced` command line

⬜ Planned — **US-7** in `STORIES.md`.

The user-facing entry point. Turns a clinical note into de-identified, coded, schema-valid FHIR via the agent.

```bash
specced extract ./note.txt          # note -> {phi_spans, FHIR} (+ eval report if gold given)
specced eval ./case.json            # score an existing prediction against gold
```

## Planned layout
- `main.py` — Typer app exposing `extract` / `eval` (wired in `pyproject.toml` `[project.scripts]`).

## Notes
- `extract` drives the `agent/` graph; `eval` calls the `evals/` harness (`python -m evals.cli`).
- Keep output friendly (rich): show validity, resource-F1, de-id recall, and the report path.
