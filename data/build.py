"""Build a curated (note -> gold FHIR) dataset.

    python -m data.build --n 50 [--out data/curated] [--offline] [--no-teacher] [--seed 0]

Pipeline: generate synthetic case -> write note (Claude / template) -> reject-sample
(concept-presence + teacher recovery) -> dedup -> split -> jsonl. Prints accept-rate
and per-reason reject counts. Never emits real PHI.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faker import Faker

from evals.case import Case, Gold
from evals.fhir_validate import MODE as FHIR_MODE, validate_resource
from . import teacher
from .generate import synth_case
from .notes import write_note, NoteError
from .filter import accept


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _gold_valid(resources: list[dict[str, Any]]) -> bool:
    return all(validate_resource(r, "R4")[0] for r in resources)


def build(n: int, out: Path, *, seed: int, offline: bool, no_teacher: bool,
          split: tuple[float, float, float]) -> dict[str, Any]:
    online = teacher.available() and not offline
    note_mode = "claude" if online else "template"
    use_teacher = online and not no_teacher

    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    ts = datetime.now(timezone.utc).isoformat()
    sha = _git_sha()
    records: list[dict[str, Any]] = []
    seen_notes: set[str] = set()
    reasons: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(n * 6, 30)

    while len(records) < n and attempts < max_attempts:
        attempts += 1
        sc = synth_case(rng, fake)
        if not _gold_valid(sc.resources):
            reasons["generator produced invalid FHIR"] += 1
            continue
        try:
            note, phi_spans = write_note(sc, note_mode)
        except NoteError as e:
            reasons[str(e).split(":")[0]] += 1
            continue
        if note in seen_notes:
            reasons["duplicate note"] += 1
            continue

        case = Case(
            id=f"case-{len(records):05d}",
            note=note,
            fhir_version="R4",
            target_resources=sc.target_resources,
            deidentify=True,
            gold=Gold(phi_spans=phi_spans, resources=sc.resources),
        )
        res = accept(case, sc.concept_variants, use_teacher=use_teacher)
        if not res.accepted:
            reasons[res.reason if res.reason.startswith(("note", "teacher", "duplicate")) else "rejected"] += 1
            continue

        seen_notes.add(note)
        records.append({
            "case": case.model_dump(),
            "prediction": res.teacher_pred.model_dump() if res.teacher_pred else None,
            "eval": res.eval,
            "provenance": {
                "teacher_model": teacher.MODEL if use_teacher else None,
                "note_mode": note_mode,
                "fhir_validator": FHIR_MODE,
                "git_sha": sha, "ts": ts, "seed": seed,
            },
        })

    # Split (deterministic by insertion order).
    n_train = int(len(records) * split[0])
    n_val = int(len(records) * split[1])
    parts = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "held_out": records[n_train + n_val:],
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    return {
        "accepted": len(records), "attempts": attempts,
        "accept_rate": round(len(records) / attempts, 3) if attempts else 0.0,
        "note_mode": note_mode, "use_teacher": use_teacher, "fhir_validator": FHIR_MODE,
        "splits": {k: len(v) for k, v in parts.items()},
        "reject_reasons": dict(reasons), "out": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="data.build")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="data/curated")
    ap.add_argument("--offline", action="store_true", help="force template notes + cheap filter")
    ap.add_argument("--no-teacher", action="store_true", help="skip the teacher recovery filter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="0.8,0.1,0.1")
    args = ap.parse_args(argv)

    split = tuple(float(x) for x in args.split.split(","))  # type: ignore
    summary = build(
        args.n, Path(args.out), seed=args.seed,
        offline=args.offline, no_teacher=args.no_teacher, split=split,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
