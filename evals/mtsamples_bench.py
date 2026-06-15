"""FHIR validity benchmark on real MTSamples transcriptions.

    python -m evals.mtsamples_bench [--n 50] [--adapter train/checkpoints/adapter]

No gold annotations — measures FHIR validity rate, resource counts, and
de-id span detection on real out-of-distribution clinical notes.
Compares base model vs fine-tuned on the same notes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.fhir_validate import validate_resource


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _load_mtsamples(n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset
    import random
    ds = load_dataset("NickyNicky/medical_mtsamples", split="train")
    rows = [r for r in ds if r.get("transcription") and len(r["transcription"]) > 100]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _infer_targets(specialty: str) -> list[str]:
    specialty = specialty.lower()
    targets = ["Condition", "MedicationStatement", "Observation", "AllergyIntolerance"]
    if "surgery" in specialty or "orthopedic" in specialty:
        targets.append("Procedure")
    return targets


def _score_row(backend, row: dict) -> dict[str, Any]:
    import time
    note = row["transcription"][:2000]  # cap at 2000 chars for speed
    targets = _infer_targets(row.get("medical_specialty", ""))

    t0 = time.monotonic()
    result = backend.extract(note, targets)
    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    resources = result.prediction.resources
    phi_spans = result.prediction.phi_spans

    valid, invalid = [], []
    for r in resources:
        ok, err = validate_resource(r, "R4")
        if ok:
            valid.append(r.get("resourceType", "?"))
        else:
            invalid.append(f"{r.get('resourceType','?')}: {err}")

    return {
        "specialty": row.get("medical_specialty", "").strip(),
        "note_chars": len(note),
        "n_resources": len(resources),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "validity_rate": len(valid) / len(resources) if resources else None,
        "resource_types": Counter(valid),
        "invalid_errors": invalid[:3],
        "n_phi_spans": len(phi_spans),
        "phi_types": Counter(s.type for s in phi_spans),
        "latency_ms": latency_ms,
        "tokens_used": result.tokens_used,
        "raw_truncated": result.raw[:200],
    }


def _mean(vals):
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def _print_results(label: str, results: list[dict]) -> None:
    try:
        from rich.table import Table
        from rich.console import Console

        # Summary table
        validity_rates = [r["validity_rate"] for r in results]
        n_resources = [r["n_resources"] for r in results]
        n_phi = [r["n_phi_spans"] for r in results]
        perfect = sum(1 for r in results if r["validity_rate"] == 1.0 and r["n_resources"] > 0)
        empty = sum(1 for r in results if r["n_resources"] == 0)

        t = Table(title=f"{label} — MTSamples ({len(results)} notes)", show_lines=True)
        t.add_column("Metric"); t.add_column("Value", justify="right")
        t.add_row("Notes evaluated", str(len(results)))
        t.add_row("Avg validity rate", str(_mean(validity_rates)))
        t.add_row("Perfect validity (1.0)", str(perfect))
        t.add_row("Empty output (0 resources)", str(empty))
        t.add_row("Avg resources/note", str(_mean(n_resources)))
        t.add_row("Avg PHI spans/note", str(_mean(n_phi)))
        t.add_row("Avg latency (ms)", str(_mean([r["latency_ms"] for r in results])))

        # Resource type breakdown
        all_types: Counter = Counter()
        for r in results:
            all_types.update(r["resource_types"])

        Console().print(t)
        Console().print(f"\n[bold]Resource types seen:[/bold] {dict(all_types.most_common(8))}")

        # Specialty breakdown
        by_spec: dict[str, list] = defaultdict(list)
        for r in results:
            by_spec[r["specialty"][:30]].append(r["validity_rate"])
        Console().print("\n[bold]Validity by specialty (top 5):[/bold]")
        for spec, vals in sorted(by_spec.items(), key=lambda x: -(_mean(x[1]) or 0))[:5]:
            Console().print(f"  {spec:32s}  {_mean(vals):.3f}  (n={len(vals)})")

    except ImportError:
        print(f"\n=== {label} ===")
        print(f"Avg validity: {_mean([r['validity_rate'] for r in results])}")
        print(f"Avg resources: {_mean([r['n_resources'] for r in results])}")
        print(f"Avg PHI spans: {_mean([r['n_phi_spans'] for r in results])}")


def benchmark(n: int, adapter_path: str | None) -> dict[str, Any]:
    from serve.client import HFBackend

    print(f"Loading {n} MTSamples transcriptions ...")
    rows = _load_mtsamples(n)
    print(f"Loaded {len(rows)} notes across specialties.\n")

    all_results: dict[str, list] = {}

    for label, adapter in [("Base model", None), ("Fine-tuned", adapter_path)]:
        if adapter is not None and not Path(adapter).exists():
            print(f"Skipping {label} — adapter not found: {adapter}")
            continue
        results = []
        print(f"\n=== {label} ===")
        with HFBackend(adapter) as backend:
            for i, row in enumerate(rows, 1):
                spec = row.get("medical_specialty", "").strip()[:25]
                print(f"  [{i:2d}/{len(rows)}] {spec:25s}", end=" ", flush=True)
                r = _score_row(backend, row)
                vr = f"{r['validity_rate']:.2f}" if r["validity_rate"] is not None else " — "
                print(f"valid={vr} resources={r['n_resources']:2d} phi={r['n_phi_spans']}")
                results.append(r)
        all_results[label] = results
        print()
        _print_results(label, results)

    return {
        "git_sha": _git_sha(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "adapter": adapter_path,
        "results": {k: v for k, v in all_results.items()},
        "summary": {
            k: {
                "avg_validity": _mean([r["validity_rate"] for r in v]),
                "avg_resources": _mean([r["n_resources"] for r in v]),
                "avg_phi_spans": _mean([r["n_phi_spans"] for r in v]),
                "perfect_validity": sum(1 for r in v if r["validity_rate"] == 1.0 and r["n_resources"] > 0),
                "empty_output": sum(1 for r in v if r["n_resources"] == 0),
            }
            for k, v in all_results.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.mtsamples_bench")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--adapter", default="train/checkpoints/adapter")
    args = ap.parse_args(argv)

    adapter = args.adapter if os.path.exists(args.adapter) else None
    report = benchmark(args.n, adapter)

    out = Path("evals/reports")
    out.mkdir(parents=True, exist_ok=True)
    ts = report["ts"].replace(":", "-").replace("+", "")[:19]
    p = out / f"mtsamples_{ts}.json"
    p.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
