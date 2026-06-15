"""Base-vs-fine-tuned comparison on the held-out set.

    python -m evals.benchmark [--data data/curated/held_out.jsonl] [--adapter train/checkpoints/adapter] [--n 20]

Loads the base model and the fine-tuned adapter sequentially (to stay within
16GB VRAM) and scores each with the eval harness. Prints a Rich comparison
table and writes a JSON report to evals/reports/benchmark_<ts>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.case import Case, Prediction
from evals.run_eval import run_eval
from evals.score import EvalScore


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_cases(jsonl_path: Path, n: int) -> list[Case]:
    cases: list[Case] = []
    with open(jsonl_path) as f:
        for line in f:
            if len(cases) >= n:
                break
            rec = json.loads(line)
            cases.append(Case.model_validate(rec["case"]))
    return cases


def _run_pass(
    cases: list[Case],
    adapter_path: str | None,
) -> list[dict[str, Any]]:
    """Run extraction + eval for all cases with one model variant."""
    from serve.client import HFBackend

    results = []
    with HFBackend(adapter_path) as backend:
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case.id}", end=" ", flush=True)
            gen = backend.extract(case.note, case.target_resources)
            score, _ = run_eval(case, gen.prediction, use_judge=False)
            print(f"score={score.score:.2f} valid={score.validity_rate} passed={score.passed}")
            results.append({
                "case_id": case.id,
                "score": score.to_dict(),
                "latency_ms": gen.latency_ms,
                "tokens_used": gen.tokens_used,
            })
    return results


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r["score"] for r in results]
    return {
        "n": len(results),
        "passed": sum(1 for s in scores if s["passed"]),
        "validity_rate": _mean([s["validity_rate"] for s in scores]),
        "resource_f1": _mean([s["resource_f1"] for s in scores]),
        "field_accuracy": _mean([s["field_accuracy"] for s in scores]),
        "deid_recall": _mean([s["deid_recall"] for s in scores]),
        "avg_score": _mean([s["score"] for s in scores]),
        "avg_latency_ms": _mean([r["latency_ms"] for r in results]),
    }


def _print_table(base_agg: dict[str, Any], ft_agg: dict[str, Any]) -> None:
    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(title="Base vs Fine-tuned (held-out set)", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Base", justify="right")
        table.add_column("Fine-tuned", justify="right")
        table.add_column("Δ", justify="right")

        def _fmt(v: Any) -> str:
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)

        def _delta(a: Any, b: Any) -> str:
            if a is None or b is None or not isinstance(a, (int, float)):
                return ""
            d = b - a
            return f"[green]+{d:.3f}[/green]" if d > 0 else f"[red]{d:.3f}[/red]" if d < 0 else "0"

        rows = [
            ("n", "n"),
            ("passed", "passed"),
            ("validity_rate", "validity_rate"),
            ("resource_f1", "resource_f1"),
            ("field_accuracy", "field_accuracy"),
            ("deid_recall", "deid_recall"),
            ("avg_score", "avg_score"),
            ("avg_latency_ms", "avg_latency_ms"),
        ]
        for label, key in rows:
            table.add_row(label, _fmt(base_agg[key]), _fmt(ft_agg[key]), _delta(base_agg[key], ft_agg[key]))

        Console().print(table)
    except ImportError:
        # Plain fallback
        header = f"{'Metric':<20}{'Base':>12}{'Fine-tuned':>12}{'Δ':>10}"
        print("\n" + header)
        print("-" * len(header))
        for key in base_agg:
            b, f_ = base_agg[key], ft_agg[key]
            delta = ""
            if isinstance(b, (int, float)) and isinstance(f_, (int, float)):
                delta = f"{f_ - b:+.3f}"
            print(f"{key:<20}{str(b):>12}{str(f_):>12}{delta:>10}")


def benchmark(
    data_path: Path,
    adapter_path: str | None,
    n: int,
) -> dict[str, Any]:
    cases = _load_cases(data_path, n)
    if not cases:
        raise RuntimeError(f"No cases found in {data_path}")
    print(f"Loaded {len(cases)} held-out cases.\n")

    print("=== Base model ===")
    base_results = _run_pass(cases, adapter_path=None)

    print(f"\n=== Fine-tuned ({adapter_path}) ===")
    ft_results = _run_pass(cases, adapter_path=adapter_path)

    base_agg = _aggregate(base_results)
    ft_agg = _aggregate(ft_results)

    print()
    _print_table(base_agg, ft_agg)

    return {
        "git_sha": _git_sha(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "data": str(data_path),
        "adapter": adapter_path,
        "n": len(cases),
        "base": {"aggregate": base_agg, "per_case": base_results},
        "ft": {"aggregate": ft_agg, "per_case": ft_results},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.benchmark")
    ap.add_argument("--data", default="data/curated/held_out.jsonl")
    ap.add_argument("--adapter", default="train/checkpoints/adapter",
                    help="path to fine-tuned LoRA adapter; pass 'none' to skip FT pass")
    ap.add_argument("--n", type=int, default=20, help="max held-out cases to evaluate")
    args = ap.parse_args(argv)

    adapter = None if args.adapter.lower() == "none" else args.adapter
    if adapter and not Path(adapter).exists():
        raise SystemExit(f"Adapter not found: {adapter}. Run train.train_qlora first.")

    report = benchmark(Path(args.data), adapter, args.n)

    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = report["ts"].replace(":", "-").replace("+", "")[:19]
    report_path = out_dir / f"benchmark_{ts_slug}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
