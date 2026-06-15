"""Three-way comparison: base model vs fine-tuned (single shot) vs FT + agent self-refine.

    python -m evals.compare [--data data/curated/held_out.jsonl] [--adapter train/checkpoints/adapter] [--n 20]

Runs all three variants on the held-out set sequentially (GPU memory constraint).
Produces:
  - Rich comparison table (terminal)
  - evals/reports/compare_<ts>.json
  - evals/reports/compare_<ts>.png  (bar chart of key metrics)
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


def _load_cases(path: Path, n: int) -> list[Case]:
    cases = []
    with open(path) as f:
        for line in f:
            if len(cases) >= n:
                break
            cases.append(Case.model_validate(json.loads(line)["case"]))
    return cases


def _mean(vals: list) -> float | None:
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def _aggregate(results: list[dict]) -> dict[str, Any]:
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


def _run_hf_pass(cases: list[Case], adapter_path: str | None, label: str) -> list[dict]:
    from serve.client import HFBackend
    results = []
    print(f"\n=== {label} ===")
    with HFBackend(adapter_path) as backend:
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case.id}", end=" ", flush=True)
            gen = backend.extract(case.note, case.target_resources)
            score, _ = run_eval(case, gen.prediction, use_judge=False)
            print(f"score={score.score:.2f} passed={score.passed}")
            results.append({"case_id": case.id, "score": score.to_dict(), "latency_ms": gen.latency_ms})
    return results


def _run_agent_pass(cases: list[Case], adapter_path: str | None) -> list[dict]:
    from agent.nodes import act as act_node
    from agent.graph import build_graph
    from agent.state import initial_state

    results = []
    print(f"\n=== FT + Agent self-refine ===")
    act_node.load_backend(adapter_path)
    graph = build_graph()
    try:
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case.id}", end=" ", flush=True)
            import time
            t0 = time.monotonic()
            final = graph.invoke(initial_state(case, max_refines=3))
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            score = final["eval_score"]
            refines = final["refine_count"]
            print(f"score={score.score:.2f} passed={score.passed} refines={refines}")
            results.append({
                "case_id": case.id,
                "score": score.to_dict(),
                "latency_ms": latency_ms,
                "refine_count": refines,
            })
    finally:
        act_node.unload_backend()
    return results


def _print_table(variants: dict[str, dict]) -> None:
    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(title="Base vs Fine-tuned vs FT+Agent (held-out set)", show_lines=True)
        table.add_column("Metric", style="bold")
        for name in variants:
            table.add_column(name, justify="right")

        metrics = ["n", "passed", "validity_rate", "resource_f1", "field_accuracy",
                   "deid_recall", "avg_score", "avg_latency_ms"]
        for m in metrics:
            row = [m] + [str(agg[m]) for agg in variants.values()]
            table.add_row(*row)
        Console().print(table)
    except ImportError:
        for m in ["passed", "resource_f1", "validity_rate", "deid_recall", "avg_score"]:
            print(f"{m:20s}", "  ".join(f"{agg[m]!s:10}" for agg in variants.values()))


def _save_chart(variants: dict[str, dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = ["validity_rate", "resource_f1", "field_accuracy", "deid_recall", "avg_score"]
        labels = ["Validity", "Resource F1", "Field Acc.", "De-id Recall", "Avg Score"]
        names = list(variants.keys())
        colors = ["#6b7280", "#3b82f6", "#10b981"]

        x = np.arange(len(metrics))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, (name, agg) in enumerate(variants.items()):
            vals = [agg.get(m) or 0.0 for m in metrics]
            ax.bar(x + idx * width, vals, width, label=name, color=colors[idx], alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Specced: Base vs Fine-tuned vs FT+Agent (held-out set)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"Chart saved → {out_path}")
    except ImportError:
        print("matplotlib not installed — skipping chart (pip install matplotlib)")


def compare(data_path: Path, adapter_path: str | None, n: int) -> dict[str, Any]:
    cases = _load_cases(data_path, n)
    if not cases:
        raise RuntimeError(f"No cases in {data_path}")
    print(f"Loaded {len(cases)} held-out cases.")

    base_results = _run_hf_pass(cases, adapter_path=None, label="Base model")
    ft_results = _run_hf_pass(cases, adapter_path=adapter_path, label="Fine-tuned (single shot)")
    agent_results = _run_agent_pass(cases, adapter_path=adapter_path)

    variants = {
        "Base": _aggregate(base_results),
        "Fine-tuned": _aggregate(ft_results),
        "FT+Agent": _aggregate(agent_results),
    }

    print()
    _print_table(variants)

    return {
        "git_sha": _git_sha(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "data": str(data_path),
        "adapter": adapter_path,
        "n": len(cases),
        "variants": variants,
        "per_case": {
            "base": base_results,
            "ft": ft_results,
            "agent": agent_results,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.compare")
    ap.add_argument("--data", default="data/curated/held_out.jsonl")
    ap.add_argument("--adapter", default="train/checkpoints/adapter")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args(argv)

    import os
    adapter = args.adapter if os.path.exists(args.adapter) else None

    report = compare(Path(args.data), adapter, args.n)

    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = report["ts"].replace(":", "-").replace("+", "")[:19]
    report_path = out_dir / f"compare_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report → {report_path}")

    _save_chart(report["variants"], out_dir / f"compare_{ts}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
