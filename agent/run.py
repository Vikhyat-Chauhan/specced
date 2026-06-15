"""CLI entry point for single-note extraction via the agent graph.

    python -m agent.run <case.json|held_out.jsonl> [--adapter PATH] [--max-refines 3] [--case-id ID]

Loads the model once, runs the plan→retrieve→act→evaluate loop (with self-refine),
prints a before/after score table, and saves the final report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.case import Case, load_case
from agent.state import initial_state
from agent.graph import build_graph
from agent.nodes import act as act_node


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_cases_from(path: Path, case_id: str | None) -> list[Case]:
    if path.suffix == ".json":
        return [load_case(path)]
    # .jsonl — pick by case_id or take first
    cases = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            case = Case.model_validate(rec["case"])
            if case_id is None or case.id == case_id:
                cases.append(case)
                if case_id is not None:
                    break
    return cases[:1] if not case_id else cases


def _print_result(state: dict[str, Any]) -> None:
    score = state["eval_score"]
    if score is None:
        print("No eval score available.")
        return
    try:
        from rich.table import Table
        from rich.console import Console
        table = Table(title=f"Agent Result — {state['case'].id}", show_lines=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for k, v in [
            ("passed", score.passed),
            ("score", score.score),
            ("validity_rate", score.validity_rate),
            ("resource_f1", score.resource_f1),
            ("field_accuracy", score.field_accuracy),
            ("deid_recall", score.deid_recall),
            ("refine_count", state["refine_count"]),
        ]:
            table.add_row(k, str(v))
        Console().print(table)
    except ImportError:
        print(json.dumps({
            "passed": score.passed, "score": score.score,
            "validity_rate": score.validity_rate, "resource_f1": score.resource_f1,
            "deid_recall": score.deid_recall, "refines": state["refine_count"],
        }, indent=2))


def run_case(case: Case, *, adapter_path: str | None, max_refines: int) -> dict[str, Any]:
    graph = build_graph()
    state = initial_state(case, max_refines=max_refines)
    return graph.invoke(state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent.run")
    ap.add_argument("input", help="case .json or held_out .jsonl")
    ap.add_argument("--adapter", default="train/checkpoints/adapter",
                    help="LoRA adapter path (default: train/checkpoints/adapter)")
    ap.add_argument("--max-refines", type=int, default=3)
    ap.add_argument("--case-id", default=None, help="pick a specific case from .jsonl")
    args = ap.parse_args(argv)

    cases = _load_cases_from(Path(args.input), args.case_id)
    if not cases:
        raise SystemExit(f"No cases found in {args.input}")

    import os
    adapter = args.adapter if os.path.exists(args.adapter) else None
    print(f"Loading model ({'adapter: ' + args.adapter if adapter else 'base model'}) ...")
    act_node.load_backend(adapter)

    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for case in cases:
            print(f"\n{'='*60}")
            final = run_case(case, adapter_path=adapter, max_refines=args.max_refines)
            _print_result(final)

            ts = datetime.now(timezone.utc).isoformat().replace(":", "-").replace("+", "")[:19]
            report = {
                "git_sha": _git_sha(), "ts": ts,
                "case_id": case.id, "adapter": adapter,
                "max_refines": args.max_refines,
                "refine_count": final["refine_count"],
                "eval_score": final["eval_score"].to_dict() if final["eval_score"] else None,
                "prediction": final["prediction"].model_dump() if final["prediction"] else None,
                "eval_report": final["eval_report"],
            }
            rpath = out_dir / f"{case.id}_agent_{ts}.json"
            rpath.write_text(json.dumps(report, indent=2))
            print(f"Report → {rpath}")
    finally:
        act_node.unload_backend()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
