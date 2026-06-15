"""End-to-end demonstration: clinical note → de-identified, coded, validated FHIR.

    python scripts/demo.py [--adapter train/checkpoints/adapter] [--max-refines 3]

Runs the full agent pipeline on a sample note and prints each stage's output
in a way that's readable for a live demo or screen recording.
"""

from __future__ import annotations

import argparse
import json
import os
import time

DEMO_NOTE = """Elizabeth Brown, a 67-year-old patient (MRN 47898562), was seen on 07/16/2024.
Contact: 575-326-0332. She has a history of type 2 diabetes mellitus, hypertension,
and hyperlipidemia. Current medications include metformin 850 mg daily, lisinopril
10 mg daily, and atorvastatin 40 mg daily. Recent A1c was 7.2%, BP 138/84 mmHg,
and LDL 98 mg/dL. She denies any drug allergies."""

TARGET_RESOURCES = ["Condition", "MedicationStatement", "Observation"]


def _separator(title: str = "") -> None:
    try:
        from rich.console import Console
        from rich.rule import Rule
        Console().print(Rule(f"[bold]{title}[/bold]" if title else ""))
    except ImportError:
        print(f"\n{'='*60}" + (f" {title} " if title else ""))


def _print_json(data: dict) -> None:
    try:
        from rich.syntax import Syntax
        from rich.console import Console
        Console().print(Syntax(json.dumps(data, indent=2), "json", theme="monokai", line_numbers=False))
    except ImportError:
        print(json.dumps(data, indent=2))


def run_demo(adapter_path: str | None, max_refines: int) -> None:
    from evals.case import Case
    from agent.state import initial_state
    from agent.graph import build_graph
    from agent.nodes import act as act_node

    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        class _FallbackConsole:
            def print(self, *args, **kwargs):
                text = " ".join(str(a) for a in args)
                import re
                print(re.sub(r'\[.*?\]', '', text))
        console = _FallbackConsole()

    _separator("SPECCED — clinical note → de-identified FHIR")
    console.print()
    console.print("[bold]Input note:[/bold]")
    console.print(DEMO_NOTE)
    console.print()
    console.print(f"[dim]Target resources: {', '.join(TARGET_RESOURCES)}[/dim]")
    console.print(f"[dim]Adapter: {adapter_path or 'base model (no adapter)'}[/dim]")
    console.print(f"[dim]Max self-refine iterations: {max_refines}[/dim]")
    console.print()

    _separator("Step 1 — RAG: retrieving standard codes")
    from rag.retrieve import retrieve_hints
    hints = retrieve_hints(DEMO_NOTE, TARGET_RESOURCES)
    if hints:
        console.print(hints)
    else:
        console.print("[dim]No RAG index found — run `make rag-index` to build it.[/dim]")
    console.print()

    _separator("Step 2 — Loading model")
    t_load = time.monotonic()
    act_node.load_backend(adapter_path)
    console.print(f"[dim]Model ready in {time.monotonic() - t_load:.1f}s[/dim]")
    console.print()

    case = Case(
        id="demo-note",
        note=DEMO_NOTE,
        target_resources=TARGET_RESOURCES,
        deidentify=True,
    )

    try:
        _separator("Step 3 — Agent loop (plan → retrieve → act → evaluate)")
        t_start = time.monotonic()

        # Monkey-patch to show iteration progress
        original_run = act_node.run
        _iter = [0]

        def _traced_run(state):
            _iter[0] += 1
            console.print(f"[bold cyan]  Iteration {_iter[0]}[/bold cyan] — generating extraction ...")
            result = original_run(state)
            return result

        act_node.run = _traced_run

        graph = build_graph()
        final = graph.invoke(initial_state(case, max_refines=max_refines))

        act_node.run = original_run  # restore
        elapsed = time.monotonic() - t_start

        pred = final["prediction"]
        score = final["eval_score"]

        _separator("PHI Spans (de-identified)")
        for span in (pred.phi_spans if pred else []):
            console.print(f"  [yellow]{span.type:12s}[/yellow] → [bold]{span.text!r}[/bold]")
        console.print()

        _separator("Extracted FHIR Resources")
        for r in (pred.resources if pred else []):
            rt = r.get("resourceType", "?")
            summary = _resource_summary(r)
            console.print(f"  [cyan]{rt:25s}[/cyan] {summary}")
        console.print()

        _separator("Eval Score")
        if score:
            rows = [
                ("passed", score.passed, score.passed),
                ("score", score.score, score.score >= 0.7),
                ("resource_f1", score.resource_f1, (score.resource_f1 or 0) >= 0.7),
                ("deid_recall", score.deid_recall, (score.deid_recall or 0) >= 0.95),
                ("validity_rate", score.validity_rate, score.validity_rate == 1.0),
                ("refine_iterations", final["refine_count"], True),
            ]
            for label, value, ok in rows:
                color = "green" if ok else "red"
                console.print(f"  {label:20s} [{color}]{value}[/{color}]")
        console.print()
        console.print(f"[dim]Total wall time: {elapsed:.1f}s[/dim]")
        _separator()

    finally:
        act_node.unload_backend()


def _resource_summary(r: dict) -> str:
    for key in ("code", "medicationCodeableConcept", "medicationReference", "substance"):
        if key in r:
            v = r[key]
            if isinstance(v, dict):
                text = v.get("text")
                if text:
                    return text
                for c in v.get("coding", []):
                    if c.get("display"):
                        return f"{c['display']} ({c.get('system','').split('/')[-1]} {c.get('code','')})"
            return str(v)
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="demo", description="Specced end-to-end demo")
    ap.add_argument("--adapter", default="train/checkpoints/adapter")
    ap.add_argument("--max-refines", type=int, default=3)
    args = ap.parse_args(argv)

    adapter = args.adapter if os.path.exists(args.adapter) else None
    run_demo(adapter, args.max_refines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
