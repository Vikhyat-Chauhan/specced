"""specced CLI — extract FHIR from a clinical note or eval a prediction.

Commands:
  specced extract <note.txt>          → PHI spans + FHIR resources (+ eval if --gold given)
  specced eval <case.json>            → score a prediction against gold
  specced compare                     → three-way benchmark on held-out set

Install: pip install -e . then `specced --help`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="specced", help="Spec-driven clinical → FHIR extraction toolkit.")
console = Console()


@app.command()
def extract(
    note: Path = typer.Argument(..., help="Plain-text clinical note file"),
    target_resources: str = typer.Option(
        "Condition,MedicationStatement,Observation,AllergyIntolerance",
        "--resources", "-r",
        help="Comma-separated FHIR resource types to extract",
    ),
    adapter: Path = typer.Option(
        Path("train/checkpoints/adapter"),
        "--adapter", "-a",
        help="LoRA adapter path (defaults to train/checkpoints/adapter)",
    ),
    gold: Optional[Path] = typer.Option(None, "--gold", "-g", help="Gold case JSON for eval"),
    max_refines: int = typer.Option(3, "--max-refines", help="Self-refine iterations"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write prediction JSON here"),
) -> None:
    """Extract de-identified FHIR resources from a clinical note."""
    import os
    from evals.case import Case
    from agent.state import initial_state
    from agent.graph import build_graph
    from agent.nodes import act as act_node

    note_text = note.read_text().strip()
    targets = [t.strip() for t in target_resources.split(",") if t.strip()]

    case = Case(
        id=note.stem,
        note=note_text,
        target_resources=targets,
        deidentify=bool(gold),
    )

    if gold:
        from evals.case import load_case
        gold_case = load_case(gold)
        case = case.model_copy(update={"gold": gold_case.gold, "deidentify": True})

    adapter_path = str(adapter) if adapter.exists() else None
    console.print(f"[bold]Loading model[/bold] ({'adapter' if adapter_path else 'base'}) ...")
    act_node.load_backend(adapter_path)

    try:
        graph = build_graph()
        final = graph.invoke(initial_state(case, max_refines=max_refines))
    finally:
        act_node.unload_backend()

    pred = final["prediction"]
    score = final["eval_score"]

    # Print resources
    console.rule("[bold green]Extracted FHIR Resources")
    for r in (pred.resources if pred else []):
        console.print(f"  [cyan]{r.get('resourceType')}[/cyan] — {_resource_summary(r)}")

    console.rule("[bold green]PHI Spans")
    for s in (pred.phi_spans if pred else []):
        console.print(f"  [{s.type}] {s.text!r}")

    if score:
        console.rule("[bold]Eval Score")
        t = Table(show_header=False, box=None)
        for k, v in [
            ("passed", score.passed), ("score", score.score),
            ("resource_f1", score.resource_f1), ("deid_recall", score.deid_recall),
            ("refines", final["refine_count"]),
        ]:
            color = "green" if v is True or (isinstance(v, float) and v >= 0.7) else "red"
            t.add_row(k, f"[{color}]{v}[/{color}]")
        console.print(t)

    if out and pred:
        out.write_text(json.dumps(pred.model_dump(), indent=2))
        console.print(f"[dim]Prediction written → {out}[/dim]")


def _resource_summary(r: dict) -> str:
    for key in ("code", "medicationCodeableConcept", "medicationReference", "substance"):
        if key in r:
            v = r[key]
            if isinstance(v, dict):
                return v.get("text") or next(
                    (c.get("display") or c.get("code", "") for c in v.get("coding", [])), str(v)
                )
            return str(v)
    return ""


@app.command()
def eval(
    case_path: Path = typer.Argument(..., help="Case JSON file with gold answer"),
    pred_path: Optional[Path] = typer.Option(None, "--pred", "-p", help="Prediction JSON (omit to use gold)"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the clinical LLM judge"),
) -> None:
    """Score a prediction against gold using the eval harness."""
    from evals.case import load_case, load_prediction
    from evals.run_eval import run_eval

    case = load_case(case_path)
    pred = load_prediction(pred_path) if pred_path else (case.gold if case.gold else None)
    if pred is None:
        console.print("[red]No prediction provided and no gold in case.[/red]")
        raise typer.Exit(1)

    from evals.case import Prediction
    if not isinstance(pred, Prediction):
        from evals.case import Gold
        pred = Prediction(phi_spans=pred.phi_spans, resources=pred.resources)

    score, report = run_eval(case, pred, use_judge=not no_judge)

    t = Table(title=f"Eval — {case.id}", show_lines=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in [
        ("passed", score.passed), ("score", score.score),
        ("validity_rate", score.validity_rate), ("resource_f1", score.resource_f1),
        ("field_accuracy", score.field_accuracy), ("deid_recall", score.deid_recall),
    ]:
        color = "green" if v is True or (isinstance(v, float) and v >= 0.7) else "red"
        t.add_row(k, f"[{color}]{v}[/{color}]")
    console.print(t)

    if score.reasons:
        console.print("[yellow]Reasons:[/yellow]", ", ".join(score.reasons))


@app.command()
def compare(
    data: Path = typer.Option(Path("data/curated/held_out.jsonl"), "--data"),
    adapter: Path = typer.Option(Path("train/checkpoints/adapter"), "--adapter"),
    n: int = typer.Option(20, "--n"),
) -> None:
    """Three-way benchmark: base vs fine-tuned vs FT+agent on held-out set."""
    import os
    from evals.compare import compare as run_compare

    adapter_path = str(adapter) if adapter.exists() else None
    run_compare(data, adapter_path, n)


if __name__ == "__main__":
    app()
