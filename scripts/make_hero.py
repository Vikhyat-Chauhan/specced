"""Render the README hero (docs/specced-hero.svg) from a real eval-harness run.

Uses rich to draw the eval result card for the example case and exports it to SVG.
The card shows the harness validating, scoring, and de-identifying an extraction —
including the issues it catches — which is the project's whole point.

    python scripts/make_hero.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python scripts/make_hero.py` from anywhere

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evals.case import load_case, load_prediction
from evals.run_eval import run_eval

OUT = ROOT / "docs" / "specced-hero.svg"


def main() -> None:
    case = load_case(ROOT / "specs/examples/cardio-visit.json")
    pred = load_prediction(ROOT / "specs/examples/cardio-visit.pred.json")
    s, _ = run_eval(case, pred, use_judge=False)

    t = Table.grid(padding=(0, 3))
    t.add_column(justify="right", style="bold cyan")
    t.add_column()
    t.add_row("FHIR validity", Text(f"{s.validity_rate:.0%}   ({len(s.invalid_resources)} invalid)", style="yellow"))
    t.add_row("resource P / R / F1", Text(f"{s.resource_precision} / {s.resource_recall} / {s.resource_f1}"))
    t.add_row("field accuracy", Text(str(s.field_accuracy)))
    t.add_row("de-id recall", Text(str(s.deid_recall), style="red"))
    verdict = Text("PASS", style="bold green") if s.passed else Text("FAIL", style="bold red")
    t.add_row("score", Text(f"{s.score}   ", style="bold").append(verdict))

    panel = Panel(
        t,
        title="[bold]specced[/]  ·  clinical note → FHIR",
        subtitle="every extraction validated · scored · de-identified",
        border_style="cyan",
        padding=(1, 2),
    )
    console = Console(record=True, width=66)
    console.print()
    console.print(panel)
    console.save_svg(str(OUT), title="specced")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
