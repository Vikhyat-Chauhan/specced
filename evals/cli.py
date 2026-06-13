"""CLI: score a prediction against a gold case.

    python -m evals.cli <case.json> --pred <prediction.json>

If --pred is omitted, looks for a sibling fixture <case-dir>/<case-id>.pred.json
so the harness is runnable before any model exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .case import load_case, load_prediction
from .run_eval import run_eval
from .report import write_report
from .fhir_validate import MODE as FHIR_MODE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.cli")
    ap.add_argument("case", help="path to a case JSON (with gold)")
    ap.add_argument("--pred", help="path to a prediction JSON")
    ap.add_argument("--no-judge", action="store_true", help="skip the clinical LLM judge")
    args = ap.parse_args(argv)

    case = load_case(args.case)

    pred_path = args.pred
    if not pred_path:
        guess = Path(args.case).with_suffix(".pred.json")
        if guess.exists():
            pred_path = str(guess)
    if not pred_path or not Path(pred_path).exists():
        print(f"no --pred given and no fixture at <case>.pred.json", file=sys.stderr)
        return 2
    pred = load_prediction(pred_path)

    print(f"▶ evaluating {case.id}  (FHIR validator: {FHIR_MODE})")
    score, report = run_eval(case, pred, use_judge=not args.no_judge)
    path = write_report(report)

    line = "─" * 52
    print(line)
    print(f"  FHIR validity   : {_pct(score.validity_rate)}"
          + (f"   ({len(score.invalid_resources)} invalid)" if score.invalid_resources else ""))
    print(f"  resource P/R/F1 : {score.resource_precision} / {score.resource_recall} / {score.resource_f1}")
    print(f"  field accuracy  : {score.field_accuracy if score.field_accuracy is not None else 'n/a'}")
    print(f"  de-id recall    : {score.deid_recall if score.deid_recall is not None else 'n/a'}")
    print(f"  clinical judge  : {report['judge']['clinical_correctness'] if report['judge'] else 'n/a (no key)'}")
    print(f"  SCORE           : {score.score}  ->  {'PASS' if score.passed else 'FAIL'}")
    if score.invalid_resources:
        for inv in score.invalid_resources:
            print(f"    invalid: {inv}")
    if score.reasons:
        print("  notes           : " + "; ".join(score.reasons))
    print(line)
    print(f"report: {path}")
    return 0 if score.passed else 1


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


if __name__ == "__main__":
    raise SystemExit(main())
