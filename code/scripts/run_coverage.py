"""Repeated-split coverage study for the alarm-budget certificate (E0R)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.data.synthetic import make_scores
from eaai.experiments.coverage import (SplitFractions, coverage_study,
                                       summarize_coverage)
from eaai.scores.api import ScoreCache


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _load(args):
    if args.synthetic:
        cohort = make_scores(args.n_patients, seed=args.seed)
        return cohort["scores"], cohort["onset"], cohort["lengths"]
    data = ScoreCache(Path(args.cache_root)).read(args.cohort, args.model, args.seed)
    return data["scores"], data["onset"], data["lengths"]


def _restrict(scores, onset, lengths, min_stay):
    keep = np.flatnonzero(np.asarray(lengths) >= min_stay)
    if not keep.size:
        raise ValueError("no records meet the declared minimum stay")
    return [scores[i] for i in keep], np.asarray(onset)[keep], np.asarray(lengths)[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--cohort", default="A_lead6")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-patients", type=int, default=800)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--procedures", nargs="+",
                        default=["blocked", "timestep", "plugin",
                                 "ltt_hoeffding_holm", "randomized"])
    parser.add_argument("--selection", default="default",
                        choices=["default", "validation"])
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--threshold-style", choices=["linear", "quantile"],
                        default="linear")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--n-probability-points", type=int, default=11)
    parser.add_argument("--out", type=Path,
                        default=Path("results/coverage/coverage.json"))
    args = parser.parse_args()

    scores, onset, lengths = _load(args)
    scores, onset, lengths = _restrict(scores, onset, lengths, args.min_stay)
    if args.threshold_style == "linear":
        thresholds = np.linspace(0.01, 0.99, args.thresholds)
    else:
        observed = np.concatenate([np.asarray(s)[:int(n)] for s, n in zip(scores, lengths)])
        thresholds = np.unique(np.quantile(observed, np.linspace(0.5, 0.999, args.thresholds)))
    if thresholds.size < 2:
        raise ValueError("need at least two distinct thresholds")

    cal_fraction = 0.5 if args.validation_fraction <= 0 else 0.5 - args.validation_fraction / 2
    fractions = SplitFractions(calibration=cal_fraction,
                               validation=args.validation_fraction,
                               test=1.0 - cal_fraction - args.validation_fraction)
    rows = coverage_study(
        scores, onset, lengths, thresholds, budgets=tuple(args.budgets),
        procedures=tuple(args.procedures), alpha=args.alpha,
        n_splits=args.n_splits, fractions=fractions,
        refractory=args.refractory, escalation=args.escalation,
        min_stay_hours=args.min_stay, lead_window_hours=args.lead_window,
        selection=args.selection,
        n_probability_points=args.n_probability_points, seed=args.seed)
    summary = summarize_coverage(rows, seed=args.seed)
    output = {
        "config": {
            "cohort": "synthetic" if args.synthetic else args.cohort,
            "model": args.model,
            "seed": args.seed,
            "n_patients": int(len(scores)),
            "n_events": int(np.sum((onset > 0) & (onset < lengths))),
            "n_splits": args.n_splits,
            "budgets": [float(b) for b in args.budgets],
            "procedures": list(args.procedures),
            "alpha": args.alpha,
            "refractory": args.refractory,
            "escalation": args.escalation,
            "min_stay_hours": args.min_stay,
            "lead_window_hours": args.lead_window,
            "selection": args.selection,
            "validation_fraction": args.validation_fraction,
            "n_thresholds": int(thresholds.size),
            "threshold_style": args.threshold_style,
        },
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean = _clean(output)
    args.out.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"config": clean["config"], "summary": clean["summary"]},
                     indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
