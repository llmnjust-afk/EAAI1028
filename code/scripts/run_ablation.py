"""One-at-a-time ablations for the stateful alarm policy and certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.coverage import SplitFractions, stratified_split
from eaai.experiments.e2_budget import evaluate_at_budget
from eaai.scores.api import ScoreCache


def _load(root, cohort, model, seed, min_stay):
    data = ScoreCache(Path(root)).read(cohort, model, seed)
    scores = [np.asarray(seq, dtype=float) for seq in data["scores"]]
    onset = np.asarray(data["onset"], dtype=int)
    lengths = np.asarray(data["lengths"], dtype=int)
    keep = np.flatnonzero(lengths >= min_stay)
    return [scores[i] for i in keep], onset[keep], lengths[keep], keep, lengths.size


def _remap(indices, eligible, n_total):
    lookup = np.full(n_total, -1, dtype=int)
    lookup[eligible] = np.arange(eligible.size)
    mapped = lookup[np.asarray(indices, dtype=int)]
    return mapped[mapped >= 0]


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _row(value, outcome, parameter):
    return {
        "parameter": parameter, "value": value,
        "budget": outcome["budget"], "threshold": outcome["threshold"],
        "alarm_rate": outcome["alarm_rate"],
        "sensitivity": outcome["sensitivity"],
        "ppv": outcome["ppv"], "nna": outcome["nna"],
        "median_lead_time": outcome["median_lead_time"],
        "patient_violation": outcome["patient_violation"],
        "certified_risk": outcome["certified_risk"],
        "status": outcome["status"], "n_feasible": outcome["n_feasible"],
        "l_max": outcome["l_max"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--cohort", default="A_lead6")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=Path,
                        default=Path("experiment_results/round2/splits/splits_A_s0.npz"))
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("results/ablation/ablation.json"))
    args = parser.parse_args()

    scores, onset, lengths, eligible, n_total = _load(
        args.cache_root, args.cohort, args.model, args.seed, args.min_stay)
    split = np.load(args.splits)
    calibration = _remap(split["calibration"], eligible, n_total)
    test = _remap(split["test"], eligible, n_total)
    base_thresholds = np.linspace(0.01, 0.99, args.thresholds)
    rows = []

    def run(parameter, value, **overrides):
        config = dict(alpha=args.alpha, refractory=args.refractory,
                      escalation=args.escalation, min_stay_hours=args.min_stay,
                      lead_window_hours=args.lead_window)
        config.update(overrides)
        outcome = evaluate_at_budget(
            scores, onset, lengths, calibration, test, budget=args.budget,
            thresholds=base_thresholds, procedure="blocked", **config)
        rows.append(_row(value, outcome, parameter))

    run("baseline", "baseline")
    for value in (2, 4, 8, 12):
        run("refractory", value, refractory=value)
    for value in (0, 1, 2):
        run("escalation", value, escalation=value)
    for value in (6.0, 12.0, 24.0):
        run("lead_window", value, lead_window_hours=value)
    for value in (0.05, 0.1, 0.2):
        run("alpha", value, alpha=value)
    for value in (20, 50, 100, 400):
        thresholds = np.linspace(0.01, 0.99, value)
        outcome = evaluate_at_budget(
            scores, onset, lengths, calibration, test, budget=args.budget,
            thresholds=thresholds, procedure="blocked", alpha=args.alpha,
            refractory=args.refractory, escalation=args.escalation,
            min_stay_hours=args.min_stay,
            lead_window_hours=args.lead_window)
        rows.append(_row(value, outcome, "threshold_grid"))

    for min_stay in (6.0, 12.0, 48.0):
        s2, o2, l2, _, _ = _load(args.cache_root, args.cohort, args.model,
                                 args.seed, min_stay)
        parts = stratified_split(o2, l2, SplitFractions(0.5, 0.0, 0.5),
                                 np.random.default_rng(args.seed))
        outcome = evaluate_at_budget(
            s2, o2, l2, parts["calibration"], parts["test"],
            budget=args.budget, thresholds=base_thresholds,
            procedure="blocked", alpha=args.alpha,
            refractory=args.refractory, escalation=args.escalation,
            min_stay_hours=min_stay, lead_window_hours=args.lead_window)
        rows.append(_row(min_stay, outcome, "min_stay"))

    output = {"config": {
        "cohort": args.cohort, "model": args.model, "seed": args.seed,
        "budget": args.budget, "alpha": args.alpha,
        "refractory": args.refractory, "escalation": args.escalation,
        "min_stay": args.min_stay, "lead_window": args.lead_window,
        "thresholds": args.thresholds,
        "n_calibration": int(calibration.size), "n_test": int(test.size),
    }, "rows": rows}
    clean = _clean(output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
