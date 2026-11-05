"""Translate calibrated alarm rates into simulated ward workload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e2_budget import alarm_loss_matrix
from eaai.experiments.ward import simulate_ward
from eaai.policy.alarm import AlarmPolicy, max_alarms_per_day, run_policy
from eaai.policy.conformal import BlockedConformalCalibrator
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
    return value


def _load(root, cohort, model, seed, min_stay):
    data = ScoreCache(Path(root)).read(cohort, model, seed)
    scores = [np.asarray(seq, dtype=float) for seq in data["scores"]]
    onset = np.asarray(data["onset"], dtype=int)
    lengths = np.asarray(data["lengths"], dtype=int)
    keep = np.flatnonzero(lengths >= min_stay)
    return ([scores[i] for i in keep], onset[keep], lengths[keep],
            np.flatnonzero(lengths >= min_stay), lengths.size)


def _remap(indices, eligible, n_total):
    lookup = np.full(n_total, -1, dtype=int)
    lookup[eligible] = np.arange(eligible.size)
    mapped = lookup[np.asarray(indices, dtype=int)]
    return mapped[mapped >= 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--cohort", default="A_lead6")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=Path,
                        default=Path("experiment_results/round2/splits/splits_A_s0.npz"))
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--arrival-rates", type=float, nargs="+",
                        default=[2.0, 5.0, 10.0])
    parser.add_argument("--n-beds", type=int, default=20)
    parser.add_argument("--capacities", type=float, nargs="+",
                        default=[1.0, 2.0, 4.0])
    parser.add_argument("--horizon-days", type=float, default=30.0)
    parser.add_argument("--warmup-days", type=float, default=7.0)
    parser.add_argument("--out", type=Path,
                        default=Path("results/ward/ward.json"))
    args = parser.parse_args()

    scores, onset, lengths, eligible, n_total = _load(
        args.cache_root, args.cohort, args.model, args.seed, args.min_stay)
    split = np.load(args.splits)
    calibration = _remap(split["calibration"], eligible, n_total)
    test = _remap(split["test"], eligible, n_total)
    thresholds = np.linspace(0.01, 0.99, args.thresholds)
    per_day = alarm_loss_matrix(scores, lengths, thresholds,
                                refractory=args.refractory,
                                escalation=args.escalation)
    l_max = max_alarms_per_day(args.refractory, step_hours=1.0,
                               min_stay_hours=args.min_stay,
                               escalation=args.escalation)

    rows = []
    for budget in args.budgets:
        calibrator = BlockedConformalCalibrator(budget=budget, alpha=args.alpha,
                                                 slack="bernstein")
        calibrator.fit(per_day[calibration], thresholds, l_max)
        threshold = float(calibrator.threshold_)
        traces = [run_policy(scores[i][:lengths[i]],
                             AlarmPolicy(threshold, args.refractory, args.escalation))
                  for i in test]
        for arrival_rate in args.arrival_rates:
            for capacity in args.capacities:
                result = simulate_ward(
                    traces, lengths[test], arrival_rate=arrival_rate,
                    horizon_days=args.horizon_days, n_beds=args.n_beds,
                    capacity_per_hour=capacity, warmup_days=args.warmup_days,
                    seed=args.seed)
                rows.append({
                    "budget": float(budget), "threshold": threshold,
                    "status": calibrator.diagnostics_["status"],
                    "capacity_per_hour": float(capacity),
                    **result,
                })

    output = {
        "config": {
            "cohort": args.cohort, "model": args.model, "seed": args.seed,
            "alpha": args.alpha, "n_test": int(test.size),
            "n_test_events": int(np.sum((onset[test] > 0) & (onset[test] < lengths[test]))),
            "budgets": [float(b) for b in args.budgets],
            "arrival_rates": [float(x) for x in args.arrival_rates],
            "n_beds": args.n_beds,
            "capacities": [float(x) for x in args.capacities],
            "horizon_days": args.horizon_days,
        },
        "rows": rows,
    }
    clean = _clean(output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
