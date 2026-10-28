"""Target-site calibration diagnostic for the cross-hospital alarm policy.

The primary manuscript experiment applies a source-calibrated threshold to the
target hospital without target labels.  This script adds a deliberately
optimistic diagnostic: split the target hospital into patient-disjoint
calibration and test partitions, recalibrate the same policy family on target
calibration labels, and evaluate on target test patients.  It is an oracle
upper bound for what target-site calibration could recover, not a deployable
label-free procedure and not a clinical validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.data.splits import four_way
from eaai.experiments.e2_budget import alarm_loss_matrix, evaluate_from_losses
from eaai.scores.api import ScoreCache


def _load(cache_root: Path, cohort: str, model: str, seed: int):
    data = ScoreCache(cache_root).read(cohort, model, seed)
    return data["scores"], data["onset"], data["lengths"]


def _restrict(scores, onset, lengths, min_stay_hours):
    eligible = np.flatnonzero(lengths >= min_stay_hours)
    if not eligible.size:
        raise ValueError("no records meet the declared minimum stay")
    return scores[eligible], onset[eligible], lengths[eligible], eligible


def _remap(indices, eligible, n_total):
    lookup = np.full(n_total, -1, dtype=int)
    lookup[eligible] = np.arange(eligible.size)
    mapped = lookup[np.asarray(indices, dtype=int)]
    return mapped[mapped >= 0]


def _stack(left, right):
    width = max(left.shape[1], right.shape[1])

    def pad(x):
        return np.pad(x, ((0, 0), (0, width - x.shape[1])), constant_values=np.nan)

    return np.concatenate([pad(left), pad(right)], axis=0)


def _events(onset, lengths):
    onset = np.asarray(onset)
    lengths = np.asarray(lengths)
    return int(np.count_nonzero((onset > 0) & (onset < lengths)))


def _clean(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _rows(rows, label):
    return [{"setting": label,
             **{key: _clean(value) for key, value in row.items()}}
            for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--source-cohort", default="A_lead6")
    parser.add_argument("--target-cohort", default="B_lead6")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=Path, required=True,
                        help="source-cohort train/validation/calibration/test split")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--target-calibration-fraction", type=float, default=0.5)
    parser.add_argument("--target-split-seed", type=int, default=-1,
                        help="defaults to 1000 + seed")
    parser.add_argument("--out", type=Path, default=Path("results/e2_target"))
    args = parser.parse_args()
    if (args.thresholds < 2 or args.min_stay <= 0 or
            not 0 < args.target_calibration_fraction < 1):
        parser.error("thresholds must be at least two, min-stay positive, and "
                     "target calibration fraction strictly between zero and one")

    grid = np.linspace(0.01, 0.99, args.thresholds)
    source_scores, source_onset, source_lengths = _load(
        args.cache_root, args.source_cohort, args.model, args.seed)
    source_total = int(source_lengths.size)
    source_scores, source_onset, source_lengths, source_eligible = _restrict(
        source_scores, source_onset, source_lengths, args.min_stay)

    target_scores, target_onset, target_lengths = _load(
        args.cache_root, args.target_cohort, args.model, args.seed)
    target_total = int(target_lengths.size)
    target_scores, target_onset, target_lengths, target_eligible = _restrict(
        target_scores, target_onset, target_lengths, args.min_stay)

    source_split = np.load(args.splits)
    source_calibration = _remap(source_split["calibration"], source_eligible, source_total)
    source_test = _remap(source_split["test"], source_eligible, source_total)
    if not source_calibration.size or not source_test.size:
        raise ValueError("source eligibility removed a calibration or test partition")

    target_split_seed = (1000 + args.seed if args.target_split_seed < 0
                         else args.target_split_seed)
    target_split = four_way(
        target_scores.shape[0],
        ratios=(0.0, 0.0, args.target_calibration_fraction,
                1.0 - args.target_calibration_fraction),
        seed=target_split_seed)
    target_calibration_local = target_split.calibration
    target_test_local = target_split.test
    offset = source_scores.shape[0]
    target_calibration = target_calibration_local + offset
    target_test = target_test_local + offset

    scores = _stack(source_scores, target_scores)
    onset = np.concatenate([source_onset, target_onset])
    lengths = np.concatenate([source_lengths, target_lengths])
    per_day = alarm_loss_matrix(
        [row for row in scores], lengths, grid,
        refractory=args.refractory, escalation=args.escalation)

    source_internal = []
    target_no_recal = []
    target_recalibrated = []
    for budget in args.budgets:
        source_internal.append(evaluate_from_losses(
            [row for row in scores], onset, lengths,
            source_calibration, source_test, per_day, grid, budget=budget,
            alpha=args.alpha, refractory=args.refractory,
            escalation=args.escalation, lead_window_hours=args.lead_window,
            procedure="blocked", min_stay_hours=args.min_stay))
        target_no_recal.append(evaluate_from_losses(
            [row for row in scores], onset, lengths,
            source_calibration, target_test, per_day, grid, budget=budget,
            alpha=args.alpha, refractory=args.refractory,
            escalation=args.escalation, lead_window_hours=args.lead_window,
            procedure="blocked", min_stay_hours=args.min_stay))
        target_recalibrated.append(evaluate_from_losses(
            [row for row in scores], onset, lengths,
            target_calibration, target_test, per_day, grid, budget=budget,
            alpha=args.alpha, refractory=args.refractory,
            escalation=args.escalation, lead_window_hours=args.lead_window,
            procedure="blocked", min_stay_hours=args.min_stay))

    output = {
        "model": args.model,
        "seed": args.seed,
        "source_cohort": args.source_cohort,
        "target_cohort": args.target_cohort,
        "min_stay_hours": args.min_stay,
        "alpha": args.alpha,
        "target_calibration_fraction": args.target_calibration_fraction,
        "target_split_seed": target_split_seed,
        "eligibility": {
            "source": {"eligible": int(source_eligible.size),
                       "total": source_total},
            "target": {"eligible": int(target_eligible.size),
                       "total": target_total},
        },
        "target_split": {
            "calibration_patients": int(target_calibration_local.size),
            "test_patients": int(target_test_local.size),
            "calibration_events": _events(
                target_onset[target_calibration_local],
                target_lengths[target_calibration_local]),
            "test_events": _events(
                target_onset[target_test_local],
                target_lengths[target_test_local]),
        },
        "source_internal": _rows(source_internal, "source_internal"),
        "target_no_recalibration": _rows(
            target_no_recal, "target_no_recalibration"),
        "target_recalibrated": _rows(
            target_recalibrated, "target_recalibrated"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"e2_target_{args.target_cohort}_{args.model}_s{args.seed}.json"
    (args.out / name).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
