"""Run E2 with one cached patient-loss matrix per seed.

This is the same experiment as ``run_e2.py`` but computes the stateful policy
loss matrix once for the source and external cohorts.  The original driver
recomputed that matrix separately for every procedure, which is unnecessary
and makes multi-seed runs expensive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e2_budget import alarm_loss_matrix, evaluate_from_losses
from eaai.scores.api import ScoreCache


def _load(cache_root, cohort, model, seed):
    data = ScoreCache(Path(cache_root)).read(cohort, model, seed)
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
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cohort", default="A_lead6")
    parser.add_argument("--splits", type=Path, default=Path("results/e1/splits_A_s0.npz"))
    parser.add_argument("--external-cohort", default=None)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--include-timestep", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/e2"))
    args = parser.parse_args()
    if args.thresholds < 2 or args.min_stay <= 0:
        parser.error("thresholds must be at least two and min-stay positive")

    grid = np.linspace(0.01, 0.99, args.thresholds)
    scores, onset, lengths = _load(args.cache_root, args.cohort, args.model, args.seed)
    n_total = int(lengths.size)
    scores, onset, lengths, eligible = _restrict(
        scores, onset, lengths, args.min_stay)
    split = np.load(args.splits)
    calibration = _remap(split["calibration"], eligible, n_total)
    test = _remap(split["test"], eligible, n_total)
    if not calibration.size or not test.size:
        raise ValueError("eligibility restriction removed a source partition")

    ext_scores = ext_onset = ext_lengths = None
    ext_eligible = None
    ext_total = None
    if args.external_cohort:
        ext_scores, ext_onset, ext_lengths = _load(
            args.cache_root, args.external_cohort, args.model, args.seed)
        ext_total = int(ext_lengths.size)
        ext_scores, ext_onset, ext_lengths, ext_eligible = _restrict(
            ext_scores, ext_onset, ext_lengths, args.min_stay)
        combined_scores = _stack(scores, ext_scores)
        combined_onset = np.concatenate([onset, ext_onset])
        combined_lengths = np.concatenate([lengths, ext_lengths])
        ext_test = np.arange(scores.shape[0], combined_scores.shape[0])
    else:
        combined_scores = scores
        combined_onset = onset
        combined_lengths = lengths
        ext_test = None

    per_day = alarm_loss_matrix(
        [row for row in combined_scores], combined_lengths, grid,
        refractory=args.refractory, escalation=args.escalation)

    def rows_for(procedure, cal, tst, label):
        rows = []
        for budget in args.budgets:
            rows.append({"setting": label, **evaluate_from_losses(
                [row for row in combined_scores], combined_onset, combined_lengths,
                cal, tst, per_day, grid, budget=budget, alpha=args.alpha,
                refractory=args.refractory, escalation=args.escalation,
                lead_window_hours=args.lead_window, procedure=procedure,
                min_stay_hours=args.min_stay)})
        return rows

    out = {
        "internal": _rows(rows_for("blocked", calibration, test, "internal"),
                          "internal"),
        "eligibility": {"min_stay_hours": args.min_stay,
                         "eligible": int(eligible.size), "total": n_total},
    }
    if args.include_timestep:
        out["internal_timestep"] = _rows(
            rows_for("timestep", calibration, test, "internal_timestep"),
            "internal_timestep")
    if ext_test is not None:
        out["external"] = _rows(
            rows_for("blocked", calibration, ext_test, args.external_cohort),
            args.external_cohort)
        if args.include_timestep:
            out["external_timestep"] = _rows(
                rows_for("timestep", calibration, ext_test,
                         f"{args.external_cohort}_timestep"),
                f"{args.external_cohort}_timestep")
        out["external_eligibility"] = {
            "min_stay_hours": args.min_stay,
            "eligible": int(ext_eligible.size), "total": ext_total}

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"e2_{args.cohort}_{args.model}_s{args.seed}.json"
    (args.out / name).write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
