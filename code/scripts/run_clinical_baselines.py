"""Matched-budget external audit for partial NEWS, SIRS and qSOFA.

Clinical scores are not probabilities. They are still useful as alarm-policy
baselines when the threshold is calibrated to the same patient-day budget on
the source site. The script uses a point-estimate global threshold, not the
finite-sample patient-blocked certificate, because the goal is a fair workload
comparison rather than a coverage claim.
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


def _load(cache_root, cohort, model):
    data = ScoreCache(Path(cache_root)).read(cohort, model, 0)
    return data["scores"], data["onset"], data["lengths"]


def _restrict(scores, onset, lengths, min_stay):
    eligible = np.flatnonzero(lengths >= min_stay)
    if not eligible.size:
        raise ValueError("no records meet the minimum stay")
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


def _rows(rows, setting):
    return [{"setting": setting,
             **{key: _clean(value) for key, value in row.items()}}
            for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-cohort", default="A_lead6")
    parser.add_argument("--target-cohort", default="B_lead6")
    parser.add_argument("--instruments", nargs="+", default=["news", "sirs", "qsofa"])
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split = np.load(args.splits)
    output = {"min_stay_hours": args.min_stay, "budgets": args.budgets,
              "instruments": {}}

    for instrument in args.instruments:
        source_scores, source_onset, source_lengths = _load(
            args.cache_root, args.source_cohort, instrument)
        source_total = int(source_lengths.size)
        source_scores, source_onset, source_lengths, source_eligible = _restrict(
            source_scores, source_onset, source_lengths, args.min_stay)
        target_scores, target_onset, target_lengths = _load(
            args.cache_root, args.target_cohort, instrument)
        target_total = int(target_lengths.size)
        target_scores, target_onset, target_lengths, target_eligible = _restrict(
            target_scores, target_onset, target_lengths, args.min_stay)
        calibration = _remap(split["calibration"], source_eligible, source_total)
        test = _remap(split["test"], source_eligible, source_total)
        if not calibration.size or not test.size:
            raise ValueError("eligibility removed a source partition")

        offset = source_scores.shape[0]
        scores = _stack(source_scores, target_scores)
        onset = np.concatenate([source_onset, target_onset])
        lengths = np.concatenate([source_lengths, target_lengths])
        external = np.arange(offset, scores.shape[0])
        upper = float(np.nanmax(source_scores[calibration]))
        if not np.isfinite(upper):
            raise ValueError("clinical score has no finite calibration values")
        grid = np.arange(0.0, np.floor(upper) + 1.0, 1.0)
        if grid.size < 2:
            grid = np.array([0.0, max(1.0, upper)])
        per_day = alarm_loss_matrix([row for row in scores], lengths, grid,
                                    refractory=args.refractory,
                                    escalation=args.escalation)

        def rows_for(cal, tst, label):
            rows = []
            for budget in args.budgets:
                rows.append(evaluate_from_losses(
                    [row for row in scores], onset, lengths, cal, tst,
                    per_day, grid, budget=budget, alpha=0.1,
                    refractory=args.refractory, escalation=args.escalation,
                    lead_window_hours=args.lead_window,
                    procedure="global", min_stay_hours=args.min_stay))
            return _rows(rows, label)

        output["instruments"][instrument] = {
            "threshold_grid": [float(v) for v in grid],
            "internal": rows_for(calibration, test, "internal"),
            "external": rows_for(calibration, external, args.target_cohort),
            "eligibility": {
                "source": {"eligible": int(source_eligible.size), "total": source_total},
                "target": {"eligible": int(target_eligible.size), "total": target_total},
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
