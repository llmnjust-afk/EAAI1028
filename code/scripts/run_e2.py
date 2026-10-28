"""Run internal and cross-hospital budget experiments from cached scores (E2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e2_budget import sweep
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


def _rows(rows, label):
    def clean(value):
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value
    return [{"setting": label, **{k: clean(v) for k, v in row.items()}} for row in rows]


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
    parser.add_argument("--include-timestep", action="store_true",
                        help="also run the invalid timestep effective-sample comparator")
    parser.add_argument("--out", type=Path, default=Path("results/e2"))
    args = parser.parse_args()
    if args.thresholds < 2 or args.min_stay <= 0:
        parser.error("thresholds must be at least two and min-stay positive")

    grid = np.linspace(0.01, 0.99, args.thresholds)
    scores, onset, lengths = _load(args.cache_root, args.cohort, args.model, args.seed)
    scores, onset, lengths, eligible = _restrict(scores, onset, lengths, args.min_stay)
    split = np.load(args.splits)
    # The original cohort size is recovered from the unfiltered cache length.
    n_total = ScoreCache(Path(args.cache_root)).read(
        args.cohort, args.model, args.seed)["lengths"].size
    calibration = _remap(split["calibration"], eligible, n_total)
    test = _remap(split["test"], eligible, n_total)
    if not calibration.size or not test.size:
        raise ValueError("eligibility restriction removed a calibration or test partition")

    def run_sweep(procedure, cal, tst, sc, on, ln):
        return sweep([row for row in sc], on, ln, cal, tst,
                     budgets=tuple(args.budgets), thresholds=grid,
                     alpha=args.alpha, refractory=args.refractory,
                     escalation=args.escalation, min_stay_hours=args.min_stay,
                     lead_window_hours=args.lead_window, procedure=procedure)

    internal = run_sweep("blocked", calibration, test, scores, onset, lengths)
    output = {"internal": _rows(internal, "internal"),
              "eligibility": {"min_stay_hours": args.min_stay,
                               "eligible": int(eligible.size),
                               "total": int(n_total)}}
    if args.include_timestep:
        output["internal_timestep"] = _rows(
            run_sweep("timestep", calibration, test, scores, onset, lengths),
            "internal_timestep")

    if args.external_cohort:
        ext_scores, ext_onset, ext_lengths = _load(
            args.cache_root, args.external_cohort, args.model, args.seed)
        ext_total = int(ext_lengths.size)
        ext_scores, ext_onset, ext_lengths, ext_eligible = _restrict(
            ext_scores, ext_onset, ext_lengths, args.min_stay)
        combined_scores = _stack(scores, ext_scores)
        combined_onset = np.concatenate([onset, ext_onset])
        combined_lengths = np.concatenate([lengths, ext_lengths])
        ext_test = np.arange(len(onset), len(combined_lengths))
        external = run_sweep("blocked", calibration, ext_test, combined_scores,
                             combined_onset, combined_lengths)
        output["external"] = _rows(external, args.external_cohort)
        if args.include_timestep:
            output["external_timestep"] = _rows(
                run_sweep("timestep", calibration, ext_test, combined_scores,
                          combined_onset, combined_lengths),
                f"{args.external_cohort}_timestep")
        output["external_eligibility"] = {
            "min_stay_hours": args.min_stay,
            "eligible": int(ext_eligible.size), "total": ext_total}

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"e2_{args.cohort}_{args.model}_s{args.seed}.json"
    (args.out / name).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
