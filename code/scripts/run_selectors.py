"""Compare certified selectors on one subject-disjoint split and transfer the threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e2_budget import (evaluate_at_budget, evaluate_threshold)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--source-cohort", default="A_lead6")
    parser.add_argument("--target-cohort", default="B_lead6")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiment_results/round2/splits/splits_A_s0.npz")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--procedures", nargs="+",
                        default=["blocked", "plugin", "ltt_hoeffding_holm",
                                 "ltt_bernstein_holm", "randomized"])
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--out", type=Path,
                        default=Path("results/selectors/selectors.json"))
    args = parser.parse_args()

    scores, onset, lengths, eligible, n_total = _load(
        args.cache_root, args.source_cohort, args.model, args.seed, args.min_stay)
    split = np.load(args.splits)
    calibration = _remap(split["calibration"], eligible, n_total)
    test = _remap(split["test"], eligible, n_total)
    target_scores, target_onset, target_lengths, target_eligible, target_total = _load(
        args.cache_root, args.target_cohort, args.model, args.seed, args.min_stay)
    thresholds = np.linspace(0.01, 0.99, args.thresholds)
    rows = []
    for budget in args.budgets:
        for procedure in args.procedures:
            internal = evaluate_at_budget(
                scores, onset, lengths, calibration, test, budget=budget,
                thresholds=thresholds, procedure=procedure, alpha=args.alpha,
                refractory=args.refractory, escalation=args.escalation,
                min_stay_hours=args.min_stay,
                lead_window_hours=args.lead_window)
            external = evaluate_threshold(
                [target_scores[i] for i in range(len(target_scores))],
                target_onset, target_lengths, internal["threshold"],
                refractory=args.refractory, escalation=args.escalation,
                lead_window_hours=args.lead_window)
            rows.append({
                "budget": float(budget), "procedure": procedure,
                "threshold": internal["threshold"],
                "internal_rate": internal["alarm_rate"],
                "internal_sensitivity": internal["sensitivity"],
                "internal_patient_violation": internal["patient_violation"],
                "internal_ppv": internal["ppv"], "internal_nna": internal["nna"],
                "certified_risk": internal["certified_risk"],
                "status": internal["status"], "n_feasible": internal["n_feasible"],
                "external_rate": external["alarm_rate"],
                "external_sensitivity": external["sensitivity"],
                "external_ppv": external["ppv"], "external_nna": external["nna"],
            })
    output = {"config": {
        "source_cohort": args.source_cohort, "target_cohort": args.target_cohort,
        "model": args.model, "seed": args.seed, "alpha": args.alpha,
        "budgets": [float(b) for b in args.budgets],
        "procedures": list(args.procedures),
        "n_calibration": int(calibration.size), "n_test": int(test.size),
        "n_target": len(target_scores),
    }, "rows": rows}
    clean = _clean(output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
