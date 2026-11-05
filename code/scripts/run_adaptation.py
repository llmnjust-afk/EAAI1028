"""Compare source transfer, label-free rate matching and target calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.coverage import SplitFractions, stratified_split
from eaai.experiments.e2_budget import (alarm_loss_matrix, evaluate_from_losses,
                                       evaluate_threshold)
from eaai.policy.adaptation import rate_matching_threshold
from eaai.policy.alarm import max_alarms_per_day
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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _load(root, cohort, model, seed, min_stay):
    data = ScoreCache(Path(root)).read(cohort, model, seed)
    scores = [np.asarray(seq, dtype=float) for seq in data["scores"]]
    onset = np.asarray(data["onset"], dtype=int)
    lengths = np.asarray(data["lengths"], dtype=int)
    keep = np.flatnonzero(lengths >= min_stay)
    if not keep.size:
        raise ValueError("no records meet the minimum stay")
    return [scores[i] for i in keep], onset[keep], lengths[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--source-cohort", default="A_lead6")
    parser.add_argument("--target-cohort", default="B_lead6")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--out", type=Path,
                        default=Path("results/adaptation/adaptation.json"))
    args = parser.parse_args()

    thresholds = np.linspace(0.01, 0.99, args.thresholds)
    source_scores, source_onset, source_lengths = _load(
        args.cache_root, args.source_cohort, args.model, args.seed, args.min_stay)
    target_scores, target_onset, target_lengths = _load(
        args.cache_root, args.target_cohort, args.model, args.seed, args.min_stay)

    rng = np.random.default_rng(args.seed)
    source_parts = stratified_split(source_onset, source_lengths,
                                    SplitFractions(0.5, 0.0, 0.5), rng)
    target_parts = stratified_split(target_onset, target_lengths,
                                    SplitFractions(0.5, 0.0, 0.5), rng)
    source_cal = source_parts["calibration"]
    target_adapt = target_parts["calibration"]
    target_test = target_parts["test"]

    l_max = max_alarms_per_day(args.refractory, step_hours=1.0,
                               min_stay_hours=args.min_stay,
                               escalation=args.escalation)
    source_loss = alarm_loss_matrix(source_scores, source_lengths, thresholds,
                                    refractory=args.refractory,
                                    escalation=args.escalation)
    target_loss = alarm_loss_matrix(target_scores, target_lengths, thresholds,
                                    refractory=args.refractory,
                                    escalation=args.escalation)

    rows = []
    for budget in args.budgets:
        source_calibrator = BlockedConformalCalibrator(
            budget=budget, alpha=args.alpha, slack="bernstein")
        source_calibrator.fit(source_loss[source_cal], thresholds, l_max)
        source_threshold = float(source_calibrator.threshold_)

        transfer = evaluate_threshold(
            [target_scores[i] for i in target_test], target_onset[target_test],
            target_lengths[target_test], source_threshold,
            refractory=args.refractory, escalation=args.escalation,
            lead_window_hours=args.lead_window)
        rate_match = rate_matching_threshold(
            [target_scores[i] for i in target_adapt],
            target_lengths[target_adapt], thresholds, target_rate=budget,
            refractory=args.refractory, escalation=args.escalation)
        matched = evaluate_threshold(
            [target_scores[i] for i in target_test], target_onset[target_test],
            target_lengths[target_test], rate_match.threshold,
            refractory=args.refractory, escalation=args.escalation,
            lead_window_hours=args.lead_window)
        oracle = evaluate_from_losses(
            target_scores, target_onset, target_lengths, target_adapt, target_test,
            target_loss, thresholds, budget=budget, alpha=args.alpha,
            refractory=args.refractory, escalation=args.escalation,
            lead_window_hours=args.lead_window, min_stay_hours=args.min_stay,
            procedure="blocked")
        plugin = evaluate_from_losses(
            target_scores, target_onset, target_lengths, target_adapt, target_test,
            target_loss, thresholds, budget=budget, alpha=args.alpha,
            refractory=args.refractory, escalation=args.escalation,
            lead_window_hours=args.lead_window, min_stay_hours=args.min_stay,
            procedure="plugin")

        for method, outcome, threshold in (
                ("source_transfer", transfer, source_threshold),
                ("rate_matched", matched, rate_match.threshold),
                ("oracle_target", oracle, oracle["threshold"]),
                ("plugin_target", plugin, plugin["threshold"])):
            rows.append({
                "budget": float(budget), "method": method,
                "threshold": float(threshold),
                "alarm_rate": outcome["alarm_rate"],
                "sensitivity": outcome["sensitivity"],
                "ppv": outcome["ppv"], "nna": outcome["nna"],
                "median_lead_time": outcome["median_lead_time"],
                "patient_violation": outcome["patient_violation"],
                "certified_risk": outcome["certified_risk"],
                "status": outcome["status"],
                "n_feasible": outcome["n_feasible"],
                "source_within_budget": float(
                    source_calibrator.diagnostics_["certified_risk"] <= budget)
                if method == "source_transfer" else None,
                "rate_match_within_budget": (bool(rate_match.within_budget)
                                             if method == "rate_matched" else None),
            })

    output = {
        "config": {
            "source_cohort": args.source_cohort, "target_cohort": args.target_cohort,
            "model": args.model, "seed": args.seed, "alpha": args.alpha,
            "budgets": [float(b) for b in args.budgets],
            "n_source": len(source_scores), "n_target": len(target_scores),
            "n_target_adapt": int(target_adapt.size),
            "n_target_test": int(target_test.size),
            "n_target_events": int(np.sum((target_onset > 0)
                                          & (target_onset < target_lengths))),
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
