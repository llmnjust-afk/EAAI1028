"""Patient-block bootstrap and paired comparisons for budgeted alarm policies.

The operational estimand is the patient-average alarm rate; sensitivity is an
event-level mean among subjects with an onset inside follow-up. Whole target
subjects are resampled with replacement, so within-subject alarm dependence is
preserved. The script compares the proposed patient-blocked selector with a
point-estimate global threshold, a LightGBM scorer, and partial clinical scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e2_budget import alarm_loss_matrix, patient_policy_outcomes
from eaai.policy.alarm import max_alarms_per_day
from eaai.policy.conformal import BlockedConformalCalibrator
from eaai.scores.api import ScoreCache


def _load(cache_root, cohort, model, seed):
    data = ScoreCache(Path(cache_root)).read(cohort, model, seed)
    return data["scores"], data["onset"], data["lengths"]


def _restrict(data, min_stay):
    scores, onset, lengths = data
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


def _threshold(per_day, calibration, grid, budget, procedure, l_max, alpha):
    if procedure == "blocked":
        calibrator = BlockedConformalCalibrator(budget=budget, alpha=alpha,
                                                slack="bernstein")
        calibrator.fit(per_day[calibration], grid, l_max)
        return float(calibrator.threshold_), bool(calibrator.feasible_)
    if procedure == "global":
        empirical = per_day[calibration].mean(axis=0)
        admissible = np.flatnonzero(empirical <= budget)
        return (float(grid[admissible[0]]) if admissible.size else float("inf"),
                bool(admissible.size))
    raise ValueError("procedure must be blocked or global")


def _metrics(outcomes, index, budget):
    rates = outcomes["alarm_rate"][index]
    detected = outcomes["detected"][index]
    event = outcomes["event"][index]
    sensitivity = float(np.nanmean(detected[event])) if event.any() else float("nan")
    return {
        "alarm_rate": float(np.mean(rates)),
        "sensitivity": sensitivity,
        "patient_violation": float(np.mean(rates > budget)),
    }


def _summarise(draws, point):
    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"point": None, "ci95": [None, None], "n_valid": 0}
    return {
        "point": float(point) if np.isfinite(point) else None,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_valid": int(arr.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-cohort", default="A_lead6")
    parser.add_argument("--target-cohort", default="B_lead6")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--models", nargs="+", default=["prefix_lr", "lightgbm"])
    parser.add_argument("--clinical", nargs="+", default=["news", "sirs", "qsofa"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n_boot < 100:
        parser.error("n-boot must be at least 100")

    split = np.load(args.splits)
    clinical = [] if args.clinical == ["none"] else args.clinical
    source_raw = {model: _load(args.cache_root, args.source_cohort, model, args.seed)
                  for model in args.models + clinical}
    target_raw = {model: _load(args.cache_root, args.target_cohort, model, args.seed)
                  for model in args.models + clinical}

    source_parts = {model: _restrict(data, args.min_stay)
                    for model, data in source_raw.items()}
    target_parts = {model: _restrict(data, args.min_stay)
                    for model, data in target_raw.items()}
    source_base = source_parts[args.models[0]][3]
    target_base = target_parts[args.models[0]][3]
    for model in source_parts:
        if not np.array_equal(source_parts[model][3], source_base):
            raise ValueError("source eligibility differs across score sources")
        if not np.array_equal(target_parts[model][3], target_base):
            raise ValueError("target eligibility differs across score sources")

    n_source_total = int(source_raw[args.models[0]][2].size)
    source_cal = _remap(split["calibration"], source_base, n_source_total)
    if not source_cal.size:
        raise ValueError("source calibration partition is empty")
    n_target = target_base.size

    prepared = {}
    for model in source_parts:
        s_scores, s_onset, s_lengths, _ = source_parts[model]
        t_scores, t_onset, t_lengths, _ = target_parts[model]
        combined_scores = _stack(s_scores, t_scores)
        combined_onset = np.concatenate([s_onset, t_onset])
        combined_lengths = np.concatenate([s_lengths, t_lengths])
        if model in clinical:
            finite = np.nanmax(s_scores[source_cal])
            grid = np.arange(0.0, np.floor(finite) + 1.0, 1.0)
            if grid.size < 2:
                grid = np.array([0.0, max(1.0, finite)])
        else:
            grid = np.linspace(0.01, 0.99, 200)
        prepared[model] = {
            "scores": [row for row in combined_scores],
            "onset": combined_onset,
            "lengths": combined_lengths,
            "per_day": alarm_loss_matrix(
                [row for row in combined_scores], combined_lengths, grid,
                refractory=args.refractory, escalation=args.escalation),
            "grid": grid,
            "external": np.arange(s_scores.shape[0], combined_scores.shape[0]),
            "l_max": max_alarms_per_day(args.refractory, min_stay_hours=args.min_stay,
                                        escalation=args.escalation),
        }

    keys = []
    for model in args.models:
        keys.extend([f"blocked:{model}", f"global:{model}"])
    for model in clinical:
        keys.append(f"global:{model}")

    rng = np.random.default_rng(10000 + args.seed)
    draws = {}
    output = {"seed": args.seed, "n_boot": args.n_boot,
              "budgets": args.budgets, "target_patients": int(n_target),
              "results": {}}
    for budget in args.budgets:
        outcomes = {}
        for key in keys:
            procedure, model = key.split(":", 1)
            data = prepared[model]
            threshold, feasible = _threshold(
                data["per_day"], source_cal, data["grid"], budget, procedure,
                data["l_max"], args.alpha)
            out = patient_policy_outcomes(
                data["scores"], data["onset"], data["lengths"], threshold,
                refractory=args.refractory, escalation=args.escalation,
                lead_window_hours=args.lead_window)
            outcomes[key] = {"outcomes": out, "threshold": threshold,
                             "feasible": feasible, "external": data["external"]}
            draws[key] = {"alarm_rate": [], "sensitivity": [], "patient_violation": []}
            for _ in range(args.n_boot):
                sample = rng.integers(0, n_target, size=n_target)
                metric = _metrics(out, data["external"][sample], budget)
                for name in draws[key]:
                    draws[key][name].append(metric[name])

        reference = f"blocked:{args.models[0]}"
        ref = outcomes[reference]
        ref_point = _metrics(ref["outcomes"], ref["external"], budget)
        block = {"reference": reference,
                 "reference_metrics": {
                     name: _summarise(draws[reference][name], ref_point[name])
                     for name in ref_point},
                 "comparisons": {}}
        for key in keys:
            point = _metrics(outcomes[key]["outcomes"], outcomes[key]["external"], budget)
            threshold = outcomes[key]["threshold"]
            entry = {
                "threshold": float(threshold) if np.isfinite(threshold) else None,
                "feasible": outcomes[key]["feasible"],
                "metrics": {name: _summarise(draws[key][name], point[name])
                            for name in point},
                "paired_difference_vs_reference": {},
            }
            if key != reference:
                for name in point:
                    diff = (np.asarray(draws[key][name], dtype=float)
                            - np.asarray(draws[reference][name], dtype=float))
                    diff = diff[np.isfinite(diff)]
                    point_diff = point[name] - ref_point[name]
                    if diff.size:
                        p_two = float(2 * min((np.sum(diff <= 0) + 1) / (diff.size + 1),
                                              (np.sum(diff >= 0) + 1) / (diff.size + 1)))
                        entry["paired_difference_vs_reference"][name] = {
                            "point": float(point_diff) if np.isfinite(point_diff) else None,
                            "ci95": [float(np.quantile(diff, 0.025)),
                                     float(np.quantile(diff, 0.975))],
                            "p_two_sided": min(1.0, p_two),
                            "n_valid": int(diff.size),
                        }
            block["comparisons"][key] = entry
        output["results"][str(budget)] = block

    tests = []
    for budget, block in output["results"].items():
        for key, entry in block["comparisons"].items():
            for metric, diff in entry["paired_difference_vs_reference"].items():
                tests.append((budget, key, metric, diff["p_two_sided"]))
    tests.sort(key=lambda row: row[3])
    for rank, (budget, key, metric, p) in enumerate(tests):
        adjusted = min(1.0, p * (len(tests) - rank))
        output["results"][budget]["comparisons"][key][
            "paired_difference_vs_reference"][metric]["p_holm"] = adjusted

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
