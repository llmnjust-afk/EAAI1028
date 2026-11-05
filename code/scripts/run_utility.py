"""Official 2019 utility and patient-level decision-curve evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.data.cohort import prepare_2019
from eaai.eval.utility import net_benefit, normalized_utility
from eaai.experiments.e2_budget import evaluate_at_budget
from eaai.policy.alarm import AlarmPolicy, run_policy
from eaai.scores.api import ScoreCache


def _load(cache_root, cohort, model, seed, min_stay):
    data = ScoreCache(Path(cache_root)).read(cohort, model, seed)
    scores = [np.asarray(seq, dtype=float) for seq in data["scores"]]
    onset = np.asarray(data["onset"], dtype=int)
    lengths = np.asarray(data["lengths"], dtype=int)
    keep = np.flatnonzero(lengths >= min_stay)
    return scores, onset, lengths, keep, lengths.size


def _remap(indices, eligible, n_total):
    lookup = np.full(n_total, -1, dtype=int)
    lookup[eligible] = np.arange(eligible.size)
    mapped = lookup[np.asarray(indices, dtype=int)]
    return mapped[mapped >= 0]


def _predictions(sequences, indices, threshold, refractory, escalation):
    out = []
    for i in indices:
        fired = run_policy(sequences[i], AlarmPolicy(float(threshold), refractory,
                                                     escalation))
        out.append(fired.astype(int))
    width = max((len(x) for x in out), default=0)
    return np.array([np.pad(x, (0, width - len(x))) for x in out], dtype=int)


def _patient_metrics(labels, predictions):
    event = np.any(labels > 0, axis=1)
    detected = np.any(predictions > 0, axis=1)
    sensitivity = float(detected[event].mean()) if event.any() else float("nan")
    specificity = float((~detected[~event]).mean()) if (~event).any() else float("nan")
    return sensitivity, specificity, float(event.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--source-cohort", default="A_lead6")
    parser.add_argument("--target-cohort", default="B_lead6")
    parser.add_argument("--source-subset", default="A")
    parser.add_argument("--target-subset", default="B")
    parser.add_argument("--model", default="prefix_lr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=Path, default=Path("results/e1/splits_A_s0.npz"))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--procedures", nargs="+",
                        default=["blocked", "crc", "plugin", "ltt_bernstein_holm"])
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--min-stay", type=float, default=24.0)
    parser.add_argument("--lead-window", type=float, default=12.0)
    parser.add_argument("--out", type=Path, default=Path("results/utility.json"))
    args = parser.parse_args()

    source_batch, _ = prepare_2019(args.data_root, subset=args.source_subset,
                                   label_lead_hours=6, horizon=6)
    target_batch, _ = prepare_2019(args.data_root, subset=args.target_subset,
                                   label_lead_hours=6, horizon=6)
    scores, onset, lengths, eligible, n_total = _load(
        args.cache_root, args.source_cohort, args.model, args.seed, args.min_stay)
    calibration = _remap(np.load(args.splits)["calibration"], eligible, n_total)
    test = _remap(np.load(args.splits)["test"], eligible, n_total)
    target_scores, target_onset, target_lengths, target_eligible, _ = _load(
        args.cache_root, args.target_cohort, args.model, args.seed, args.min_stay)
    thresholds = np.linspace(0.01, 0.99, args.thresholds)
    internal_labels = np.asarray(source_batch["targets"])[eligible]
    external_labels = np.asarray(target_batch["targets"])[target_eligible]

    rows = []
    for budget in args.budgets:
        for procedure in args.procedures:
            result = evaluate_at_budget(
                scores, onset, lengths, calibration, test, budget=budget,
                thresholds=thresholds, procedure=procedure,
                refractory=args.refractory, escalation=args.escalation,
                min_stay_hours=args.min_stay,
                lead_window_hours=args.lead_window)
            threshold = float(result["threshold"])
            for label, seqs, idx, target in (
                    ("internal", scores, test, False),
                    ("external", target_scores, np.arange(len(target_scores)), True)):
                pred = _predictions(seqs, idx, threshold, args.refractory,
                                    args.escalation)
                labels = external_labels if target else internal_labels
                labels = labels[:pred.shape[0], :pred.shape[1]]
                utility = normalized_utility(labels, pred)
                sens, spec, prevalence = _patient_metrics(labels, pred)
                rows.append({
                    "budget": float(budget), "procedure": procedure,
                    "partition": label, "threshold": threshold,
                    "normalized_utility": utility,
                    "patient_sensitivity": sens,
                    "patient_specificity": spec,
                    "prevalence": prevalence,
                    "net_benefit_005": net_benefit(sens, spec, prevalence, 0.05),
                    "net_benefit_010": net_benefit(sens, spec, prevalence, 0.10),
                    "net_benefit_020": net_benefit(sens, spec, prevalence, 0.20),
                })
    output = {"config": {
        "source_cohort": args.source_cohort, "target_cohort": args.target_cohort,
        "model": args.model, "seed": args.seed, "budgets": list(args.budgets),
        "procedures": list(args.procedures), "thresholds": args.thresholds,
    }, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
