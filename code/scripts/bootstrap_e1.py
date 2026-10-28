"""Patient-level bootstrap intervals for E1 discrimination metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.scores.api import ScoreCache
from eaai.scores.labels import event_labels


def _patient_arrays(scores, labels, lengths):
    rows = []
    for i, length in enumerate(lengths):
        usable = np.isfinite(labels[i, :length]) & np.isfinite(scores[i, :length])
        rows.append((labels[i, :length][usable], scores[i, :length][usable]))
    return rows


def _metrics(rows, index):
    ys = [rows[i][0] for i in index if rows[i][0].size]
    ss = [rows[i][1] for i in index if rows[i][1].size]
    if not ys:
        return np.nan, np.nan
    y, s = np.concatenate(ys), np.concatenate(ss)
    if not np.any(y == 0) or not np.any(y == 1):
        return np.nan, np.nan
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _summarise(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "n_valid_resamples": int(values.size),
    }


def _bootstrap(rows_by_model, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(next(iter(rows_by_model.values())))
    draws = {name: {"auroc": [], "auprc": []} for name in rows_by_model}
    paired = {"auroc": [], "auprc": []}
    names = list(rows_by_model)
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        current = {}
        for name in names:
            current[name] = _metrics(rows_by_model[name], index)
            draws[name]["auroc"].append(current[name][0])
            draws[name]["auprc"].append(current[name][1])
        if len(names) == 2:
            paired["auroc"].append(current[names[1]][0] - current[names[0]][0])
            paired["auprc"].append(current[names[1]][1] - current[names[0]][1])
    return draws, paired, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cohort", default="A_lead6")
    parser.add_argument("--external-cohort", default="B_lead6")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["prefix_lr", "masked_gru"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n_boot < 20:
        parser.error("n-boot must be at least 20")

    split = np.load(args.splits)
    output = {"cohort": args.cohort, "n_boot": args.n_boot, "seed": args.seed,
              "test": {}, "external": {}}
    for scope, cohort, index in (
            ("test", args.cohort, split["test"]),
            ("external", args.external_cohort, None)):
        readings = {}
        for name in args.models:
            data = ScoreCache(args.cache_root).read(cohort, name, args.seed)
            labels = event_labels(data["onset"], data["lengths"], horizon=6)
            readings[name] = _patient_arrays(data["scores"], labels, data["lengths"])
        n = len(next(iter(readings.values())))
        chosen = np.arange(n) if index is None else np.asarray(index, dtype=int)
        rows = {name: [readings[name][i] for i in chosen] for name in args.models}
        draws, paired, names = _bootstrap(rows, args.n_boot, args.seed)
        point = {}
        for name in names:
            auroc, auprc = _metrics(rows[name], np.arange(len(chosen)))
            point[name] = {"auroc": auroc, "auprc": auprc}
            point[name]["auroc_boot"] = _summarise(draws[name]["auroc"])
            point[name]["auprc_boot"] = _summarise(draws[name]["auprc"])
        output[scope]["patients"] = int(len(chosen))
        output[scope]["models"] = point
        if len(names) == 2:
            output[scope]["paired"] = {
                "comparison": f"{names[1]}_minus_{names[0]}",
                "auroc": _summarise(paired["auroc"]),
                "auprc": _summarise(paired["auprc"]),
                "p_two_sided_auroc": float(
                    2 * min(np.mean(np.asarray(paired["auroc"]) <= 0),
                            np.mean(np.asarray(paired["auroc"]) >= 0))),
                "p_two_sided_auprc": float(
                    2 * min(np.mean(np.asarray(paired["auprc"]) <= 0),
                            np.mean(np.asarray(paired["auprc"]) >= 0))),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
