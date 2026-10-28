"""Aggregate target-site recalibration diagnostics across seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ms(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not values.size:
        return {"mean": None, "sd": None, "n": 0, "values": []}
    return {"mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "n": int(values.size),
            "values": [float(v) for v in values]}


def _collect(rows):
    table = {}
    for row in rows:
        key = str(row["budget"])
        table.setdefault(key, {})
        for metric in ("alarm_rate", "sensitivity", "median_lead_time",
                       "patient_violation", "threshold"):
            table[key].setdefault(metric, []).append(row.get(metric))
    return {key: {metric: _ms(values) for metric, values in metrics.items()}
            for key, metrics in table.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=["prefix_lr"])
    parser.add_argument("--cohort", default="B_lead6")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = {"seeds": args.seeds, "cohort": args.cohort, "models": {}}
    for model in args.models:
        paths = [args.target_dir / f"e2_target_{args.cohort}_{model}_s{seed}.json"
                 for seed in args.seeds]
        paths = [path for path in paths if path.exists()]
        if not paths:
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        out["models"][model] = {
            "runs": len(runs),
            "target_split": {
                "calibration_patients": _ms([r["target_split"]["calibration_patients"]
                                             for r in runs]),
                "test_patients": _ms([r["target_split"]["test_patients"]
                                      for r in runs]),
                "calibration_events": _ms([r["target_split"]["calibration_events"]
                                           for r in runs]),
                "test_events": _ms([r["target_split"]["test_events"]
                                    for r in runs]),
            },
            "target_no_recalibration": _collect(
                [row for run in runs for row in run["target_no_recalibration"]]),
            "target_recalibrated": _collect(
                [row for run in runs for row in run["target_recalibrated"]]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
