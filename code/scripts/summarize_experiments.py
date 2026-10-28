"""Aggregate completed seed runs into mean and sample-standard-deviation summaries."""

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
            "n": int(values.size), "values": [float(v) for v in values]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-dir", type=Path, required=True)
    parser.add_argument("--e2-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=["prefix_lr", "masked_gru"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = {"seeds": args.seeds, "e1": {}, "e2": {}}
    for model in args.models:
        out["e1"][model] = {}
        for scope, key in (("test", "partitions"), ("external", "external")):
            rows = []
            for seed in args.seeds:
                path = args.e1_dir / f"e1_A_{model}_s{seed}.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                if scope == "test":
                    row = data["partitions"]["test"]
                else:
                    row = data["external"]["report"] if data.get("external") else {}
                if row:
                    rows.append({m: row.get(m) for m in ("auroc", "auprc", "ece")})
            if rows:
                out["e1"][model][scope] = {
                    metric: _ms([r[metric] for r in rows])
                    for metric in ("auroc", "auprc", "ece")}

    for model in args.models:
        out["e2"][model] = {}
        for seed in args.seeds:
            path = args.e2_dir / f"e2_A_lead6_{model}_s{seed}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for scope in ("internal", "external"):
                if scope not in data:
                    continue
                table = out["e2"][model].setdefault(scope, {})
                for row in data[scope]:
                    key = str(row["budget"])
                    table.setdefault(key, {}).setdefault("alarm_rate", []).append(row["alarm_rate"])
                    table[key].setdefault("sensitivity", []).append(row["sensitivity"])
                    table[key].setdefault("median_lead_time", []).append(row["median_lead_time"])
                    table[key].setdefault("patient_violation", []).append(row["patient_violation"])
        for scope, table in out["e2"][model].items():
            for key, values in table.items():
                table[key] = {metric: _ms(vals) for metric, vals in values.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
