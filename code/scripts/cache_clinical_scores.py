"""Cache partial NEWS, SIRS and qSOFA scores for the 2019 A/B cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.data.cohort import clinical_score_batch_multi
from eaai.scores.api import ScoreCache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=["A", "B"])
    parser.add_argument("--instruments", nargs="+", default=["news", "sirs", "qsofa"])
    parser.add_argument("--label-lead-hours", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {}
    for subset in args.subsets:
        batch = clinical_score_batch_multi(
            args.data_root, subset=subset, instruments=args.instruments,
            label_lead_hours=args.label_lead_hours)
        for instrument in args.instruments:
            data = batch[instrument]
            path = ScoreCache(args.cache_root).write(
                f"{subset}_lead{args.label_lead_hours}", instrument, 0,
                data["scores"], data["onset"], data["lengths"])
            report[f"{subset}:{instrument}"] = {
                "patients": int(len(data["lengths"])),
                "cache": str(path),
                "finite_fraction": float(
                    np.isfinite(data["scores"]).sum() / data["scores"].size),
            }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
