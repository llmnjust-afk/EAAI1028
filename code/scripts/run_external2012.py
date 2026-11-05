"""Run the independent PhysioNet 2012 mortality-alarm transport test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.external2012 import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/external2012.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresholds", type=int, default=200)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    report = run(args.data_root, seed=args.seed, budgets=tuple(args.budgets),
                 thresholds=args.thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["cohorts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
