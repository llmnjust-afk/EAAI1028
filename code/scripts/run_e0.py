"""Run the synthetic validation of the alarm-budget guarantee.

Usage
-----
    python scripts/run_e0.py --patients 4000 --splits 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e0_synthetic import format_table, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=4000)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--calibration-sizes", type=int, nargs="+", default=[100, 200, 800])
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--refractory", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = run(n_patients=args.patients, n_splits=args.splits,
                 calibration_sizes=tuple(args.calibration_sizes), budget=args.budget,
                 refractory=args.refractory, seed=args.seed)

    print(format_table(result))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
