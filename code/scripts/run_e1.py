"""Command-line driver for the real-cohort score experiment (E1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.experiments.e1_real import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory containing training_setA and training_setB")
    parser.add_argument("--subset", default="A", choices=("A", "B"))
    parser.add_argument("--limit", type=int, default=None,
                        help="optional patient cap for a pipeline smoke test")
    parser.add_argument("--model", default="prefix_lr",
                        choices=("prefix_lr", "lightgbm", "gru"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/e1"))
    parser.add_argument("--external", default="B", choices=("A", "B", "none"))
    parser.add_argument("--external-limit", type=int, default=None,
                        help="0 disables the external cohort")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="torch device for GRU, e.g. cuda")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    external = None if args.external == "none" else args.external
    external_limit = args.external_limit
    if external_limit == 0:
        external, external_limit = None, None

    kwargs = {}
    if args.model == "gru":
        kwargs = {"epochs": args.epochs, "hidden": args.hidden, "device": args.device}
    report = run(args.data_root, subset=args.subset, limit=args.limit,
                 model_name=args.model, seed=args.seed,
                 cache_root=args.cache_root, results_dir=args.results_dir,
                 external_subset=external, external_limit=external_limit,
                 chunk_size=args.chunk_size, **kwargs)
    summary = {name: {key: row.get(key) for key in
                      ("patients", "timesteps", "positive_patients", "auroc", "auprc", "ece")}
               for name, row in report["partitions"].items()}
    if report["external"]:
        summary["external"] = {
            "subset": report["external"]["subset"],
            "patients": report["external"]["patients"],
            **{key: report["external"]["report"].get(key)
               for key in ("timesteps", "positive_patients", "auroc", "auprc", "ece")},
        }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
