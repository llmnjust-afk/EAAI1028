"""Recompute audit seeds or compare a fresh diagnostic with saved references."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaai.experiments.e0_synthetic import run


def compare(a, b, path="root"):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(f"different keys: {path}")
        for key in a:
            compare(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        # JSON encodes runtime tuple intervals as arrays (loaded as lists).
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise AssertionError(f"different list sizes: {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
            raise AssertionError(f"different numbers: {path}: {a} != {b}")
    elif a != b:
        raise AssertionError(f"different values: {path}: {a!r} != {b!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--references", type=Path,
                        default=Path(__file__).resolve().parents[1] / "audit_results")
    args = parser.parse_args()
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    rows = []
    for seed in args.seeds:
        result = run(n_patients=4000, n_splits=200, seed=seed)
        path = args.references / f"e0_seed{seed}.json"
        if args.recompute:
            args.references.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
        else:
            compare(result, json.loads(path.read_text()))
        row = result["summary"]["timestep_naive|n=800|alpha=0.1"]
        rows.append({"seed": seed, "settings": len(result["summary"]),
                     "naive_finite_mean_exceedance_n800_alpha01": row["mean_rate_violation"],
                     "reference_match": not args.recompute})
    print(json.dumps({"mode": "recompute" if args.recompute else "verify",
                      "versions": versions, "runs": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
