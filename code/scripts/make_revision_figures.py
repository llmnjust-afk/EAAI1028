"""Create the coverage and ward-load figures for the revision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _series(summary, procedure):
    rows = [row for row in summary if row["procedure"] == procedure]
    rows.sort(key=lambda row: row["budget"])
    return (np.array([row["budget"] for row in rows]),
            np.array([row["budget_utilization_mean"] for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path,
                        default=Path("results/coverage/coverage_A_logistic_s0.json"))
    parser.add_argument("--ward", type=Path,
                        default=Path("results/ward/ward_A_logistic_s0.json"))
    parser.add_argument("--out", type=Path, default=Path("figures/revision.pdf"))
    args = parser.parse_args()

    coverage = json.loads(args.coverage.read_text())["summary"]
    ward = json.loads(args.ward.read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)

    styles = {"blocked": ("o", "-"), "ltt_bernstein_holm": ("s", "--"),
              "randomized": ("^", "-."), "plugin": ("D", ":"),
              "timestep": ("v", (0, (3, 1, 1, 1)))}
    for procedure, (marker, linestyle) in styles.items():
        budgets, utilization = _series(coverage, procedure)
        if budgets.size:
            axes[0].plot(budgets, utilization, marker=marker, linestyle=linestyle,
                         label=procedure.replace("_", " "), markersize=4)
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Budget (alarms/patient-day)", fontsize=7)
    axes[0].set_ylabel("Realised rate / budget", fontsize=7)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=5.3, frameon=False, loc="lower right")
    axes[0].set_title("(a) Budget utilisation, site A", fontsize=8)

    for capacity in (1.0, 2.0):
        subset = [row for row in ward
                  if np.isclose(row["capacity_per_hour"], capacity)
                  and np.isclose(row["arrival_rate"], 5.0)]
        subset.sort(key=lambda row: row["budget"])
        budgets = [row["budget"] for row in subset]
        over = [100.0 * row["fraction_hours_over_capacity"] for row in subset]
        axes[1].plot(budgets, over, marker="o" if capacity == 1 else "s",
                     linestyle="-" if capacity == 1 else "--",
                     label=f"{capacity:.0f} alarm/hour", markersize=4)
    axes[1].set_xlabel("Budget (alarms/patient-day)", fontsize=7)
    axes[1].set_ylabel("Hours over capacity (%)", fontsize=7)
    axes[1].legend(fontsize=6, frameon=False, loc="upper left")
    axes[1].set_title("(b) Ward load, 20 beds, 5/day", fontsize=8)

    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.2, linewidth=0.5)
        args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
