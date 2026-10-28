"""Render the E2 budget/sensitivity and budget/realised-rate panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "figures" / "e2_summary.csv"
OUT = ROOT / "figures" / "e2_budget.pdf"

records = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding="utf-8")
budgets = np.array([0.25, 0.50, 1.00, 2.00])
colors = {"A": "#0072B2", "B": "#D55E00"}
labels = {"A": "Site A (calibration/test)", "B": "Site B (external audit)"}

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), constrained_layout=True)
for site in ("A", "B"):
    rows = records[records["site"] == site]
    rows = rows[np.isin(rows["budget"], budgets)]
    order = np.argsort(rows["budget"])
    rows = rows[order]
    axes[0].errorbar(rows["budget"], rows["sensitivity"],
                     yerr=rows["sensitivity_sd"], marker="o", markersize=4.5,
                     linewidth=1.5, capsize=2.5, color=colors[site],
                     label=labels[site])
    axes[1].errorbar(rows["budget"], rows["alarm_rate"],
                     yerr=rows["alarm_sd"], marker="s", markersize=4.5,
                     linewidth=1.5, capsize=2.5, color=colors[site],
                     label=labels[site])

for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.set_xlabel("Requested budget (alarms patient$^{-1}$ day$^{-1}$)")
    ax.grid(True, which="major", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

axes[0].set_ylabel("Event sensitivity")
axes[0].set_ylim(0, 1.0)
axes[0].set_title("(a) Detection at a declared budget", fontsize=9, loc="left")

axes[1].plot(budgets, budgets, linestyle="--", linewidth=1.0, color="#777777",
             label="budget = realised")
axes[1].set_ylabel("Realised alarms patient$^{-1}$ day$^{-1}$")
axes[1].set_ylim(0, 1.7)
axes[1].set_title("(b) Budget utilisation", fontsize=9, loc="left")
axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
print(OUT)
