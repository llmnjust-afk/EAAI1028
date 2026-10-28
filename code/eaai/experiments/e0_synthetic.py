"""Validation of the alarm-budget guarantee on a synthetic cohort.

Two quantities are reported and they must not be conflated:

``mean_rate_violation``
    Fraction of splits in which the *test-cohort mean* alarm rate exceeds the
    budget. This is the quantity Proposition 1 controls.

``patient_violation``
    Fraction of individual patients whose own per-day alarm count exceeds the
    budget. A bound on the mean does not bound this tail; it is reported as a
    diagnostic because a ward experiences both.

The timestep-level baseline selects a threshold from pooled per-step alarm
frequency, which is the natural thing to do if one treats observations as
exchangeable. It is included to show how far that shortcut drifts.
"""

from __future__ import annotations

import numpy as np

from ..data.synthetic import make_scores, policy_summaries
from ..policy.alarm import HOURS_PER_DAY, max_alarms_per_day
from ..policy.conformal import BlockedConformalCalibrator


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (float(max(centre - half, 0.0)), float(centre + half))


def run(n_patients: int = 4000, n_splits: int = 200,
        calibration_sizes: tuple[int, ...] = (100, 200, 800),
        budget: float = 1.0, refractory: int = 4,
        alphas: tuple[float, ...] = (0.05, 0.10, 0.20), seed: int = 0) -> dict:
    """Estimate mean-rate and patient-level violations for all procedures."""
    cohort = make_scores(n_patients, seed=seed)
    thresholds = np.linspace(0.05, 0.95, 60)
    per_day, per_step = policy_summaries(cohort, thresholds, refractory=refractory)
    l_max = max_alarms_per_day(refractory)

    rng = np.random.default_rng(seed + 1)
    records: dict[tuple[str, int, float], dict[str, list[float]]] = {}

    def bucket(name: str, n: int, alpha: float) -> dict[str, list[float]]:
        key = (name, n, alpha)
        if key not in records:
            records[key] = {"mean_violation": [], "patient_violation": [],
                            "test_mean": [], "threshold": []}
        return records[key]

    for _ in range(n_splits):
        order = rng.permutation(n_patients)
        for n in calibration_sizes:
            calibration = order[:n]
            test = order[n:]
            for alpha in alphas:
                for slack in ("bernstein", "hoeffding"):
                    calibrator = BlockedConformalCalibrator(
                        budget=budget, alpha=alpha, slack=slack).fit(
                            per_day[calibration], thresholds, l_max)
                    index = int(calibrator.diagnostics_["index"])
                    test_losses = per_day[test, index]
                    store = bucket(f"blocked_{slack}", n, alpha)
                    store["mean_violation"].append(float(test_losses.mean() > budget))
                    store["patient_violation"].append(float((test_losses > budget).mean()))
                    store["test_mean"].append(float(test_losses.mean()))
                    store["threshold"].append(float(thresholds[index]))

                naive_target = budget / HOURS_PER_DAY
                pooled = per_step[calibration].mean(axis=0)
                feasible = np.flatnonzero(pooled <= naive_target)
                index = int(feasible[0]) if feasible.size else thresholds.size - 1
                test_losses = per_day[test, index]
                store = bucket("timestep_naive", n, alpha)
                store["mean_violation"].append(float(test_losses.mean() > budget))
                store["patient_violation"].append(float((test_losses > budget).mean()))
                store["test_mean"].append(float(test_losses.mean()))
                store["threshold"].append(float(thresholds[index]))

    summary = {}
    for (name, n, alpha), values in records.items():
        mean_rate = np.asarray(values["mean_violation"])
        patient = np.asarray(values["patient_violation"])
        test_mean = np.asarray(values["test_mean"])
        summary[f"{name}|n={n}|alpha={alpha:g}"] = {
            "procedure": name,
            "calibration_size": n,
            "alpha": float(alpha),
            "mean_rate_violation": float(mean_rate.mean()),
            "mean_rate_ci": wilson_interval(int(mean_rate.sum()), mean_rate.size),
            "patient_violation": float(patient.mean()),
            "mean_test_rate": float(test_mean.mean()),
            "rate_ratio": float(test_mean.mean() / budget),
            "median_threshold": float(np.median(values["threshold"])),
        }

    return {"summary": summary, "budget": budget, "refractory": refractory,
            "l_max": l_max, "n_patients": n_patients, "n_splits": n_splits}


def format_table(result: dict) -> str:
    """Render the validation table as fixed-width text."""
    lines = [
        f"budget={result['budget']} alarms/patient-day   refractory={result['refractory']}"
        f"   L_max={result['l_max']:.0f}   splits={result['n_splits']}",
        "",
        f"{'procedure':18s} {'n_cal':>6s} {'alpha':>6s} {'P(mean>B)':>10s} "
        f"{'95% CI':>16s} {'P(pt>B)':>8s} {'mean rate':>10s} {'rate/B':>7s}",
    ]
    order = {"blocked_bernstein": 0, "blocked_hoeffding": 1, "timestep_naive": 2}
    rows = sorted(result["summary"].values(),
                  key=lambda r: (order.get(r["procedure"], 9), r["calibration_size"], r["alpha"]))
    for row in rows:
        lo, hi = row["mean_rate_ci"]
        lines.append(
            f"{row['procedure']:18s} {row['calibration_size']:6d} {row['alpha']:6.2f} "
            f"{row['mean_rate_violation']:10.3f} {f'[{lo:.3f}, {hi:.3f}]':>16s} "
            f"{row['patient_violation']:8.3f} {row['mean_test_rate']:10.3f} "
            f"{row['rate_ratio']:7.3f}"
        )
    return "\n".join(lines)
