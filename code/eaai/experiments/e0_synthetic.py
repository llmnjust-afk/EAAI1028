"""Validation of the alarm-budget guarantee on a synthetic cohort.

Two quantities are reported and they must not be conflated:

``mean_rate_violation``
    Fraction of splits in which the *test-cohort mean* alarm rate exceeds the
    budget. This is a finite-cohort diagnostic, not the population expectation
    controlled by Proposition 1. Reused patients also make split indicators
    dependent; Wilson intervals are descriptive rather than coverage guarantees.

``patient_violation``
    Fraction of individual patients whose own per-day alarm count exceeds the
    budget. A bound on the mean does not bound this tail; it is reported as a
    diagnostic because a ward experiences both.

The timestep-level baseline uses the same point estimator as the blocked
procedure and differs in exactly one respect: it takes the number of timesteps
as the effective sample size. That is what an analyst does when treating
observations as exchangeable, and it shrinks the guard band by roughly
``sqrt(n_steps / n_patients)``. Holding the estimator fixed is what makes the
comparison interpretable: the only thing that changes is the unit of
information.
"""

from __future__ import annotations

import numpy as np

from ..data.synthetic import make_scores, policy_summaries
from ..policy.alarm import HOURS_PER_DAY, max_alarms_per_day
from ..policy.conformal import BlockedConformalCalibrator, TimestepConformalCalibrator


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
    if (not isinstance(n_patients, int) or n_patients < 2 or
            not isinstance(n_splits, int) or n_splits <= 0 or not calibration_sizes or
            any(not isinstance(n, int) or n <= 0 for n in calibration_sizes) or
            not np.isfinite(budget) or budget <= 0 or not alphas):
        raise ValueError("positive cohort/split/calibration sizes, budget and alphas required")
    if max(calibration_sizes) >= n_patients:
        raise ValueError(
            f"calibration size {max(calibration_sizes)} must be smaller than the "
            f"cohort ({n_patients}); otherwise the test partition is empty and the "
            "violation rate is undefined")

    cohort = make_scores(n_patients, seed=seed)
    thresholds = np.linspace(0.05, 0.95, 60)
    per_day, _ = policy_summaries(cohort, thresholds, refractory=refractory)
    # Fixed by the synthetic generator's support, NOT estimated from the cohort.
    l_max = max_alarms_per_day(refractory, min_stay_hours=24.0)
    if per_day.max() > l_max + 1e-9:
        raise AssertionError(
            f"per-patient loss {per_day.max():.3f} exceeds L_max={l_max:.3f}; "
            "the conformal bound would be invalid")

    n_steps = int(np.sum(cohort["lengths"]))
    rng = np.random.default_rng(seed + 1)
    records: dict[tuple[str, int, float], dict[str, list[float]]] = {}

    def bucket(name: str, n: int, alpha: float) -> dict[str, list[float]]:
        key = (name, n, alpha)
        if key not in records:
            records[key] = {"mean_violation": [], "patient_violation": [],
                            "test_mean": [], "threshold": [], "slack": []}
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
                    test_losses = per_day[test, index] if index >= 0 else np.zeros(test.size)
                    store = bucket(f"blocked_{slack}", n, alpha)
                    store["mean_violation"].append(float(test_losses.mean() > budget))
                    store["patient_violation"].append(float((test_losses > budget).mean()))
                    store["test_mean"].append(float(test_losses.mean()))
                    store["threshold"].append(float(calibrator.threshold_))
                    store["slack"].append(float(calibrator.diagnostics_["slack"]))

                naive = TimestepConformalCalibrator(
                    budget=budget, alpha=alpha).fit(
                        per_day[calibration], thresholds,
                        n_steps=int(cohort["lengths"][calibration].sum()), l_max=l_max)
                index = int(naive.diagnostics_["index"])
                test_losses = per_day[test, index] if index >= 0 else np.zeros(test.size)
                store = bucket("timestep_naive", n, alpha)
                store["mean_violation"].append(float(test_losses.mean() > budget))
                store["patient_violation"].append(float((test_losses > budget).mean()))
                store["test_mean"].append(float(test_losses.mean()))
                store["threshold"].append(float(naive.threshold_))
                store["slack"].append(float(naive.diagnostics_["slack"]))

    summary = {}
    for (name, n, alpha), values in records.items():
        mean_rate = np.asarray(values["mean_violation"])
        patient = np.asarray(values["patient_violation"])
        test_mean = np.asarray(values["test_mean"])
        median_threshold = float(np.median(values["threshold"]))
        summary[f"{name}|n={n}|alpha={alpha:g}"] = {
            "procedure": name,
            "calibration_size": n,
            "alpha": float(alpha),
            "mean_rate_violation": float(mean_rate.mean()),
            "mean_rate_ci": wilson_interval(int(mean_rate.sum()), mean_rate.size),
            "patient_violation": float(patient.mean()),
            "mean_test_rate": float(test_mean.mean()),
            "rate_ratio": float(test_mean.mean() / budget),
            "median_threshold": median_threshold if np.isfinite(median_threshold) else None,
            "fallback_fraction": float(np.isinf(values["threshold"]).mean()),
            "median_slack": float(np.median(values["slack"])),
        }

    return {"summary": summary, "budget": budget, "refractory": refractory,
            "l_max": l_max, "n_patients": n_patients, "n_splits": n_splits,
             "n_steps": n_steps, "seed": seed,
             "calibration_sizes": list(calibration_sizes), "alphas": list(alphas),
             "threshold_count": int(thresholds.size),
             "interpretation": "finite reused-cohort diagnostic; intervals are descriptive"}


def format_table(result: dict) -> str:
    """Render the validation table as fixed-width text."""
    lines = [
        f"budget={result['budget']} alarms/patient-day   refractory={result['refractory']}"
        f"   L_max={result['l_max']:.2f}   splits={result['n_splits']}",
        "",
        f"{'procedure':18s} {'n_cal':>6s} {'alpha':>6s} {'P(mean>B)':>10s} "
        f"{'95% CI':>16s} {'P(pt>B)':>8s} {'mean rate':>10s} {'rate/B':>7s} {'slack':>7s} {'fallback':>8s}",
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
            f"{row['rate_ratio']:7.3f} {row['median_slack']:7.3f} {row['fallback_fraction']:8.3f}"
        )
    return "\n".join(lines)
