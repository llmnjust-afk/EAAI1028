"""Repeated-split coverage diagnostics for the alarm-budget certificate.

The single-split experiments answer whether one realised test path exceeded the
budget.  That is not the same as the frequentist statement the certificate
makes.  This module repeatedly partitions the cohort, calibrates on one part and
evaluates on a disjoint part, and reports the empirical probability that the
realised mean alarm rate exceeds the budget, the budget utilisation, the
fallback rate and the operational alarm metrics.  The same loss matrix is
computed once and reused across splits and procedures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .e2_budget import alarm_loss_matrix, evaluate_from_losses


@dataclass(frozen=True)
class SplitFractions:
    calibration: float = 0.5
    validation: float = 0.0
    test: float = 0.5

    def __post_init__(self):
        total = self.calibration + self.validation + self.test
        if not np.isclose(total, 1.0):
            raise ValueError("split fractions must sum to one")
        if min(self.calibration, self.test) <= 0:
            raise ValueError("calibration and test fractions must be positive")


def stratified_split(onset: np.ndarray, lengths: np.ndarray,
                     fractions: SplitFractions, rng: np.random.Generator) -> dict:
    """Split event and non-event patients separately to preserve event counts."""
    onset = np.asarray(onset)
    lengths = np.asarray(lengths)
    event = (onset > 0) & (onset < lengths)
    parts = {"calibration": [], "validation": [], "test": []}
    for mask in (event, ~event):
        indices = np.flatnonzero(mask)
        rng.shuffle(indices)
        n = indices.size
        n_cal = int(round(fractions.calibration * n))
        n_val = int(round(fractions.validation * n))
        n_test = n - n_cal - n_val
        if min(n_cal, n_test) <= 0 and n >= 3:
            n_cal = max(1, n_cal)
            n_test = max(1, n_test)
            n_val = n - n_cal - n_test
        parts["calibration"].append(indices[:n_cal])
        parts["validation"].append(indices[n_cal:n_cal + n_val])
        parts["test"].append(indices[n_cal + n_val:n_cal + n_val + n_test])
    result = {key: np.sort(np.concatenate(value))
              for key, value in parts.items() if value}
    if not result["calibration"].size or not result["test"].size:
        raise ValueError("split produced an empty calibration or test partition")
    return result


def _row(outcome: dict, budget: float, procedure: str, split: int,
         tolerance: float) -> dict:
    rate = float(outcome["alarm_rate"])
    return {
        "split": int(split),
        "budget": float(budget),
        "procedure": procedure,
        "status": outcome["status"],
        "fallback": outcome["status"] == "no_alarm_fallback",
        "alarm_rate": rate,
        "mean_rate_violation": bool(rate > budget + tolerance),
        "mean_excess": float(max(0.0, rate - budget)),
        "budget_utilization": float(rate / budget) if budget > 0 else float("nan"),
        "patient_violation": float(outcome["patient_violation"]),
        "sensitivity": float(outcome["sensitivity"]),
        "median_lead_time": float(outcome["median_lead_time"]),
        "ppv": float(outcome["ppv"]),
        "nna": float(outcome["nna"]),
        "n_alarms": int(outcome["n_alarms"]),
        "n_true_alarms": int(outcome["n_true_alarms"]),
        "n_feasible": int(outcome["n_feasible"]),
        "certified_risk": float(outcome["certified_risk"]),
        "empirical_risk": float(outcome["empirical_risk"]),
        "threshold": float(outcome["threshold"]),
    }


def coverage_study(scores, onset, lengths, thresholds, budgets,
                   procedures=("blocked", "crc", "timestep", "plugin",
                               "ltt_hoeffding_holm", "randomized"),
                   alpha: float = 0.1, n_splits: int = 20,
                   fractions: SplitFractions | None = None,
                   refractory: int = 4, escalation: int = 0,
                   min_stay_hours: float = 24.0, lead_window_hours: float = 12.0,
                   step_hours: float = 1.0, selection: str = "default",
                   n_probability_points: int = 11, seed: int = 0,
                   tolerance: float = 1e-9) -> list[dict]:
    """Run repeated calibration/test splits for every procedure and budget."""
    scores = [np.asarray(seq, dtype=float) for seq in scores]
    onset = np.asarray(onset, dtype=int)
    lengths = np.asarray(lengths, dtype=int)
    thresholds = np.asarray(thresholds, dtype=float)
    fractions = SplitFractions() if fractions is None else fractions
    if fractions.validation > 0 and selection != "validation":
        raise ValueError("a validation partition requires selection='validation'")
    per_day = alarm_loss_matrix(scores, lengths, thresholds,
                                refractory=refractory, escalation=escalation,
                                step_hours=step_hours)
    rng = np.random.default_rng(seed)
    rows = []
    for split in range(n_splits):
        parts = stratified_split(onset, lengths, fractions, rng)
        calibration = parts["calibration"]
        test = parts["test"]
        validation = parts.get("validation")
        for budget in budgets:
            for procedure in procedures:
                outcome = evaluate_from_losses(
                    scores, onset, lengths, calibration, test, per_day,
                    thresholds, budget=budget, alpha=alpha,
                    refractory=refractory, escalation=escalation,
                    lead_window_hours=lead_window_hours, step_hours=step_hours,
                    min_stay_hours=min_stay_hours, procedure=procedure,
                    validation=validation if validation is not None and validation.size else None,
                    selection=selection,
                    n_probability_points=n_probability_points)
                rows.append(_row(outcome, budget, procedure, split, tolerance))
    return rows


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _bootstrap_mean(values, rng, n_boot=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return (float(values.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def summarize_coverage(rows, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Aggregate split-level rows into procedure/budget summaries with CIs."""
    rows = list(rows)
    if not rows:
        return []
    budgets = sorted({float(row["budget"]) for row in rows})
    procedures = sorted({str(row["procedure"]) for row in rows})
    rng = np.random.default_rng(seed)
    summaries = []
    for budget in budgets:
        for procedure in procedures:
            subset = [row for row in rows
                      if np.isclose(row["budget"], budget)
                      and row["procedure"] == procedure]
            if not subset:
                continue
            rate_mean, rate_lo, rate_hi = _bootstrap_mean(
                [row["alarm_rate"] for row in subset], rng, n_boot)
            sens_mean, sens_lo, sens_hi = _bootstrap_mean(
                [row["sensitivity"] for row in subset], rng, n_boot)
            util_mean, util_lo, util_hi = _bootstrap_mean(
                [row["budget_utilization"] for row in subset], rng, n_boot)
            summaries.append({
                "budget": budget,
                "procedure": procedure,
                "n_splits": len(subset),
                "mean_rate_violation": float(np.mean(
                    [row["mean_rate_violation"] for row in subset])),
                "fallback_rate": float(np.mean([row["fallback"] for row in subset])),
                "alarm_rate_mean": rate_mean,
                "alarm_rate_ci": [rate_lo, rate_hi],
                "budget_utilization_mean": util_mean,
                "budget_utilization_ci": [util_lo, util_hi],
                "sensitivity_mean": sens_mean,
                "sensitivity_ci": [sens_lo, sens_hi],
                "patient_violation_mean": float(np.mean(
                    [row["patient_violation"] for row in subset])),
                "ppv_mean": _nanmean([row["ppv"] for row in subset]),
                "nna_mean": _nanmean([row["nna"] for row in subset]),
                "n_feasible_mean": float(np.mean([row["n_feasible"] for row in subset])),
            })
    return summaries
