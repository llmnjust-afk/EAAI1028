"""Matched-budget comparison and the lead-time/budget Pareto frontier.

This is the driver behind the headline figure. It operates on cached score
trajectories, so it needs no training and no GPU.
"""

from __future__ import annotations

import numpy as np

from ..policy.alarm import HOURS_PER_DAY, AlarmPolicy, max_alarms_per_day, run_policy
from ..policy.conformal import BlockedConformalCalibrator


def _policy_outcomes(scores: list[np.ndarray], onset: np.ndarray,
                     lengths: np.ndarray, threshold: float,
                     refractory: int, escalation: int,
                     lead_window_hours: float, step_hours: float) -> dict:
    horizon = int(round(lead_window_hours / step_hours))
    detected, leads, rates = [], [], []

    for i, seq in enumerate(scores):
        fired = run_policy(seq, AlarmPolicy(threshold, refractory, escalation))
        hours = lengths[i] * step_hours
        rates.append(float(np.count_nonzero(fired)) * HOURS_PER_DAY / hours)

        t_event = int(onset[i])
        if t_event < 0:
            continue
        start = max(t_event - horizon, 0)
        window = fired[start:t_event]
        hit = np.flatnonzero(window)
        if hit.size:
            detected.append(True)
            leads.append((t_event - (start + hit[0])) * step_hours)
        else:
            detected.append(False)

    return {
        "alarm_rate": float(np.mean(rates)),
        "sensitivity": float(np.mean(detected)) if detected else float("nan"),
        "median_lead_time": float(np.median(leads)) if leads else float("nan"),
        "detected": int(sum(detected)),
        "n_events": len(detected),
    }


def evaluate_at_budget(scores: list[np.ndarray], onset: np.ndarray,
                       lengths: np.ndarray, calibration: np.ndarray,
                       test: np.ndarray, budget: float,
                       thresholds: np.ndarray, alpha: float = 0.1,
                       refractory: int = 4, escalation: int = 0,
                       lead_window_hours: float = 12.0,
                       step_hours: float = 1.0,
                       slack: str = "bernstein") -> dict:
    """Calibrate on ``calibration`` patients and evaluate on ``test`` patients."""
    per_day = np.zeros((len(scores), thresholds.size))
    for i, seq in enumerate(scores):
        hours = lengths[i] * step_hours
        for k, threshold in enumerate(thresholds):
            fired = run_policy(seq, AlarmPolicy(float(threshold), refractory, escalation))
            per_day[i, k] = float(np.count_nonzero(fired)) * HOURS_PER_DAY / hours

    l_max = max_alarms_per_day(refractory, step_hours=step_hours,
                               min_stay_hours=float(np.min(lengths)) * step_hours)
    if per_day.max() > l_max + 1e-9:
        raise AssertionError("per-patient loss exceeds L_max; declare the minimum stay")
    calibrator = BlockedConformalCalibrator(budget=budget, alpha=alpha, slack=slack)
    calibrator.fit(per_day[calibration], thresholds, l_max)

    outcomes = _policy_outcomes([scores[i] for i in test], onset[test], lengths[test],
                                calibrator.threshold_, refractory, escalation,
                                lead_window_hours, step_hours)
    outcomes.update({
        "budget": float(budget),
        "threshold": float(calibrator.threshold_),
        "patient_violation": calibrator.violation_rate(
            per_day[test, calibrator.diagnostics_["index"]]),
        "certified_risk": calibrator.diagnostics_["certified_risk"],
    })
    return outcomes


def sweep(scores: list[np.ndarray], onset: np.ndarray, lengths: np.ndarray,
          calibration: np.ndarray, test: np.ndarray,
          budgets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
          thresholds: np.ndarray | None = None,
          **kwargs) -> list[dict]:
    """Evaluate the policy at every budget in ``budgets``."""
    thresholds = np.linspace(0.05, 0.95, 200) if thresholds is None else thresholds
    return [evaluate_at_budget(scores, onset, lengths, calibration, test,
                               budget=b, thresholds=thresholds, **kwargs)
            for b in budgets]


def pareto_front(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (alarm rate, median lead time) pairs on the non-dominated frontier."""
    points = sorted(((r["alarm_rate"], r["median_lead_time"]) for r in rows
                     if np.isfinite(r["median_lead_time"])), key=lambda p: p[0])
    frontier = []
    best = -np.inf
    for rate, lead in points:
        if lead > best:
            frontier.append((rate, lead))
            best = lead
    if not frontier:
        return np.array([]), np.array([])
    rates, leads = zip(*frontier)
    return np.asarray(rates), np.asarray(leads)
