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
    horizon = int(np.floor(lead_window_hours / step_hours))
    detected, leads, rates = [], [], []

    for i, seq in enumerate(scores):
        fired = run_policy(seq[:lengths[i]], AlarmPolicy(threshold, refractory, escalation))
        hours = lengths[i] * step_hours
        rates.append(float(np.count_nonzero(fired)) * HOURS_PER_DAY / hours)

        t_event = int(onset[i])
        if t_event <= 0 or t_event >= lengths[i]:
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
                       slack: str = "bernstein",
                       min_stay_hours: float | None = None) -> dict:
    """Calibrate on disjoint patients and evaluate only observed follow-up.

    The support bound must be specified independently of calibration/test data.
    The default allows any positive one-step stay and is conservative. A larger
    minimum defines an eligibility restriction which must hold for deployment.
    Events outside observed follow-up (or at admission) are not lead-time cases.
    """
    lengths = np.asarray(lengths)
    onset = np.asarray(onset)
    thresholds = np.asarray(thresholds, dtype=float)
    n = len(scores)
    if (not n or lengths.shape != (n,) or onset.shape != (n,) or
            not np.isfinite(lengths).all() or np.any(lengths <= 0) or
            np.any(lengths != np.floor(lengths)) or not np.isfinite(onset).all() or
            np.any(onset < -1) or np.any(onset != np.floor(onset))):
        raise ValueError("positive integer lengths and integer onsets (-1 for none) required")
    if (not np.isfinite(step_hours) or step_hours <= 0 or
            not np.isfinite(lead_window_hours) or lead_window_hours <= 0):
        raise ValueError("sampling interval and lead window must be finite and positive")
    if (thresholds.ndim != 1 or not thresholds.size or
            not np.isfinite(thresholds).all() or not np.all(np.diff(thresholds) > 0)):
        raise ValueError("threshold grid must be nonempty, finite and strictly increasing")
    lengths, onset = lengths.astype(int), onset.astype(int)
    partitions = []
    for indices in (calibration, test):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not indices.size or not np.isfinite(indices).all() or
                np.any(indices != np.floor(indices)) or np.any(indices < 0) or
                np.any(indices >= n) or np.unique(indices).size != indices.size):
            raise ValueError("partition indices must be nonempty, unique and in range")
        partitions.append(indices.astype(int))
    calibration, test = partitions
    if np.intersect1d(calibration, test).size:
        raise ValueError("calibration and test patients must be disjoint")
    sequences = []
    for i, seq in enumerate(scores):
        seq = np.asarray(seq, dtype=float)
        if seq.ndim != 1 or seq.size < lengths[i]:
            raise ValueError("score sequences must cover each declared follow-up")
        seq = seq[:lengths[i]]
        if np.isinf(seq).any():
            raise ValueError("valid scores must be finite or NaN (missing)")
        sequences.append(seq)
    scores = sequences
    min_stay_hours = step_hours if min_stay_hours is None else min_stay_hours
    l_max = max_alarms_per_day(refractory, step_hours=step_hours,
                               min_stay_hours=min_stay_hours, escalation=escalation)
    if np.any(lengths * step_hours < min_stay_hours):
        raise ValueError("records violate the independently declared minimum stay")
    per_day = np.zeros((len(scores), thresholds.size))
    for i, seq in enumerate(scores):
        hours = lengths[i] * step_hours
        for k, threshold in enumerate(thresholds):
            fired = run_policy(seq, AlarmPolicy(float(threshold), refractory, escalation))
            per_day[i, k] = float(np.count_nonzero(fired)) * HOURS_PER_DAY / hours

    if per_day.max() > l_max + 1e-9:
        raise AssertionError("per-patient loss exceeds L_max; declare the minimum stay")
    calibrator = BlockedConformalCalibrator(budget=budget, alpha=alpha, slack=slack)
    calibrator.fit(per_day[calibration], thresholds, l_max)
    assert calibrator.threshold_ is not None

    outcomes = _policy_outcomes([scores[i] for i in test], onset[test], lengths[test],
                                calibrator.threshold_, refractory, escalation,
                                lead_window_hours, step_hours)
    outcomes.update({
        "budget": float(budget),
        "threshold": float(calibrator.threshold_),
        "patient_violation": calibrator.violation_rate(
            per_day[test, calibrator.diagnostics_["index"]]
            if calibrator.feasible_ else np.zeros(test.size)),
        "certified_risk": calibrator.diagnostics_["certified_risk"],
        "status": calibrator.diagnostics_["status"],
        "l_max": l_max,
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
                     if np.isfinite(r["median_lead_time"]) and np.isfinite(r["alarm_rate"])),
                    key=lambda p: (p[0], -p[1]))
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
