"""Matched-budget comparison and the lead-time/budget Pareto frontier.

Loss matrices are computed once per score cache and policy configuration, then
reused across calibration/test partitions, budgets and seeds. This is required
for cohort-scale experiments: recomputing every patient's stateful trajectory
for every budget is unnecessarily expensive.
"""

from __future__ import annotations

import numpy as np

from ..policy.alarm import HOURS_PER_DAY, AlarmPolicy, max_alarms_per_day, run_policy
from ..policy.conformal import BlockedConformalCalibrator, TimestepConformalCalibrator


def _validate_sequences(scores, onset, lengths, step_hours=1.0):
    lengths = np.asarray(lengths)
    onset = np.asarray(onset)
    n = len(scores)
    if (not n or lengths.shape != (n,) or onset.shape != (n,) or
            not np.isfinite(lengths).all() or np.any(lengths <= 0) or
            np.any(lengths != np.floor(lengths)) or not np.isfinite(onset).all() or
            np.any(onset < -1) or np.any(onset != np.floor(onset))):
        raise ValueError("positive integer lengths and integer onsets (-1 for none) required")
    if not np.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("sampling interval must be finite and positive")
    lengths, onset = lengths.astype(int), onset.astype(int)
    sequences = []
    for i, seq in enumerate(scores):
        seq = np.asarray(seq, dtype=float)
        if seq.ndim != 1 or seq.size < lengths[i]:
            raise ValueError("score sequences must cover each declared follow-up")
        seq = seq[:lengths[i]]
        if np.isinf(seq).any():
            raise ValueError("valid scores must be finite or NaN (missing)")
        sequences.append(seq)
    return sequences, onset, lengths


def _validate_partitions(n, calibration, test):
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
    return calibration, test


def alarm_loss_matrix(scores, lengths, thresholds, refractory: int = 4,
                      escalation: int = 0, step_hours: float = 1.0) -> np.ndarray:
    """Per-patient alarms/day at every threshold, shape ``(patients, thresholds)``.

    The result depends only on the score cache and the fixed policy family, so it
    must be computed once and reused across budgets and repeated partitions.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if (thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all()
            or not np.all(np.diff(thresholds) > 0)):
        raise ValueError("threshold grid must be nonempty, finite and strictly increasing")
    sequences, _, lengths = _validate_sequences(scores, np.zeros(len(scores)), lengths,
                                                step_hours=step_hours)
    n = len(sequences)
    width = max((seq.size for seq in sequences), default=0)
    padded = np.full((n, width), np.nan)
    for i, seq in enumerate(sequences):
        padded[i, :seq.size] = seq
    per_day = np.zeros((n, thresholds.size))
    for k, threshold in enumerate(thresholds):
        exceeded = np.isfinite(padded) & (padded >= float(threshold))
        fired = np.zeros_like(exceeded, dtype=bool)
        refractory_left = np.zeros(n, dtype=int)
        streak = np.zeros(n, dtype=int)
        for t in range(width):
            hit = exceeded[:, t]
            streak = np.where(hit, streak + 1, 0)
            override = (escalation > 0) & (streak >= escalation)
            fire = hit & ((refractory_left == 0) | override)
            fired[:, t] = fire
            refractory_left = np.where(fire, refractory,
                                       np.maximum(refractory_left - 1, 0))
            streak = np.where(fire, 0, streak)
        per_day[:, k] = (fired.sum(axis=1) * HOURS_PER_DAY /
                         (lengths * step_hours))
    return per_day


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


def evaluate_from_losses(scores, onset, lengths, calibration, test, per_day,
                         thresholds, budget: float, alpha: float = 0.1,
                         refractory: int = 4, escalation: int = 0,
                         lead_window_hours: float = 12.0, step_hours: float = 1.0,
                         slack: str = "bernstein",
                         min_stay_hours: float | None = None,
                         procedure: str = "blocked") -> dict:
    """Calibrate a supplied loss matrix and evaluate a disjoint test partition.

    ``procedure="timestep"`` is the deliberately invalid effective-sample-size
    shortcut, included only as a comparator.
    """
    sequences, onset, lengths = _validate_sequences(
        scores, onset, lengths, step_hours=step_hours)
    calibration, test = _validate_partitions(len(sequences), calibration, test)
    per_day = np.asarray(per_day, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if per_day.shape != (len(sequences), thresholds.size):
        raise ValueError("loss matrix shape does not match patients and thresholds")
    if (not np.isfinite(per_day).all() or np.any(per_day < 0)):
        raise ValueError("loss matrix must be finite and nonnegative")

    min_stay_hours = step_hours if min_stay_hours is None else min_stay_hours
    l_max = max_alarms_per_day(refractory, step_hours=step_hours,
                               min_stay_hours=min_stay_hours, escalation=escalation)
    if np.any(lengths * step_hours < min_stay_hours):
        raise ValueError("records violate the independently declared minimum stay")
    if per_day.max() > l_max + 1e-9:
        raise AssertionError("per-patient loss exceeds L_max; declare the minimum stay")

    if procedure == "timestep":
        calibrator = TimestepConformalCalibrator(budget=budget, alpha=alpha)
        calibrator.fit(per_day[calibration], thresholds,
                       n_steps=int(lengths[calibration].sum()), l_max=l_max)
        feasible = calibrator.diagnostics_["index"] >= 0
    elif procedure == "blocked":
        calibrator = BlockedConformalCalibrator(budget=budget, alpha=alpha, slack=slack)
        calibrator.fit(per_day[calibration], thresholds, l_max)
        feasible = calibrator.feasible_
    else:
        raise ValueError("procedure must be 'blocked' or 'timestep'")
    raw_threshold = calibrator.threshold_
    if raw_threshold is None:
        raise RuntimeError("calibrator did not produce a threshold")
    threshold = float(raw_threshold)
    outcomes = _policy_outcomes([sequences[i] for i in test], onset[test], lengths[test],
                                threshold, refractory, escalation,
                                lead_window_hours, step_hours)
    test_losses = (per_day[test, calibrator.diagnostics_["index"]]
                   if feasible else np.zeros(test.size))
    outcomes.update({
        "budget": float(budget),
        "threshold": threshold,
        "patient_violation": float(np.mean(test_losses > budget)),
        "certified_risk": calibrator.diagnostics_["certified_risk"],
        "status": calibrator.diagnostics_["status"],
        "procedure": procedure,
        "n_feasible": int(calibrator.diagnostics_.get("n_feasible", 0)),
        "l_max": l_max,
    })
    return outcomes


def evaluate_at_budget(scores, onset, lengths, calibration, test, budget: float,
                       thresholds: np.ndarray, alpha: float = 0.1,
                       refractory: int = 4, escalation: int = 0,
                       lead_window_hours: float = 12.0, step_hours: float = 1.0,
                       slack: str = "bernstein",
                       min_stay_hours: float | None = None,
                       procedure: str = "blocked") -> dict:
    """Calibrate on disjoint patients and evaluate only observed follow-up.

    The support bound must be specified independently of calibration/test data.
    The default allows any positive one-step stay and is conservative. A larger
    minimum defines an eligibility restriction which must hold for deployment.
    Events outside observed follow-up (or at admission) are not lead-time cases.
    """
    sequences, onset, lengths = _validate_sequences(scores, onset, lengths,
                                                    step_hours=step_hours)
    calibration, test = _validate_partitions(len(sequences), calibration, test)
    per_day = alarm_loss_matrix(sequences, lengths, thresholds, refractory=refractory,
                                escalation=escalation, step_hours=step_hours)
    return evaluate_from_losses(sequences, onset, lengths, calibration, test, per_day,
                                thresholds, budget=budget, alpha=alpha,
                                refractory=refractory, escalation=escalation,
                                lead_window_hours=lead_window_hours,
                                step_hours=step_hours, slack=slack,
                                min_stay_hours=min_stay_hours,
                                procedure=procedure)


def sweep(scores, onset, lengths, calibration, test,
          budgets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
          thresholds: np.ndarray | None = None,
          **kwargs) -> list[dict]:
    """Evaluate the policy at every budget, reusing one loss matrix."""
    thresholds = np.linspace(0.05, 0.95, 200) if thresholds is None else thresholds
    sequences, onset, lengths = _validate_sequences(
        scores, onset, lengths, step_hours=kwargs.get("step_hours", 1.0))
    calibration, test = _validate_partitions(len(sequences), calibration, test)
    per_day = alarm_loss_matrix(sequences, lengths, thresholds,
                                refractory=kwargs.get("refractory", 4),
                                escalation=kwargs.get("escalation", 0),
                                step_hours=kwargs.get("step_hours", 1.0))
    return [evaluate_from_losses(sequences, onset, lengths, calibration, test, per_day,
                                 thresholds, budget=b, **kwargs)
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
