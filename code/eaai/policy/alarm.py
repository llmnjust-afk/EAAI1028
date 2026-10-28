"""Stateful alarm policy described in Eq. (1) of the manuscript."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class AlarmPolicy:
    """A threshold rule with a refractory window and an escalation tier.

    Parameters
    ----------
    threshold:
        Score level at or above which an alarm is a candidate.
    refractory:
        Number of observation steps during which alarms are suppressed after
        a firing. ``0`` recovers the memoryless threshold rule.
    escalation:
        Number of consecutive suppressed exceedances that overrides the
        refractory window. ``0`` disables the override.
    """

    threshold: float
    refractory: int = 0
    escalation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.refractory, (int, np.integer)) or self.refractory < 0:
            raise ValueError("refractory must be non-negative")
        if not isinstance(self.escalation, (int, np.integer)) or self.escalation < 0:
            raise ValueError("escalation must be non-negative")
        if np.isnan(self.threshold):
            raise ValueError("threshold must not be NaN")


def run_policy(scores: np.ndarray, policy: AlarmPolicy) -> np.ndarray:
    """Apply ``policy`` to one patient's score sequence.

    Parameters
    ----------
    scores:
        One-dimensional score sequence, one entry per observation step.
    policy:
        Policy configuration.

    Returns
    -------
    Boolean array of the same length, ``True`` where an alarm fires.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    alarms = np.zeros(s.shape[0], dtype=bool)
    refractory_left = 0
    streak = 0

    for t, value in enumerate(s):
        exceeded = np.isfinite(value) and value >= policy.threshold
        streak = streak + 1 if exceeded else 0
        override = policy.escalation > 0 and streak >= policy.escalation
        if exceeded and (refractory_left == 0 or override):
            alarms[t] = True
            refractory_left = policy.refractory
            streak = 0
        elif refractory_left > 0:
            refractory_left -= 1

    return alarms


def run_policy_batch(scores: np.ndarray, policy: AlarmPolicy) -> np.ndarray:
    """Apply ``policy`` to a batch of score sequences with padding.

    ``scores`` may be a list of 1-D arrays or a padded 2-D array. Padded
    entries must be ``np.nan``; they are treated as non-observations and never
    produce an alarm. Returns a boolean array of the same shape.
    """
    if isinstance(scores, np.ndarray) and scores.ndim == 1:
        seqs = [scores]
    elif isinstance(scores, np.ndarray) and scores.ndim == 2:
        seqs = [row for row in scores]
    else:
        seqs = list(scores)

    out = np.zeros((len(seqs), max((len(s) for s in seqs), default=0)), dtype=bool)
    for i, row in enumerate(seqs):
        row = np.asarray(row, dtype=float)
        if row.ndim != 1:
            raise ValueError("each patient's scores must be one-dimensional")
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        filled = np.where(valid, row, -np.inf)
        fired = run_policy(filled, policy)
        out[i, : fired.shape[0]] = fired & valid
    return out


def alarms_per_day(alarms: np.ndarray, length_hours: float,
                   hours_per_day: float = HOURS_PER_DAY) -> float:
    """Alarm rate for a single patient, expressed per patient-day."""
    if not all(np.isfinite(v) and v > 0 for v in (length_hours, hours_per_day)):
        raise ValueError("length_hours and hours_per_day must be finite and positive")
    return float(np.count_nonzero(alarms)) * hours_per_day / length_hours


def max_alarms_per_day(refractory: int, step_hours: float = 1.0,
                       hours_per_day: float = HOURS_PER_DAY,
                       min_stay_hours: float = 24.0, escalation: int = 0) -> float:
    """Upper bound on alarms per patient-day for a policy with a refractory window.

    Without escalation a policy fires at most every ``refractory + 1`` steps, so
    the rate is at most ``24 / (refractory + 1)`` for a long stay. Short stays
    are the binding case: a six-hour stay permits two alarms, which is eight
    alarms per day, while a 24-hour stay permits only five. The supremum is
    therefore taken over every admissible stay length rather than evaluated at
    one day, which is why ``min_stay_hours`` is a required modelling decision.

    Escalation can shorten that gap and is included in the bound. Declare a
    support-wide minimum independently of calibration/test data. Records shorter
    than that must be excluded before calibration, otherwise the per-patient
    loss can exceed ``L_max`` and the conformal slack is invalid.
    """
    AlarmPolicy(0.0, refractory, escalation)
    if not all(np.isfinite(v) and v > 0 for v in
               (step_hours, hours_per_day, min_stay_hours)):
        raise ValueError("sampling interval, day length and minimum stay must be positive")
    # The override may shorten the minimum firing gap. With a gap g, the
    # rate peaks at n=1+q*g; subsequent peaks decrease towards 24/(g*step).
    # Checking n_min and the next peak is exact, unlike a truncated grid search.
    gap = min(refractory + 1, escalation) if escalation else refractory + 1
    n_min = max(1, int(np.ceil(min_stay_hours / step_hours)))
    next_peak = 1 + ((n_min - 1) // gap + 1) * gap
    rate = lambda n: ((n - 1) // gap + 1) * hours_per_day / (n * step_hours)
    return float(max(rate(n_min), rate(next_peak)))
