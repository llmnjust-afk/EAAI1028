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
        if self.refractory < 0:
            raise ValueError("refractory must be non-negative")
        if self.escalation < 0:
            raise ValueError("escalation must be non-negative")


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
    s = np.asarray(scores, dtype=float).ravel()
    alarms = np.zeros(s.shape[0], dtype=bool)
    refractory_left = 0
    streak = 0

    for t, value in enumerate(s):
        exceeded = value >= policy.threshold
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
    if isinstance(scores, np.ndarray) and scores.ndim == 2:
        seqs = [row for row in scores]
    else:
        seqs = list(scores)

    out = np.zeros((len(seqs), max((len(s) for s in seqs), default=0)), dtype=bool)
    for i, row in enumerate(seqs):
        row = np.asarray(row, dtype=float).ravel()
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
    if length_hours <= 0:
        raise ValueError("length_hours must be positive")
    return float(np.count_nonzero(alarms)) * hours_per_day / length_hours


def max_alarms_per_day(refractory: int, step_hours: float = 1.0,
                       hours_per_day: float = HOURS_PER_DAY) -> float:
    """Tight upper bound on alarms per day for a policy with a refractory window.

    A policy cannot fire more often than once every ``refractory + 1`` steps,
    so with hourly sampling and ``refractory = 4`` the bound is 5 rather than
    24. This constant is the ``L_max`` used by the conformal slack.
    """
    steps_per_day = hours_per_day / step_hours
    return float(np.ceil(steps_per_day / (refractory + 1)))
