"""Synthetic cohort generator used to validate the calibration guarantee.

The generator reproduces the two properties that make the problem non-trivial:
scores are autocorrelated within a patient, and alarms cluster around onset, so
a policy with a refractory window produces strongly dependent decision
sequences. It is used only by the validation experiments and the unit tests; no
scientific claim in the paper rests on it.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_scores(n_patients: int, seed: int = 0, prevalence: float = 0.08,
                phi: float = 0.90, noise: float = 0.6,
                min_hours: int = 24, max_hours: int = 120) -> dict:
    """Draw a synthetic cohort of score trajectories with onset times.

    Returns a dict with ``scores`` (list of 1-D arrays), ``onset`` (int array,
    -1 for patients without an event), ``lengths`` and ``event``.
    """
    rng = np.random.default_rng(seed)
    scores: list[np.ndarray] = []
    onset = np.full(n_patients, -1, dtype=int)
    lengths = rng.integers(min_hours, max_hours + 1, size=n_patients)
    event = rng.random(n_patients) < prevalence

    for i in range(n_patients):
        t_max = int(lengths[i])
        patient_effect = rng.normal(0.0, 0.7)
        latent = np.empty(t_max)
        latent[0] = rng.normal(0.0, 1.0)
        for t in range(1, t_max):
            latent[t] = phi * latent[t - 1] + rng.normal(0.0, noise)

        bump = np.zeros(t_max)
        if event[i]:
            t_event = int(rng.integers(max(6, t_max // 3), t_max))
            onset[i] = t_event
            ramp = 1.0 / (1.0 + np.exp(-(np.arange(t_max) - (t_event - 6)) / 2.0))
            bump = 2.5 * ramp

        scores.append(_sigmoid(1.4 * latent + bump + patient_effect - 2.2))

    return {"scores": scores, "onset": onset, "lengths": lengths.astype(int),
            "event": event}


def policy_summaries(cohort: dict, thresholds: np.ndarray,
                     refractory: int = 0, escalation: int = 0,
                     step_hours: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient alarms per day and per-step alarm frequency.

    Both matrices have shape ``(n_patients, n_thresholds)``. The per-day matrix
    is the unit of calibration; the per-step matrix exists only to reproduce the
    invalid timestep-level baseline, which pools dependent observations.
    """
    from ..policy.alarm import HOURS_PER_DAY, AlarmPolicy, run_policy

    per_day = np.zeros((len(cohort["scores"]), thresholds.size))
    per_step = np.zeros_like(per_day)
    for i, seq in enumerate(cohort["scores"]):
        steps = seq.size
        for k, threshold in enumerate(thresholds):
            fired = run_policy(seq, AlarmPolicy(float(threshold), refractory, escalation))
            count = float(np.count_nonzero(fired))
            per_day[i, k] = count * HOURS_PER_DAY / (steps * step_hours)
            per_step[i, k] = count / steps
    return per_day, per_step


def scores_to_matrix(cohort: dict, thresholds: np.ndarray,
                     refractory: int = 0, escalation: int = 0,
                     step_hours: float = 1.0) -> np.ndarray:
    """Per-patient alarms-per-day at every threshold, shape ``(n, n_thresholds)``."""
    per_day, _ = policy_summaries(cohort, thresholds, refractory, escalation, step_hours)
    return per_day
