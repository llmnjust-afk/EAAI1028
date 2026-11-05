"""Label-free target adaptation by matching the requested alarm rate.

A site that cannot label early-warning events can still measure how often the
policy fires.  This module chooses the most conservative threshold on a warm-up
window of unlabelled target scores whose empirical mean alarm rate does not
exceed the requested budget.  The resulting threshold is a deployment heuristic,
not a distribution-free certificate; the certificate still requires labelled
calibration patients at the target site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..experiments.e2_budget import alarm_loss_matrix


@dataclass(frozen=True)
class RateMatch:
    index: int
    threshold: float
    achieved_rate: float
    target_rate: float
    within_budget: bool


def rate_matching_threshold(scores, lengths, thresholds, target_rate: float,
                            refractory: int = 4, escalation: int = 0,
                            step_hours: float = 1.0) -> RateMatch:
    """Pick the most aggressive threshold whose empirical rate meets the target.

    Thresholds are ordered from most to least alarm-producing, so the first
    eligible index is the lowest threshold that still respects the rate.  The
    loss matrix is the same patient-level alarm-rate matrix used by the
    calibration experiments, so the returned rate is directly comparable with
    the budget reported elsewhere.
    """
    scores = [np.asarray(seq, dtype=float) for seq in scores]
    lengths = np.asarray(lengths, dtype=int)
    thresholds = np.asarray(thresholds, dtype=float)
    if not 0 < target_rate < np.inf:
        raise ValueError("target_rate must be positive and finite")
    per_day = alarm_loss_matrix(scores, lengths, thresholds,
                                refractory=refractory, escalation=escalation,
                                step_hours=step_hours)
    mean = per_day.mean(axis=0)
    eligible = np.flatnonzero(mean <= target_rate)
    index = int(eligible[0]) if eligible.size else 0
    return RateMatch(index=index, threshold=float(thresholds[index]),
                     achieved_rate=float(mean[index]), target_rate=float(target_rate),
                     within_budget=bool(mean[index] <= target_rate))
