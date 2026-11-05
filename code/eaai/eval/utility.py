"""Official PhysioNet 2019 utility score and patient-level net benefit.

The utility implementation follows the challenge definition: for a septic
patient the reward is maximal six hours before clinical onset, decays to zero
at the early/late boundaries, and false alarms are penalised.  The score is
normalised so that an inactive policy scores zero and the optimal oracle scores
one.  This is an evaluation-only module; no model is fitted to it.
"""

from __future__ import annotations

import numpy as np


def prediction_utility_rows(labels, predictions, dt_early: float = -12.0,
                          dt_optimal: float = -6.0, dt_late: float = 3.0,
                          max_u_tp: float = 1.0, min_u_fn: float = -2.0,
                          u_fp: float = -0.05, u_tn: float = 0.0) -> np.ndarray:
    """Return each patient's summed challenge utility.

    ``prediction_utility`` returns the total; this row-wise form is used when
    utilities are combined with patient-blocked confidence bounds.
    """
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("labels and predictions must be equal 2-D arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("labels and predictions must be binary")
    if not dt_early < dt_optimal < dt_late:
        raise ValueError("require dt_early < dt_optimal < dt_late")

    m1 = max_u_tp / (dt_optimal - dt_early)
    b1 = -m1 * dt_early
    m2 = -max_u_tp / (dt_late - dt_optimal)
    b2 = -m2 * dt_late
    m3 = min_u_fn / (dt_late - dt_optimal)
    b3 = -m3 * dt_optimal

    rows = np.zeros(labels.shape[0], dtype=float)
    for i, (label_row, pred_row) in enumerate(zip(labels, predictions)):
        if np.any(label_row):
            t_sepsis = float(np.argmax(label_row) - dt_optimal)
        else:
            t_sepsis = float("inf")
        n = label_row.size
        u = np.zeros(n, dtype=float)
        if np.isfinite(t_sepsis):
            upper = min(n, int(np.floor(t_sepsis + dt_late)) + 1)
            t = np.arange(upper, dtype=float)
            pred = pred_row[:upper].astype(bool)
            tp_early = pred & (t <= t_sepsis + dt_optimal)
            tp_late = pred & (t > t_sepsis + dt_optimal)
            fn_early = (~pred) & (t <= t_sepsis + dt_optimal)
            fn_late = (~pred) & (t > t_sepsis + dt_optimal)
            u[:upper] = np.where(
                tp_early, np.maximum(m1 * (t - t_sepsis) + b1, u_fp),
                np.where(tp_late, m2 * (t - t_sepsis) + b2,
                         np.where(fn_early, 0.0,
                                  np.where(fn_late,
                                           m3 * (t - t_sepsis) + b3, 0.0))))
        else:
            u = np.where(pred_row.astype(bool), u_fp, u_tn)
        rows[i] = float(u.sum())
    return rows


def prediction_utility(labels, predictions, **kwargs) -> float:
    """Return the summed challenge utility for binary predictions."""
    return float(prediction_utility_rows(labels, predictions, **kwargs).sum())


def normalized_utility_rows(labels, predictions, **kwargs) -> np.ndarray:
    """Return patient-level contributions to the normalised utility."""
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("labels and predictions must be equal 2-D arrays")
    no_predictions = np.zeros_like(predictions)
    optimal = np.zeros_like(predictions)
    for i, label_row in enumerate(labels):
        if np.any(label_row):
            optimal[i, int(np.argmax(label_row))] = 1.0
    denominator = (prediction_utility(labels, optimal, **kwargs)
                   - prediction_utility(labels, no_predictions, **kwargs))
    if denominator <= 0:
        raise ValueError("optimal utility must exceed inactive utility")
    return ((prediction_utility_rows(labels, predictions, **kwargs)
             - prediction_utility_rows(labels, no_predictions, **kwargs))
            / denominator)


def normalized_utility(labels, predictions, **kwargs) -> float:
    """Challenge-normalised utility: zero for inactivity, one for the oracle."""
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    no_predictions = np.zeros_like(predictions)
    optimal = np.zeros_like(predictions)
    for i, label_row in enumerate(labels):
        if np.any(label_row):
            onset = int(np.argmax(label_row))
            optimal[i, onset] = 1.0
    denominator = (prediction_utility(labels, optimal, **kwargs)
                   - prediction_utility(labels, no_predictions, **kwargs))
    if denominator <= 0:
        raise ValueError("optimal utility must exceed inactive utility")
    numerator = (prediction_utility(labels, predictions, **kwargs)
                 - prediction_utility(labels, no_predictions, **kwargs))
    return float(numerator / denominator)


def net_benefit(sensitivity: float, specificity: float, prevalence: float,
                threshold_probability: float) -> float:
    """Patient-level decision-curve net benefit for a binary alarm rule."""
    if not 0 <= sensitivity <= 1 or not 0 <= specificity <= 1:
        raise ValueError("sensitivity and specificity must lie in [0,1]")
    if not 0 < prevalence < 1 or not 0 < threshold_probability < 1:
        raise ValueError("prevalence and threshold probability must lie in (0,1)")
    odds = threshold_probability / (1.0 - threshold_probability)
    return float(sensitivity * prevalence
                 - (1.0 - specificity) * (1.0 - prevalence) * odds)
