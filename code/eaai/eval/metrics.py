"""Discrimination, calibration, operational and inferential utilities."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def discrimination(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUROC and AUPRC, guarded against single-class inputs."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.min() == y_true.max():
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 15) -> float:
    """Equal-width expected calibration error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue
        gap = abs(y_true[mask].mean() - y_prob[mask].mean())
        total += gap * mask.sum() / y_prob.size
    return float(total)


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Slope of the logistic recalibration of ``y_prob`` on ``y_true``."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    eps = 1e-6
    logit = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
    design = np.column_stack([np.ones_like(logit), logit])
    coef, *_ = np.linalg.lstsq(design, y_true, rcond=None)
    return float(coef[1])


def lead_time_hours(scores: Sequence[float], onset_index: int | None,
                    threshold: float, refractory: int = 0,
                    escalation: int = 0, step_hours: float = 1.0) -> float:
    """Hours between the first alarm and onset, or ``nan`` if not detected earlier."""
    from ..policy.alarm import AlarmPolicy, run_policy

    if onset_index is None or onset_index <= 0:
        return float("nan")
    seq = np.asarray(scores, dtype=float).ravel()
    fired = run_policy(seq[:onset_index], AlarmPolicy(threshold, refractory, escalation))
    hits = np.flatnonzero(fired)
    if hits.size == 0:
        return float("nan")
    return float((onset_index - hits[0]) * step_hours)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Net benefit of treating at ``threshold`` (decision curve analysis)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = y_true.size
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    odds = threshold / (1.0 - threshold)
    return float(tp / n - fp / n * odds)


def net_benefit_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Net benefit across a grid of decision thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.20, 20)
    values = np.array([net_benefit(y_true, y_prob, t) for t in thresholds])
    return np.asarray(thresholds), values


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve built by sorting on confidence descending."""
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct).astype(float).ravel()
    order = np.argsort(-confidence)
    errors = 1.0 - correct[order]
    coverage = np.arange(1, errors.size + 1) / errors.size
    risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return coverage, risk


def bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean,
                 n_resamples: int = 1000, alpha: float = 0.05,
                 groups: np.ndarray | None = None, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over independent units.

    If ``groups`` is supplied, whole groups are resampled with replacement,
    which is the correct unit for patient-level clustering.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    if groups is None:
        draws = rng.integers(0, values.size, size=(n_resamples, values.size))
        stats_ = np.array([statistic(values[d]) for d in draws])
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        buckets = [values[groups == g] for g in unique]
        stats_ = np.empty(n_resamples)
        for r in range(n_resamples):
            pick = rng.integers(0, len(buckets), size=len(buckets))
            stats_[r] = statistic(np.concatenate([buckets[p] for p in pick]))

    point = float(statistic(values))
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.size
    midrank = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midrank
    return out


def delong_test(y_true: np.ndarray, scores_a: np.ndarray,
                scores_b: np.ndarray) -> dict:
    """DeLong test for two correlated AUROCs.

    Returns both areas, the difference, its standard error, the z statistic and
    a two-sided p-value.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()

    positive = y_true == 1
    m, n = int(positive.sum()), int((~positive).sum())
    if m == 0 or n == 0:
        raise ValueError("both classes are required")

    def placement(scores: np.ndarray) -> np.ndarray:
        x, y = scores[positive], scores[~positive]
        tx, ty, tz = _midrank(x), _midrank(y), _midrank(np.concatenate([x, y]))
        tz = tz.reshape(-1)
        return (tz[:m].sum() - m * (m + 1) / 2) / (m * n), tx, ty, tz

    auc_a, tx_a, ty_a, tz_a = placement(a)
    auc_b, tx_b, ty_b, tz_b = placement(b)

    v10_a = (tz_a[:m] - tx_a) / n
    v01_a = 1.0 - (tz_a[m:] - ty_a) / m
    v10_b = (tz_b[:m] - tx_b) / n
    v01_b = 1.0 - (tz_b[m:] - ty_b) / m

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / m + s01 / n
    diff = auc_a - auc_b
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12)))
    z = float(diff / se)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "difference": float(diff),
            "standard_error": se, "z": z, "p_value": p}


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test with the matched-pairs rank-biserial effect size."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("paired samples must have equal length")
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    res = stats.wilcoxon(a, b)
    ranks = stats.rankdata(np.abs(diff))
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    effect = float((positive - negative) / ranks.sum())
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "effect_size": effect}
