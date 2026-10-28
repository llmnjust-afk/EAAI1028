"""Discrimination, calibration, operational and inferential utilities."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_inputs(y_true, y_score, probabilities=False):
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if not y.size or y.shape != s.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("nonempty matched binary targets and scores required")
    if not np.isfinite(s).all() or (probabilities and np.any((s < 0) | (s > 1))):
        raise ValueError("scores must be finite; probabilities must lie in [0,1]")
    return y.astype(int), s


def discrimination(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUROC and AUPRC, guarded against single-class inputs."""
    y_true, y_score = _binary_inputs(y_true, y_score)
    if y_true.min() == y_true.max():
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 15) -> float:
    """Equal-width expected calibration error."""
    y_true, y_prob = _binary_inputs(y_true, y_prob, probabilities=True)
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
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
    """Cox calibration slope: the coefficient of ``logit(p)`` in a logistic fit.

    A slope below one means the scores are over-confident, above one means they
    are under-confident. The model is fitted by maximum likelihood rather than by
    least squares, because the binary outcome makes the linear probability model
    a different and unbounded estimator.
    """
    from sklearn.linear_model import LogisticRegression

    y_true, y_prob = _binary_inputs(y_true, y_prob, probabilities=True)
    if np.unique(y_true).size < 2:
        return float("nan")

    eps = 1e-6
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000,
                               fit_intercept=True)
    model.fit(logit, y_true)
    return float(model.coef_[0, 0])


def lead_time_hours(scores: Sequence[float], onset_index: int | None,
                    threshold: float, refractory: int = 0,
                    escalation: int = 0, step_hours: float = 1.0) -> float:
    """Hours between the first alarm and onset, or ``nan`` if not detected earlier."""
    from ..policy.alarm import AlarmPolicy, run_policy

    if not np.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("step_hours must be finite and positive")
    if onset_index is None or onset_index <= 0:
        return float("nan")
    seq = np.asarray(scores, dtype=float).ravel()
    if onset_index >= seq.size:
        return float("nan")
    fired = run_policy(seq[:onset_index], AlarmPolicy(threshold, refractory, escalation))
    hits = np.flatnonzero(fired)
    if hits.size == 0:
        return float("nan")
    return float((onset_index - hits[0]) * step_hours)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Net benefit of treating at ``threshold`` (decision curve analysis)."""
    y_true, y_prob = _binary_inputs(y_true, y_prob, probabilities=True)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
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
    correct, confidence = _binary_inputs(correct, confidence)
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
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all() or
            not isinstance(n_resamples, (int, np.integer)) or n_resamples <= 0 or
            not 0 < alpha < 1):
        raise ValueError("finite nonempty 1-D values, positive resamples and alpha in (0,1) required")
    rng = np.random.default_rng(seed)

    if groups is None:
        stats_ = np.array([statistic(values[rng.integers(0, values.size, size=values.size)])
                           for _ in range(n_resamples)])
    else:
        groups = np.asarray(groups)
        if groups.shape != values.shape or pd_is_missing(groups):
            raise ValueError("groups must match values and contain no missing labels")
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
    y_true, a = _binary_inputs(y_true, scores_a)
    _, b = _binary_inputs(y_true, scores_b)

    positive = y_true == 1
    m, n = int(positive.sum()), int((~positive).sum())
    if m < 2 or n < 2:
        raise ValueError("at least two observations per class are required for variance")

    def placement(scores: np.ndarray) -> tuple:
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
    if not a.size or a.size != b.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired samples must be nonempty, finite and have equal length")
    diff = a - b
    if np.all(diff == 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    # Match SciPy's default zero_method='wilcox': discard zero differences
    # before computing both ranks and the rank-biserial denominator.
    diff = diff[diff != 0]
    res = stats.wilcoxon(diff, zero_method="wilcox")
    ranks = stats.rankdata(np.abs(diff))
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    effect = float((positive - negative) / ranks.sum())
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "effect_size": effect}


def pd_is_missing(groups):
    """Support numeric and string group identifiers without casting them."""
    import pandas as pd
    return bool(pd.isna(groups).any())
