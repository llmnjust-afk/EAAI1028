"""Patient-blocked finite-grid upper-confidence risk control.

The unit of calibration is the patient, not the timestep. This is the change
that permits independent-sample concentration despite internal policy state.
The scorer, policy grid and support bound must be fixed independently of risk
calibration; patient blocks must be iid from the deployment population.
Class names retain the project's conformal terminology for API compatibility;
these are Hoeffding/empirical-Bernstein selectors, not standard expected-risk CRC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _validate(losses, lambdas, l_max, budget, alpha):
    losses = np.asarray(losses, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if losses.ndim != 2 or not all(losses.shape):
        raise ValueError("losses must be a nonempty two-dimensional matrix")
    if lambdas.ndim != 1 or losses.shape[1] != lambdas.size:
        raise ValueError("one loss column per one-dimensional threshold is required")
    if not np.isfinite(lambdas).all() or not np.all(np.diff(lambdas) > 0):
        raise ValueError("lambdas must be finite and strictly increasing")
    if not np.isfinite(l_max) or l_max <= 0:
        raise ValueError("l_max must be finite and positive")
    if not np.isfinite(budget) or budget < 0 or not 0 < alpha < 1:
        raise ValueError("budget must be finite and nonnegative; alpha must lie in (0,1)")
    if not np.isfinite(losses).all() or np.any(losses < 0) or np.any(losses > l_max):
        raise ValueError("every loss must be finite and in [0, l_max]; clipping is not valid")
    return losses, lambdas


def _finite_sample_slack(losses: np.ndarray, alpha: float, l_max: float,
                         n_grid: int, kind: str) -> np.ndarray:
    """Per-threshold slack added to the empirical risk.

    ``hoeffding`` is the bound stated in Proposition 1. ``bernstein`` is the
    empirical-Bernstein variant used in the experiments; it is tighter when
    the per-patient alarm counts have low variance, which is the typical case.
    """
    n = losses.shape[0]
    if kind == "hoeffding":
        return np.full(losses.shape[1], l_max * np.sqrt(np.log(n_grid / alpha) / (2.0 * n)))

    if kind == "bernstein":
        if n < 2:
            raise ValueError("bernstein slack needs at least two patients")
        log_term = np.log(2.0 * n_grid / alpha)
        sigma = losses.std(axis=0, ddof=1)
        return sigma * np.sqrt(2.0 * log_term / n) + 7.0 * l_max * log_term / (3.0 * (n - 1))

    raise ValueError(f"unknown slack kind: {kind}")


@dataclass
class BlockedConformalCalibrator:
    """Select the alarm threshold that certifies a per-day alarm budget.

    Parameters
    ----------
    budget:
        Tolerated alarms per patient-day.
    alpha:
        Violation probability to certify.
    slack:
        ``"bernstein"`` (default, tighter) or ``"hoeffding"`` (Proposition 1).
    smooth:
        Compute an optional isotonic diagnostic curve. It does not affect
        selection or certification, which always use the unsmoothed risk.
        Selection does not require a monotonicity assumption on the loss curve.
    """

    budget: float
    alpha: float = 0.1
    slack: str = "bernstein"
    smooth: bool = False
    threshold_: float | None = field(default=None, init=False)
    feasible_: bool = field(default=False, init=False)
    diagnostics_: dict = field(default_factory=dict, init=False)

    def fit(self, losses: np.ndarray, lambdas: np.ndarray,
            l_max: float) -> "BlockedConformalCalibrator":
        """Certify a threshold.

        Parameters
        ----------
        losses:
            Array of shape ``(n_patients, n_thresholds)`` holding each patient's
            alarms per day at every candidate threshold.
        lambdas:
            Candidate thresholds, ascending.
        l_max:
            Upper bound on the per-patient loss, from
            :func:`eaai.policy.alarm.max_alarms_per_day`.
        """
        # Clear a previous fit before validation so an unsuccessful refit cannot
        # leave a stale deployable policy behind.
        self.threshold_, self.feasible_, self.diagnostics_ = None, False, {}
        losses, lambdas = _validate(losses, lambdas, l_max, self.budget, self.alpha)

        empirical = losses.mean(axis=0)
        if self.smooth:
            isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
            smoothed = isotonic.fit_transform(lambdas, empirical)
        else:
            smoothed = empirical

        slack = _finite_sample_slack(losses, self.alpha, l_max,
                                     lambdas.shape[0], self.slack)
        certified = empirical + slack
        feasible = np.flatnonzero(certified <= self.budget)

        self.feasible_ = bool(feasible.size)
        if feasible.size == 0:
            index = -1
        else:
            index = int(feasible[0])

        self.threshold_ = float(lambdas[index]) if self.feasible_ else float("inf")
        self.diagnostics_ = {
            "index": index,
            "empirical_risk": float(empirical[index]) if self.feasible_ else 0.0,
            "smoothed_risk": float(smoothed[index]) if self.feasible_ else 0.0,
            "slack": float(slack[index]) if self.feasible_ else 0.0,
            "certified_risk": float(certified[index]) if self.feasible_ else 0.0,
            "status": "grid_feasible" if self.feasible_ else "no_alarm_fallback",
            "min_grid_upper_bound": float(np.min(certified)),
            "n_patients": int(losses.shape[0]),
            "n_feasible": int(feasible.size),
            "feasible": bool(feasible.size),
            "n_clipped": 0,
            "budget": float(self.budget),
            "alpha": float(self.alpha),
            "l_max": float(l_max),
            "slack_kind": self.slack,
            "smoothed": self.smooth,
        }
        return self

    def violation_rate(self, losses: np.ndarray) -> float:
        """Diagnostic: proportion of patients whose per-day alarm count exceeds the budget.

        This is reported alongside the certified quantity. A bound on the mean
        alarm rate does not bound this tail, and the two are deliberately kept
        separate.
        """
        losses = np.asarray(losses, dtype=float)
        if not losses.size or not np.isfinite(losses).all() or np.any(losses < 0):
            raise ValueError("diagnostic losses must be nonempty, finite and nonnegative")
        return float(np.mean(losses > self.budget))


class TimestepConformalCalibrator:
    """The invalid baseline: count timesteps as independent calibration units.

    The point estimator is identical to :class:`BlockedConformalCalibrator`, so
    the only difference is the sample size entering the finite-sample slack.
    A timestep-level procedure treats the ``N`` dependent observations as ``N``
    independent draws, which shrinks the guard band by roughly
    ``sqrt(N / n_patients)`` relative to the correct one. That shrinkage, not the
    estimator, is what breaks the budget.

    Included only to quantify the drift. It should not be used for deployment.
    """

    def __init__(self, budget: float = 1.0, alpha: float = 0.1) -> None:
        self.budget = budget
        self.alpha = alpha
        self.threshold_: float | None = None
        self.diagnostics_: dict = {}

    def fit(self, losses: np.ndarray, lambdas: np.ndarray, n_steps: int,
            l_max: float) -> "TimestepConformalCalibrator":
        """Select a threshold using ``n_steps`` as the effective sample size.

        ``losses`` are per-patient per-day losses of shape
        ``(n_patients, n_thresholds)``, exactly as for the blocked calibrator, so
        that the two procedures differ only in the slack. ``n_steps`` is the
        total number of observations the analyst believes to be independent,
        i.e. the sum of record lengths.
        """
        self.threshold_, self.diagnostics_ = None, {}
        losses, lambdas = _validate(losses, lambdas, l_max, self.budget, self.alpha)
        if not isinstance(n_steps, (int, np.integer)) or n_steps < losses.shape[0]:
            raise ValueError("n_steps must be an integer at least the calibration patient count")
        empirical = losses.mean(axis=0)
        slack = l_max * np.sqrt(np.log(lambdas.size / self.alpha) / (2.0 * n_steps))
        certified = empirical + slack
        feasible = np.flatnonzero(certified <= self.budget)
        index = int(feasible[0]) if feasible.size else -1

        self.threshold_ = float(lambdas[index]) if feasible.size else float("inf")
        self.diagnostics_ = {
            "index": index,
            "empirical_risk": float(empirical[index]) if feasible.size else 0.0,
            "slack": float(slack) if feasible.size else 0.0,
            "certified_risk": float(certified[index]) if feasible.size else 0.0,
            "status": "grid_feasible" if feasible.size else "no_alarm_fallback",
            "n_steps": int(n_steps),
            "n_feasible": int(feasible.size),
            "effective_n_inflation": float(n_steps / losses.shape[0]),
        }
        return self
