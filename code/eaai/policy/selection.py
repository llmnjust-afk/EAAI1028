"""Alternative finite-grid selectors for the alarm-budget calibration study.

The original :class:`~eaai.policy.conformal.BlockedConformalCalibrator` applies a
Bonferroni union bound over the fixed threshold grid.  This module adds the
comparators required to separate the *guarantee* from the *point estimator*:

``PluginSelector``
    Select the most aggressive threshold whose empirical loss is at most the
    budget.  This is the usual plug-in rule and has no finite-sample guarantee.
``LTTSelector``
    Learn-then-Test style fixed-grid testing with Hoeffding or empirical
    Bernstein p-values and a Holm step-down correction.  With ``correction=
    "bonferroni"`` it reproduces the original Bonferroni selector.
``RandomizedMixtureSelector``
    A randomized policy that mixes two adjacent thresholds.  The mixture can
    use the budget exactly, which the discrete grid cannot, while retaining a
    finite-grid certificate over the augmented candidate family.
``select_by_validation``
    Choose among certified candidates using a disjoint validation partition.
    The certificate is computed on calibration patients only; validation is
    used solely to break ties, so the coverage statement is unchanged.

All selectors use the same patient-level loss matrix convention as
:mod:`eaai.policy.conformal`: rows are independent patients and columns are
fixed policy candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_losses(losses, lambdas, l_max, budget, alpha):
    losses = np.asarray(losses, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if losses.ndim != 2 or not all(losses.shape):
        raise ValueError("losses must be a nonempty two-dimensional matrix")
    if lambdas.ndim != 1 or losses.shape[1] != lambdas.size:
        raise ValueError("one loss column per one-dimensional candidate is required")
    if not np.isfinite(lambdas).all() or not np.all(np.diff(lambdas) > 0):
        raise ValueError("lambdas must be finite and strictly increasing")
    if not np.isfinite(l_max) or l_max <= 0:
        raise ValueError("l_max must be finite and positive")
    if not np.isfinite(budget) or budget < 0 or not 0 < alpha < 1:
        raise ValueError("budget must be finite and nonnegative; alpha must lie in (0,1)")
    if not np.isfinite(losses).all() or np.any(losses < 0) or np.any(losses > l_max):
        raise ValueError("every loss must be finite and in [0, l_max]; clipping is not valid")
    return losses, lambdas


def _hoeffding_slack(l_max: float, n: int, alpha: float, n_candidates: int) -> float:
    return l_max * np.sqrt(np.log(n_candidates / alpha) / (2.0 * n))


def _eb_slack(losses: np.ndarray, l_max: float, alpha: float,
              n_candidates: int) -> np.ndarray:
    n = losses.shape[0]
    if n < 2:
        raise ValueError("empirical Bernstein needs at least two patients")
    log_term = np.log(2.0 * n_candidates / alpha)
    sigma = losses.std(axis=0, ddof=1)
    return (sigma * np.sqrt(2.0 * log_term / n)
            + 7.0 * l_max * log_term / (3.0 * (n - 1)))


def _hoeffding_pvalues(losses: np.ndarray, budget: float, l_max: float) -> np.ndarray:
    """One-sided p-values for H0: R(lambda) > budget under Hoeffding."""
    n = losses.shape[0]
    mean = losses.mean(axis=0)
    gap = budget - mean
    p = np.ones_like(mean)
    positive = gap > 0
    p[positive] = np.exp(-2.0 * n * gap[positive] ** 2 / l_max ** 2)
    return np.clip(p, 0.0, 1.0)


def _eb_pvalues(losses: np.ndarray, budget: float, l_max: float) -> np.ndarray:
    """Conservative one-sided p-values obtained by inverting the EB bound.

    For a fixed candidate the EB upper bound is monotone decreasing in the
    failure probability ``delta``.  The p-value is the largest ``delta`` for
    which the bound is still at most the budget.  Using it in a Holm procedure
    controls the family-wise error rate because each p-value is valid at its
    own level.
    """
    n = losses.shape[0]
    if n < 2:
        raise ValueError("empirical Bernstein needs at least two patients")
    mean = losses.mean(axis=0)
    sigma = losses.std(axis=0, ddof=1)

    def bound(delta: float) -> np.ndarray:
        log_term = np.log(2.0 / delta)
        return (mean + sigma * np.sqrt(2.0 * log_term / n)
                + 7.0 * l_max * log_term / (3.0 * (n - 1)))

    p = np.ones_like(mean)
    # Candidates already above the budget cannot reject H0.
    for j in np.flatnonzero(mean < budget):
        lo, hi = 1e-12, 0.5
        if bound(lo)[j] > budget:
            p[j] = 0.0
            continue
        if bound(hi)[j] <= budget:
            p[j] = hi
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if bound(mid)[j] <= budget:
                hi = mid
            else:
                lo = mid
        p[j] = hi
    return np.clip(p, 0.0, 1.0)


@dataclass
class PluginSelector:
    """Empirical-risk plug-in threshold with no finite-sample correction."""

    budget: float
    alpha: float = 0.1
    threshold_: float | None = field(default=None, init=False)
    diagnostics_: dict = field(default_factory=dict, init=False)

    def fit(self, losses, lambdas, l_max) -> "PluginSelector":
        losses, lambdas = _as_losses(losses, lambdas, l_max, self.budget, self.alpha)
        mean = losses.mean(axis=0)
        feasible = np.flatnonzero(mean <= self.budget)
        index = int(feasible[0]) if feasible.size else -1
        self.threshold_ = float(lambdas[index]) if feasible.size else float("inf")
        self.diagnostics_ = {
            "index": index,
            "feasible_indices": feasible.tolist(),
            "empirical_risk": float(mean[index]) if feasible.size else 0.0,
            "certified_risk": float(mean[index]) if feasible.size else 0.0,
            "slack": 0.0,
            "status": "grid_feasible" if feasible.size else "no_alarm_fallback",
            "n_feasible": int(feasible.size),
            "n_patients": int(losses.shape[0]),
            "feasible": bool(feasible.size),
            "budget": float(self.budget),
            "alpha": float(self.alpha),
            "l_max": float(l_max),
            "slack_kind": "none",
            "procedure": "plugin",
        }
        return self


@dataclass
class LTTSelector:
    """Learn-then-Test fixed-grid selector with Holm or Bonferroni correction."""

    budget: float
    alpha: float = 0.1
    slack: str = "hoeffding"
    correction: str = "holm"
    threshold_: float | None = field(default=None, init=False)
    diagnostics_: dict = field(default_factory=dict, init=False)

    def fit(self, losses, lambdas, l_max) -> "LTTSelector":
        losses, lambdas = _as_losses(losses, lambdas, l_max, self.budget, self.alpha)
        if self.slack == "hoeffding":
            p = _hoeffding_pvalues(losses, self.budget, l_max)
        elif self.slack == "bernstein":
            p = _eb_pvalues(losses, self.budget, l_max)
        else:
            raise ValueError("slack must be 'hoeffding' or 'bernstein'")
        if self.correction == "holm":
            order = np.argsort(p, kind="stable")
            rejected = np.zeros(p.size, dtype=bool)
            for rank, index in enumerate(order):
                if p[index] <= self.alpha / (p.size - rank):
                    rejected[index] = True
                else:
                    break
        elif self.correction == "bonferroni":
            rejected = p <= self.alpha / p.size
        else:
            raise ValueError("correction must be 'holm' or 'bonferroni'")

        feasible = np.flatnonzero(rejected)
        index = int(feasible[0]) if feasible.size else -1
        mean = losses.mean(axis=0)
        slack = (_hoeffding_slack(l_max, losses.shape[0], self.alpha, p.size)
                 if self.slack == "hoeffding"
                 else float(_eb_slack(losses[:, [index]], l_max, self.alpha, p.size)[0])
                 if feasible.size else 0.0)
        self.threshold_ = float(lambdas[index]) if feasible.size else float("inf")
        self.diagnostics_ = {
            "index": index,
            "feasible_indices": feasible.tolist(),
            "empirical_risk": float(mean[index]) if feasible.size else 0.0,
            "certified_risk": float(mean[index] + slack) if feasible.size else 0.0,
            "slack": float(slack) if feasible.size else 0.0,
            "status": "grid_feasible" if feasible.size else "no_alarm_fallback",
            "n_feasible": int(feasible.size),
            "n_patients": int(losses.shape[0]),
            "feasible": bool(feasible.size),
            "budget": float(self.budget),
            "alpha": float(self.alpha),
            "l_max": float(l_max),
            "slack_kind": self.slack,
            "correction": self.correction,
            "p_values": p.tolist(),
            "procedure": f"ltt_{self.slack}_{self.correction}",
        }
        return self


@dataclass
class RandomizedMixtureSelector:
    """Randomized mixture of adjacent thresholds.

    The deployed policy draws one threshold per patient from
    ``{lambda_low, lambda_high}`` with probabilities ``{p_low, 1 - p_low}``.
    The candidate family is the Cartesian product of adjacent grid pairs and a
    fixed probability grid; the certificate is a union bound over that family.
    The selector returns the certified candidate with the largest empirical
    risk, i.e. the one that uses the budget most closely.
    """

    budget: float
    alpha: float = 0.1
    slack: str = "bernstein"
    n_probability_points: int = 11
    threshold_low_: float | None = field(default=None, init=False)
    threshold_high_: float | None = field(default=None, init=False)
    probability_low_: float | None = field(default=None, init=False)
    diagnostics_: dict = field(default_factory=dict, init=False)

    def fit(self, losses, lambdas, l_max) -> "RandomizedMixtureSelector":
        losses, lambdas = _as_losses(losses, lambdas, l_max, self.budget, self.alpha)
        n, k = losses.shape
        if k < 2:
            raise ValueError("a randomized mixture needs at least two thresholds")
        if (not isinstance(self.n_probability_points, (int, np.integer))
                or self.n_probability_points < 2):
            raise ValueError("n_probability_points must be at least two")
        p_low = np.linspace(0.0, 1.0, self.n_probability_points)
        candidates = []
        for i in range(k - 1):
            for p in p_low:
                candidates.append((i, i + 1, float(p)))
        # Include the pure grid points so the randomized family nests the
        # deterministic policy whenever that is the best certified option.
        for i in range(k):
            candidates.append((i, i, 1.0))
        n_candidates = len(candidates)

        means = np.empty(n_candidates)
        slacks = np.empty(n_candidates)
        for c, (low, high, p) in enumerate(candidates):
            column = p * losses[:, low] + (1.0 - p) * losses[:, high]
            means[c] = column.mean()
            if self.slack == "hoeffding":
                slacks[c] = _hoeffding_slack(l_max, n, self.alpha, n_candidates)
            elif self.slack == "bernstein":
                slacks[c] = float(_eb_slack(column.reshape(-1, 1), l_max,
                                            self.alpha, n_candidates)[0])
            else:
                raise ValueError("slack must be 'hoeffding' or 'bernstein'")
        certified = means + slacks <= self.budget
        if np.any(certified):
            # Maximise budget utilisation among certified candidates; break
            # ties towards the lower expected threshold.
            score = np.where(certified, means, -np.inf)
            best = int(np.argmax(score))
            low, high, p = candidates[best]
            self.threshold_low_ = float(lambdas[low])
            self.threshold_high_ = float(lambdas[high])
            self.probability_low_ = float(p)
            self.threshold_ = self.threshold_low_
            index = low
            index_high = high
            status = "randomized_feasible"
        else:
            self.threshold_low_ = float("inf")
            self.threshold_high_ = float("inf")
            self.probability_low_ = 0.0
            self.threshold_ = float("inf")
            index = -1
            index_high = -1
            best = -1
            status = "no_alarm_fallback"
        self.diagnostics_ = {
            "index": index,
            "index_high": index_high,
            "threshold_low": self.threshold_low_,
            "threshold_high": self.threshold_high_,
            "probability_low": self.probability_low_,
            "empirical_risk": float(means[best]) if best >= 0 else 0.0,
            "certified_risk": float(means[best] + slacks[best]) if best >= 0 else 0.0,
            "slack": float(slacks[best]) if best >= 0 else 0.0,
            "status": status,
            "n_candidates": int(n_candidates),
            "n_patients": int(n),
            "feasible": bool(best >= 0),
            "budget": float(self.budget),
            "alpha": float(self.alpha),
            "l_max": float(l_max),
            "slack_kind": self.slack,
            "procedure": f"randomized_{self.slack}",
        }
        return self


def select_by_validation(calibrator, validation_losses: np.ndarray,
                         thresholds: np.ndarray, budget: float,
                         validation_scores=None, validation_onset=None,
                         validation_lengths=None, refractory: int = 4,
                         escalation: int = 0, lead_window_hours: float = 12.0) -> dict:
    """Re-rank certified candidates by validation sensitivity.

    ``calibrator`` must expose ``diagnostics_["feasible_indices"]``.  The
    validation partition is assumed disjoint from calibration and test.  If
    outcome data are supplied, candidates are ranked by validation sensitivity;
    otherwise by validation risk closest to the budget.  Returns the selected
    index and the validation report; the caller is responsible for applying the
    choice to the test partition.
    """
    indices = list(calibrator.diagnostics_.get("feasible_indices", []))
    thresholds = np.asarray(thresholds, dtype=float)
    validation_losses = np.asarray(validation_losses, dtype=float)
    if validation_losses.ndim != 2 or validation_losses.shape[1] != thresholds.size:
        raise ValueError("validation losses must match the threshold grid")
    if not indices:
        return {"index": -1, "threshold": float("inf"),
                "status": "no_alarm_fallback", "validation_sensitivity": None,
                "validation_risk": None}

    sensitivities = []
    risks = []
    for index in indices:
        risks.append(float(validation_losses[:, index].mean()))
        if validation_scores is not None and validation_onset is not None:
            from ..experiments.e2_budget import patient_policy_outcomes

            outcome = patient_policy_outcomes(
                validation_scores, validation_onset, validation_lengths,
                thresholds[index], refractory=refractory, escalation=escalation,
                lead_window_hours=lead_window_hours)
            event = np.isfinite(outcome["detected"])
            sensitivities.append(float(np.nanmean(outcome["detected"][event]))
                                 if event.any() else float("nan"))
        else:
            sensitivities.append(float("nan"))

    if np.isfinite(sensitivities).any():
        # Highest validation sensitivity; ties go to the higher validation risk
        # (closer to the requested budget).
        order = sorted(range(len(indices)),
                       key=lambda j: (-np.nan_to_num(sensitivities[j], nan=-1.0),
                                      -risks[j]))
    else:
        order = sorted(range(len(indices)),
                       key=lambda j: (abs(risks[j] - budget), risks[j]))
    chosen = order[0]
    return {
        "index": int(indices[chosen]),
        "threshold": float(thresholds[indices[chosen]]),
        "status": "validation_selected",
        "validation_sensitivity": (None if not np.isfinite(sensitivities[chosen])
                                   else float(sensitivities[chosen])),
        "validation_risk": float(risks[chosen]),
        "n_feasible": len(indices),
    }
