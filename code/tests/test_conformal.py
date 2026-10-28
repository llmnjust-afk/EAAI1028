import numpy as np
import pytest

from eaai.policy.conformal import BlockedConformalCalibrator, TimestepConformalCalibrator


def synthetic_losses(n=600, k=40, seed=0):
    rng = np.random.default_rng(seed)
    lambdas = np.linspace(0.05, 0.95, k)
    base = np.exp(-3.0 * lambdas) * 6.0
    noise = rng.gamma(shape=0.6, scale=1.0, size=(n, k))
    return np.clip(base[None, :] * noise, 0.0, 24.0), lambdas


def test_threshold_is_on_the_grid():
    losses, lambdas = synthetic_losses()
    cal = BlockedConformalCalibrator(budget=1.0, alpha=0.1).fit(losses, lambdas, l_max=5.0)
    assert round(cal.threshold_, 12) in set(np.round(lambdas, 12))


def test_certified_risk_does_not_exceed_budget_when_feasible():
    losses, lambdas = synthetic_losses()
    cal = BlockedConformalCalibrator(budget=2.0, alpha=0.1).fit(losses, lambdas, l_max=5.0)
    diag = cal.diagnostics_
    if diag["n_feasible"] > 0:
        assert diag["certified_risk"] <= 2.0 + 1e-9


def test_infeasible_budget_falls_back_to_most_conservative_threshold():
    losses, lambdas = synthetic_losses()
    cal = BlockedConformalCalibrator(budget=0.0, alpha=0.05).fit(losses, lambdas, l_max=5.0)
    assert cal.diagnostics_["n_feasible"] == 0
    assert cal.threshold_ == pytest.approx(lambdas.max())


def test_hoeffding_slack_is_wider_than_bernstein():
    losses, lambdas = synthetic_losses()
    bern = BlockedConformalCalibrator(budget=1.0, alpha=0.1, slack="bernstein").fit(losses, lambdas, 5.0)
    hoef = BlockedConformalCalibrator(budget=1.0, alpha=0.1, slack="hoeffding").fit(losses, lambdas, 5.0)
    assert hoef.diagnostics_["slack"] >= bern.diagnostics_["slack"]
    assert hoef.threshold_ >= bern.threshold_


def test_rejects_unsorted_grid():
    losses, _ = synthetic_losses()
    with pytest.raises(ValueError):
        BlockedConformalCalibrator(budget=1.0).fit(losses, np.linspace(0.9, 0.1, 40), 5.0)


def test_violation_rate_counts_patients_above_budget():
    losses, lambdas = synthetic_losses(n=100)
    cal = BlockedConformalCalibrator(budget=1.0).fit(losses, lambdas, 5.0)
    index = int(np.searchsorted(lambdas, cal.threshold_))
    column = losses[:, min(index, lambdas.size - 1)]
    assert cal.violation_rate(column) == pytest.approx(float(np.mean(column > 1.0)))


def test_timestep_calibrator_uses_a_smaller_slack_than_blocked():
    losses, lambdas = synthetic_losses()
    blocked = BlockedConformalCalibrator(budget=1.0, alpha=0.1).fit(losses, lambdas, 6.0)
    naive = TimestepConformalCalibrator(budget=1.0, alpha=0.1).fit(
        losses, lambdas, n_steps=int(600 * 60), l_max=6.0)
    assert naive.diagnostics_["slack"] < blocked.diagnostics_["slack"]
    assert naive.diagnostics_["effective_n_inflation"] == pytest.approx(60.0)
    assert naive.threshold_ <= blocked.threshold_


def test_timestep_calibrator_returns_a_grid_point():
    losses, lambdas = synthetic_losses()
    cal = TimestepConformalCalibrator(budget=1.0, alpha=0.1).fit(
        losses, lambdas, n_steps=5000, l_max=6.0)
    assert round(cal.threshold_, 12) in set(np.round(lambdas, 12))


def test_clipping_is_recorded_rather_than_silent():
    losses, lambdas = synthetic_losses()
    cal = BlockedConformalCalibrator(budget=1.0).fit(losses, lambdas, 1.0)
    assert cal.diagnostics_["n_clipped"] > 0
