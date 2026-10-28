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


def test_timestep_calibrator_returns_a_grid_point():
    rng = np.random.default_rng(1)
    lambdas = np.linspace(0.05, 0.95, 30)
    steps = rng.random((2000, 30))
    cal = TimestepConformalCalibrator(alpha=0.1).fit(steps, lambdas)
    assert round(cal.threshold_, 12) in set(np.round(lambdas, 12))
