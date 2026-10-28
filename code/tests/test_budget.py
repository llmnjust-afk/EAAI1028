import numpy as np

from eaai.data.synthetic import make_scores
from eaai.experiments.e2_budget import evaluate_at_budget, pareto_front, sweep


def _fixture(n=400, seed=0):
    cohort = make_scores(n, seed=seed)
    order = np.random.default_rng(1).permutation(n)
    return cohort, order[: n // 2], order[n // 2 :]


def test_budget_sweep_is_monotone_in_alarm_rate():
    cohort, calibration, test = _fixture()
    rows = sweep(cohort["scores"], cohort["onset"], cohort["lengths"],
                 calibration, test, budgets=(0.5, 1.0, 2.0, 4.0),
                 thresholds=np.linspace(0.05, 0.95, 40))
    rates = [r["alarm_rate"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(rates, rates[1:]))


def test_higher_budget_never_reduces_sensitivity():
    cohort, calibration, test = _fixture()
    rows = sweep(cohort["scores"], cohort["onset"], cohort["lengths"],
                 calibration, test, budgets=(0.5, 2.0, 8.0),
                 thresholds=np.linspace(0.05, 0.95, 40))
    sens = [r["sensitivity"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(sens, sens[1:]))


def test_patient_violation_is_reported_separately():
    cohort, calibration, test = _fixture()
    row = evaluate_at_budget(cohort["scores"], cohort["onset"], cohort["lengths"],
                             calibration, test, budget=1.0,
                             thresholds=np.linspace(0.05, 0.95, 40))
    assert 0.0 <= row["patient_violation"] <= 1.0
    assert row["alarm_rate"] <= 1.0 + 1e-6


def test_pareto_front_is_non_dominated():
    rows = [
        {"alarm_rate": 0.5, "median_lead_time": 2.0},
        {"alarm_rate": 1.0, "median_lead_time": 3.0},
        {"alarm_rate": 1.5, "median_lead_time": 2.5},
        {"alarm_rate": 2.0, "median_lead_time": 5.0},
    ]
    rates, leads = pareto_front(rows)
    assert rates.tolist() == [0.5, 1.0, 2.0]
    assert leads.tolist() == [2.0, 3.0, 5.0]


def test_pareto_front_handles_empty_input():
    rates, leads = pareto_front([{"alarm_rate": 1.0, "median_lead_time": float("nan")}])
    assert rates.size == 0 and leads.size == 0
