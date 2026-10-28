import numpy as np

from eaai.data.synthetic import make_scores
from eaai.experiments.e2_budget import evaluate_at_budget, pareto_front, sweep, _policy_outcomes


def _fixture(n=400, seed=0):
    cohort = make_scores(n, seed=seed)
    order = np.random.default_rng(1).permutation(n)
    return cohort, order[: n // 2], order[n // 2 :]


def test_larger_budget_does_not_raise_selected_threshold():
    cohort, calibration, test = _fixture()
    rows = sweep(cohort["scores"], cohort["onset"], cohort["lengths"],
                 calibration, test, budgets=(0.5, 1.0, 2.0, 4.0),
                 thresholds=np.linspace(0.05, 0.95, 40))
    thresholds = [r["threshold"] for r in rows]
    assert all(b <= a for a, b in zip(thresholds, thresholds[1:]))


def test_lower_threshold_can_reduce_window_sensitivity_under_refractory_policy():
    scores, onset, lengths = [np.array([.6, .2, .8, .2])], np.array([3]), np.array([4])
    high = _policy_outcomes(scores, onset, lengths, .7, 4, 0, 1., 1.)
    low = _policy_outcomes(scores, onset, lengths, .5, 4, 0, 1., 1.)
    assert high["sensitivity"] == 1. and low["sensitivity"] == 0.


def test_patient_violation_is_reported_separately():
    cohort, calibration, test = _fixture()
    row = evaluate_at_budget(cohort["scores"], cohort["onset"], cohort["lengths"],
                             calibration, test, budget=1.0,
                             thresholds=np.linspace(0.05, 0.95, 40))
    assert 0.0 <= row["patient_violation"] <= 1.0
    assert row["certified_risk"] <= 1.0 + 1e-6


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


def test_alarm_loss_matrix_matches_direct_policy_replay():
    from eaai.experiments.e2_budget import alarm_loss_matrix
    from eaai.policy.alarm import AlarmPolicy, run_policy

    cohort, _, _ = _fixture(n=60, seed=3)
    thresholds = np.linspace(0.1, 0.9, 9)
    matrix = alarm_loss_matrix(cohort["scores"], cohort["lengths"], thresholds,
                               refractory=4, escalation=0)
    for k, threshold in enumerate(thresholds):
        for i, seq in enumerate(cohort["scores"]):
            fired = run_policy(np.asarray(seq)[:cohort["lengths"][i]],
                               AlarmPolicy(float(threshold), 4, 0))
            expected = np.count_nonzero(fired) * 24.0 / cohort["lengths"][i]
            assert matrix[i, k] == expected


def test_timestep_comparator_is_labelled_and_bounded():
    cohort, calibration, test = _fixture()
    row = evaluate_at_budget(cohort["scores"], cohort["onset"], cohort["lengths"],
                             calibration, test, budget=1.0,
                             thresholds=np.linspace(0.05, 0.95, 40),
                             procedure="timestep")
    assert row["procedure"] == "timestep"
    assert 0.0 <= row["patient_violation"] <= 1.0
    assert row["status"] in ("grid_feasible", "no_alarm_fallback")


def test_global_threshold_uses_only_point_estimate():
    cohort, calibration, test = _fixture()
    row = evaluate_at_budget(cohort["scores"], cohort["onset"], cohort["lengths"],
                             calibration, test, budget=1.0,
                             thresholds=np.linspace(0.05, 0.95, 40),
                             procedure="global")
    assert row["procedure"] == "global"
    assert row["status"] in ("grid_feasible", "no_alarm_fallback")
    assert row["certified_risk"] <= 1.0 + 1e-9
