"""Regression tests for defects discovered during the second correctness audit."""
import itertools
import numpy as np
import pandas as pd
import pytest

from eaai.policy.alarm import AlarmPolicy, max_alarms_per_day, run_policy
from eaai.policy.conformal import BlockedConformalCalibrator, TimestepConformalCalibrator
from eaai.data.harmonize import harmonise, pad_records
from eaai.data.splits import four_way, stratified_by_site
from eaai.experiments.e2_budget import evaluate_at_budget, pareto_front
from eaai.scores.baselines import MaskedGRU
from eaai.scores.labels import event_labels, valid_steps


@pytest.mark.parametrize("refractory", [0, 1, 4, 11, 1000])
@pytest.mark.parametrize("escalation", [0, 1, 2, 5])
def test_loss_bound_matches_discrete_supremum_and_includes_override(refractory, escalation):
    for minimum in (1, 6, 12, 24, 1001):
        bound = max_alarms_per_day(refractory, min_stay_hours=minimum, escalation=escalation)
        gap = min(refractory + 1, escalation) if escalation else refractory + 1
        lengths = np.arange(minimum, max(10000, minimum + 2 * gap))
        rates = (((lengths - 1) // gap) + 1) * 24 / lengths
        assert bound == pytest.approx(rates.max())
        # Exercise the actual state machine as well as the closed-form bound.
        for n in (minimum, minimum + gap, minimum + 2 * gap):
            actual = run_policy(np.ones(n), AlarmPolicy(.5, refractory, escalation)).sum() * 24 / n
            assert actual <= bound + 1e-12


def test_state_machine_bound_for_all_small_binary_trajectories():
    for n in range(1, 8):
        for tau, kappa in itertools.product((0, 2, 4), (0, 1, 2, 5)):
            bound = max_alarms_per_day(tau, min_stay_hours=n, escalation=kappa)
            for seq in itertools.product((0., 1.), repeat=n):
                count = run_policy(seq, AlarmPolicy(.5, tau, kappa)).sum()
                assert count * 24 / n <= bound + 1e-12


def test_infinite_threshold_is_an_exact_no_alarm_policy_even_with_invalid_scores():
    assert not run_policy(np.array([1., np.inf, np.nan, -np.inf]), AlarmPolicy(np.inf, 4, 1)).any()


def test_smoothing_never_replaces_the_raw_empirical_certificate():
    losses = np.tile([.6, .2, .8, .1], (10000, 1))
    grid = np.arange(4.)
    cal = BlockedConformalCalibrator(.55, slack="hoeffding", smooth=True).fit(losses, grid, 1.)
    d = cal.diagnostics_
    assert d["index"] == 1
    assert d["smoothed_risk"] != pytest.approx(d["empirical_risk"])
    assert d["certified_risk"] == pytest.approx(losses[:, 1].mean() + d["slack"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.1, 1.01])
def test_invalid_losses_rejected_and_failed_refit_clears_old_policy(bad):
    cal = BlockedConformalCalibrator(1., slack="hoeffding").fit(np.zeros((20, 1)), [0.5], 1.)
    with pytest.raises(ValueError):
        cal.fit(np.array([[bad]]), [.5], 1.)
    assert cal.threshold_ is None and not cal.feasible_ and not cal.diagnostics_


def test_naive_fallback_is_safe_and_uses_same_hoeffding_grid_factor():
    cal = TimestepConformalCalibrator(0.).fit(np.zeros((2, 2)), [0., 1.], 20, 1.)
    assert cal.threshold_ == np.inf and cal.diagnostics_["certified_risk"] == 0
    cal = TimestepConformalCalibrator(10.).fit(np.zeros((2, 2)), [0., 1.], 20, 1.)
    assert cal.diagnostics_["slack"] == pytest.approx(np.sqrt(np.log(2/.1)/40))


def test_harmonizer_to_gru_is_finite_and_prediction_is_causal():
    pytest.importorskip("torch")
    records = [harmonise(pd.DataFrame({"ICULOS": [1, 3, 5, 6],
                   "HR": [70., np.nan, 85., 90.]}), variables=("HR",)) for _ in range(4)]
    batch = pad_records(records)
    assert np.isfinite(batch["delta"]).all()
    assert batch["delta"][0, :, 0].tolist() == [0., 1., 2., 3., 4., 1.]
    labels = event_labels(np.array([5, -1, 5, -1]), batch["lengths"], horizon=2)
    assert np.isnan(labels[0, 5:]).all()
    model = MaskedGRU(hidden=4, epochs=1, batch_size=2).fit(batch, labels)
    scores = model.predict(batch)
    assert np.isfinite(scores).all()
    perturbed = dict(batch)
    perturbed["values"] = batch["values"].copy()
    perturbed["values"][:, 4:, :] += 200.
    assert np.allclose(scores[:, :4], model.predict(perturbed)[:, :4], atol=1e-7)
    # Patient prediction batching must not change scores.
    model.batch_size = 1
    assert np.allclose(scores, model.predict(batch), atol=1e-7)


def _budget_fixture():
    return dict(scores=[np.ones(8)] * 4, onset=np.array([3, -1, 3, 8]),
                lengths=np.array([4] * 4), calibration=np.array([0, 1]),
                test=np.array([2, 3]), thresholds=np.array([.1, .9]),
                refractory=0, slack="hoeffding")


def test_budget_fallback_and_padding_and_censored_events():
    arguments = _budget_fixture()
    row = evaluate_at_budget(**arguments, budget=0.)
    assert row["threshold"] == np.inf and row["status"] == "no_alarm_fallback"
    assert row["alarm_rate"] == row["patient_violation"] == row["certified_risk"] == 0.
    assert row["n_events"] == 1  # event at time 8 is beyond four-step follow-up.
    row = evaluate_at_budget(**arguments, budget=100.)
    assert row["alarm_rate"] == 24.  # padding after length four must not double this.


def test_budget_rejects_overlapping_partitions_and_undeclared_short_stays():
    arguments = _budget_fixture()
    arguments["test"] = np.array([0, 2])
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_at_budget(**arguments, budget=1.)
    arguments["test"] = np.array([2, 3])
    with pytest.raises(ValueError, match="minimum stay"):
        evaluate_at_budget(**arguments, budget=1., min_stay_hours=24.)


def test_pareto_same_rate_retains_only_the_longest_lead():
    rates, leads = pareto_front([{"alarm_rate": 1., "median_lead_time": v} for v in [1., 4., 2.]])
    assert rates.tolist() == [1.] and leads.tolist() == [4.]


def test_string_sites_and_small_partition_counts_are_valid():
    split = stratified_by_site(np.array(["A"] * 9 + ["B"] * 11))
    joined = np.concatenate(list(split.as_dict().values()))
    assert sorted(joined.tolist()) == list(range(20))
    split = four_way(2, ratios=(.26, .26, .26, .22))
    assert sum(len(v) for v in split.as_dict().values()) == 2


@pytest.mark.parametrize("clock", [[1, 1], [2, 1], [1, 1.5], [1, np.nan]])
def test_invalid_clocks_fail_explicitly(clock):
    with pytest.raises(ValueError):
        harmonise(pd.DataFrame({"ICULOS": clock, "HR": [70., 80.]}), variables=("HR",))


def test_event_labels_mask_padding_and_post_onset():
    labels = event_labels(np.array([2, -1]), np.array([4, 3]), horizon=2, width=5)
    assert labels[0, :2].tolist() == [1., 1.]
    assert np.isnan(labels[0, 2:]).all() and np.isnan(labels[1, 3:]).all()
    assert valid_steps([4, 3], 5).sum() == 7


def test_initial_positive_label_is_left_censored_not_shifted_to_fabricated_onset():
    record = harmonise(pd.DataFrame({"HR": [80., 85.], "SepsisLabel": [1, 1]}),
                       variables=("HR",), label_lead_hours=6)
    assert record.onset_index == 0 and record.onset_left_censored
    batch = pad_records([record])
    assert batch["onset_left_censored"].tolist() == [True]
    assert np.isnan(event_labels(batch["onset"], batch["lengths"], horizon=6)).all()


def test_invalid_day_length_and_multidimensional_patient_scores_are_rejected():
    from eaai.policy.alarm import alarms_per_day, run_policy_batch
    for day in (0., -24., np.nan, np.inf):
        with pytest.raises(ValueError):
            alarms_per_day(np.array([True]), 24., hours_per_day=day)
    with pytest.raises(ValueError):
        run_policy_batch(np.ones((2, 3, 4)), AlarmPolicy(.5))
    with pytest.raises(ValueError):
        BlockedConformalCalibrator(1.).violation_rate([-1., 0.])
