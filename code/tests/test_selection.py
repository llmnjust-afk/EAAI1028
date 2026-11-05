import numpy as np
import pytest

from eaai.policy.selection import (
    CRCSelector,
    LTTSelector,
    PluginSelector,
    RandomizedMixtureSelector,
    select_by_utility,
    select_by_validation,
)


def _losses(column_values, n=None):
    values = np.asarray(column_values, dtype=float)
    if n is None:
        n = 100
    return np.tile(values, (n, 1))


def test_plugin_selects_first_empirically_feasible_threshold():
    losses = _losses([0.9, 0.4, 0.1])
    selector = PluginSelector(budget=0.5).fit(losses, [0.1, 0.2, 0.3], l_max=1.0)
    assert selector.diagnostics_["index"] == 1
    assert selector.threshold_ == 0.2
    assert selector.diagnostics_["feasible_indices"] == [1, 2]


def test_plugin_falls_back_when_no_threshold_meets_budget():
    selector = PluginSelector(budget=0.05).fit(_losses([0.2, 0.3]), [0.1, 0.2], 1.0)
    assert selector.diagnostics_["status"] == "no_alarm_fallback"
    assert selector.diagnostics_["index"] == -1
    assert np.isinf(selector.threshold_)


def test_ltt_rejects_a_clearly_feasible_threshold_and_never_an_infeasible_one():
    losses = _losses([0.0, 1.0])
    selector = LTTSelector(budget=0.5, alpha=0.1, slack="hoeffding",
                           correction="holm").fit(losses, [0.1, 0.2], 1.0)
    assert selector.diagnostics_["feasible_indices"] == [0]
    assert selector.threshold_ == 0.1


def test_ltt_is_more_conservative_than_the_plugin():
    losses = _losses([0.4])
    plugin = PluginSelector(budget=0.5).fit(losses, [0.1], 1.0)
    ltt = LTTSelector(budget=0.5, alpha=0.1, slack="hoeffding").fit(
        losses, [0.1], 1.0)
    assert plugin.diagnostics_["feasible"]
    assert not ltt.diagnostics_["feasible"]


def test_holm_rejects_at_least_as_many_as_bonferroni():
    rng = np.random.default_rng(0)
    losses = rng.uniform(0.0, 0.6, size=(200, 5))
    losses[:, 0] = 0.0
    holm = LTTSelector(budget=0.5, alpha=0.1, slack="hoeffding",
                       correction="holm").fit(losses, np.arange(5) / 10, 1.0)
    bonf = LTTSelector(budget=0.5, alpha=0.1, slack="hoeffding",
                       correction="bonferroni").fit(losses, np.arange(5) / 10, 1.0)
    assert len(holm.diagnostics_["feasible_indices"]) >= len(
        bonf.diagnostics_["feasible_indices"])


def test_randomized_mixture_returns_a_certified_pair_within_budget():
    losses = np.zeros((2000, 2))
    selector = RandomizedMixtureSelector(
        budget=0.1, alpha=0.1, slack="hoeffding").fit(losses, [0.1, 0.2], 1.0)
    assert selector.diagnostics_["feasible"]
    assert selector.threshold_low_ <= selector.threshold_high_
    assert selector.diagnostics_["certified_risk"] <= 0.1 + 1e-12


def test_randomized_mixture_requests_a_sensible_probability_grid():
    try:
        RandomizedMixtureSelector(budget=0.1, n_probability_points=1).fit(
            np.zeros((10, 2)), [0.1, 0.2], 1.0)
    except ValueError as exc:
        assert "n_probability_points" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validation_selection_uses_the_disjoint_partition():
    losses = _losses([0.0, 0.0, 0.0])
    calibrator = PluginSelector(budget=0.1).fit(losses, [0.1, 0.2, 0.3], 1.0)
    validation = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]])
    report = select_by_validation(calibrator, validation, [0.1, 0.2, 0.3],
                                  budget=0.1)
    assert report["index"] in (0, 1, 2)
    assert report["status"] == "validation_selected"


def test_validation_selection_rejects_overlapping_inputs():
    losses = _losses([0.0, 0.0])
    calibrator = PluginSelector(budget=0.1).fit(losses, [0.1, 0.2], 1.0)
    try:
        select_by_validation(calibrator, np.zeros((1, 3)), [0.1, 0.2], 0.1)
    except ValueError as exc:
        assert "match the threshold grid" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ltt_bernstein_rejects_when_the_bound_at_delta_half_is_below_budget():
    losses = np.zeros((100, 1))
    selector = LTTSelector(budget=0.5, alpha=0.1, slack="bernstein").fit(
        losses, [0.1], 1.0)
    assert selector.diagnostics_["feasible"]


def test_crc_uses_the_finite_sample_slack():
    losses = _losses([0.0, 0.49], n=100)
    selector = CRCSelector(budget=0.49).fit(losses, [0.1, 0.2], l_max=1.0)
    assert selector.diagnostics_["feasible_indices"] == [0]
    assert selector.threshold_ == 0.1
    assert selector.diagnostics_["certified_risk"] <= 0.49


def test_crc_falls_back_when_the_bound_exceeds_the_budget():
    selector = CRCSelector(budget=0.01).fit(_losses([0.0], n=10), [0.1], 1.0)
    assert selector.diagnostics_["status"] == "no_alarm_fallback"
    assert np.isinf(selector.threshold_)


def test_utility_selection_chooses_the_best_rate_certified_candidate():
    losses = _losses([0.0, 0.0, 0.0])
    calibrator = PluginSelector(budget=0.1).fit(losses, [0.1, 0.2, 0.3], 1.0)
    utility = np.array([[0.0, 0.2, 0.1],
                        [0.0, 0.4, 0.1],
                        [0.0, 0.3, 0.1]])
    report = select_by_utility(calibrator, np.zeros((3, 3)),
                               [0.1, 0.2, 0.3], 0.1, utility)
    assert report["index"] == 1
    assert report["status"] == "utility_selected"
    assert report["validation_utility"] == pytest.approx(0.3)


def test_utility_selection_falls_back_when_no_candidate_is_certified():
    losses = _losses([1.0, 1.0])
    calibrator = PluginSelector(budget=0.1).fit(losses, [0.1, 0.2], 1.0)
    report = select_by_utility(calibrator, np.zeros((2, 2)),
                               [0.1, 0.2], 0.1, np.zeros((2, 2)))
    assert report["index"] == -1
    assert report["status"] == "no_alarm_fallback"
