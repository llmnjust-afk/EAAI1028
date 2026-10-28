import numpy as np
import pytest

from eaai.eval.metrics import (
    bootstrap_ci,
    calibration_slope,
    delong_test,
    discrimination,
    expected_calibration_error,
    lead_time_hours,
    net_benefit,
    paired_wilcoxon,
    risk_coverage,
)


def test_auroc_of_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert discrimination(y, s)["auroc"] == pytest.approx(1.0)


def test_single_class_is_nan_not_error():
    out = discrimination(np.zeros(5), np.random.default_rng(0).random(5))
    assert np.isnan(out["auroc"])


def test_ece_zero_for_calibrated_binary_cases():
    y = np.array([0.0, 1.0])
    p = np.array([0.0, 1.0])
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_bounds():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    assert 0.0 <= expected_calibration_error(y, p) <= 1.0


def test_calibration_slope_of_perfect_predictions_is_large():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < p).astype(float)
    assert calibration_slope(y, p) > 0


def test_net_benefit_matches_manual_computation():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.05])
    threshold = 0.5
    expected = 1 / 4 - 1 / 4 * (threshold / (1 - threshold))
    assert net_benefit(y, p, threshold) == pytest.approx(expected)


def test_lead_time_reports_hours_before_onset():
    scores = np.array([0.1, 0.1, 0.9, 0.2, 0.2, 0.2, 0.2])
    assert lead_time_hours(scores, onset_index=6, threshold=0.5) == pytest.approx(4.0)


def test_lead_time_is_nan_without_detection():
    scores = np.array([0.1, 0.1, 0.1, 0.1])
    assert np.isnan(lead_time_hours(scores, onset_index=3, threshold=0.5))


def test_risk_coverage_is_monotone_in_coverage():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    correct = (rng.random(200) < conf).astype(float)
    coverage, risk = risk_coverage(conf, correct)
    assert np.all(np.diff(coverage) > 0)
    assert risk[0] <= risk[-1] + 1e-9


def test_delong_gives_unit_auc_for_perfect_score():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    good = np.concatenate([rng.normal(0, 1, 50), rng.normal(6, 1, 50)])
    bad = rng.normal(3, 1, 100)
    out = delong_test(y, good, bad)
    assert out["auc_a"] > 0.95
    assert out["p_value"] < 0.01


def test_delong_identical_scores_gives_zero_difference():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    s = rng.normal(size=80)
    out = delong_test(y, s, s)
    assert out["difference"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


def test_paired_wilcoxon_detects_shift():
    a = np.array([0.80, 0.81, 0.79, 0.82, 0.80])
    b = a - 0.05
    out = paired_wilcoxon(a, b)
    assert out["p_value"] < 0.1
    assert out["effect_size"] > 0


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, 300)
    point, lo, hi = bootstrap_ci(values, seed=1)
    assert lo < point < hi
    assert lo < 5.0 < hi
