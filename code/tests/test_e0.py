import numpy as np
import pytest

from eaai.experiments.e0_synthetic import run, wilson_interval


def test_wilson_interval_brackets_the_proportion():
    lo, hi = wilson_interval(0, 200)
    assert lo == 0.0 and 0.0 < hi < 0.03
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_run_rejects_calibration_sizes_larger_than_the_cohort():
    with pytest.raises(ValueError):
        run(n_patients=300, n_splits=2, calibration_sizes=(400,))


def test_run_reports_the_certified_and_diagnostic_quantities_separately():
    result = run(n_patients=1200, n_splits=20, calibration_sizes=(200,))
    row = result["summary"]["blocked_bernstein|n=200|alpha=0.1"]
    assert 0.0 <= row["mean_rate_violation"] <= 1.0
    assert 0.0 <= row["patient_violation"] <= 1.0
    assert row["rate_ratio"] <= 1.0 + 1e-9 or row["mean_rate_violation"] > 0
    assert result["l_max"] == pytest.approx(6 * 24 / 26)


def test_blocked_procedure_never_exceeds_the_budget_on_the_mean():
    result = run(n_patients=1500, n_splits=30, calibration_sizes=(300,))
    for name, row in result["summary"].items():
        if name.startswith("blocked"):
            assert row["mean_rate_violation"] <= 0.10


def test_naive_effective_sample_size_contains_only_calibration_steps(monkeypatch):
    from eaai.data.synthetic import make_scores
    from eaai.policy.conformal import TimestepConformalCalibrator
    original = TimestepConformalCalibrator.fit
    observed = []

    def recording_fit(self, losses, lambdas, n_steps, l_max):
        observed.append(n_steps)
        return original(self, losses, lambdas, n_steps=n_steps, l_max=l_max)

    monkeypatch.setattr(TimestepConformalCalibrator, "fit", recording_fit)
    run(n_patients=120, n_splits=2, calibration_sizes=(40, 60), alphas=(.1,), seed=5)
    lengths = make_scores(120, seed=5)["lengths"]
    rng = np.random.default_rng(6)
    expected = []
    for _ in range(2):
        order = rng.permutation(120)
        expected.extend(int(lengths[order[:n]].sum()) for n in (40, 60))
    assert observed == expected
    assert max(observed) < int(lengths.sum())
