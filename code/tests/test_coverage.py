import numpy as np

from eaai.data.synthetic import make_scores
from eaai.experiments.coverage import (SplitFractions, coverage_study,
                                       stratified_split, summarize_coverage)


def test_stratified_split_is_disjoint_and_preserves_event_counts():
    cohort = make_scores(200, seed=2)
    parts = stratified_split(cohort["onset"], cohort["lengths"],
                             SplitFractions(0.5, 0.0, 0.5),
                             np.random.default_rng(0))
    calibration, test = parts["calibration"], parts["test"]
    assert not np.intersect1d(calibration, test).size
    assert calibration.size + test.size == 200
    event = (cohort["onset"] > 0) & (cohort["onset"] < cohort["lengths"])
    assert event[calibration].sum() > 0
    assert event[test].sum() > 0


def test_coverage_study_reports_required_fields():
    cohort = make_scores(240, seed=3)
    thresholds = np.linspace(0.05, 0.95, 30)
    rows = coverage_study(
        cohort["scores"], cohort["onset"], cohort["lengths"], thresholds,
        budgets=(1.0,), procedures=("blocked", "plugin", "timestep"),
        n_splits=3, seed=0)
    assert len(rows) == 9
    required = {"split", "budget", "procedure", "mean_rate_violation",
                "budget_utilization", "fallback", "sensitivity", "ppv", "nna"}
    assert required <= set(rows[0])
    assert {row["procedure"] for row in rows} == {"blocked", "plugin", "timestep"}


def test_coverage_summary_has_bootstrap_intervals():
    cohort = make_scores(200, seed=4)
    rows = coverage_study(
        cohort["scores"], cohort["onset"], cohort["lengths"],
        np.linspace(0.05, 0.95, 25), budgets=(1.0,),
        procedures=("blocked", "plugin"), n_splits=3, seed=0)
    summary = summarize_coverage(rows, n_boot=50, seed=0)
    assert len(summary) == 2
    for item in summary:
        assert item["n_splits"] == 3
        assert len(item["alarm_rate_ci"]) == 2
        assert item["alarm_rate_ci"][0] <= item["alarm_rate_ci"][1]


def test_validation_split_requires_validation_selection():
    cohort = make_scores(120, seed=5)
    try:
        coverage_study(
            cohort["scores"], cohort["onset"], cohort["lengths"],
            np.linspace(0.05, 0.95, 20), budgets=(1.0,),
            procedures=("blocked",), n_splits=1,
            fractions=SplitFractions(0.4, 0.2, 0.4), selection="default")
    except ValueError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("expected ValueError")
