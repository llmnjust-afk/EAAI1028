import numpy as np
import pytest

from eaai.experiments.ward import simulate_alarm_queue, simulate_ward


def _traces(rate_per_hour=0.2, n=10, length=72):
    rng = np.random.default_rng(0)
    fired = [rng.random(length) < rate_per_hour for _ in range(n)]
    lengths = np.full(n, length, dtype=int)
    return fired, lengths


def test_ward_metrics_are_bounded():
    fired, lengths = _traces()
    result = simulate_ward(fired, lengths, arrival_rate=3.0, horizon_days=14,
                           n_beds=10, capacity_per_hour=2.0, warmup_days=2,
                           seed=1)
    assert 0.0 <= result["fraction_hours_over_capacity"] <= 1.0
    assert 0.0 <= result["bed_utilization"] <= 1.0
    assert result["peak_hourly_alarms"] >= result["p95_hourly_alarms"] >= 0
    assert result["admitted"] >= 0 and result["refused"] >= 0


def test_higher_alarm_rate_increases_ward_demand():
    low, lengths = _traces(0.05)
    high, _ = _traces(0.5)
    low_result = simulate_ward(low, lengths, arrival_rate=4.0, horizon_days=14,
                               n_beds=12, capacity_per_hour=3.0, warmup_days=2,
                               seed=2)
    high_result = simulate_ward(high, lengths, arrival_rate=4.0, horizon_days=14,
                                n_beds=12, capacity_per_hour=3.0, warmup_days=2,
                                seed=2)
    assert high_result["alarm_rate_per_bed_day"] > low_result["alarm_rate_per_bed_day"]


def test_ward_rejects_empty_traces():
    with pytest.raises(ValueError):
        simulate_ward([], np.array([], dtype=int))


def test_alarm_queue_reports_response_fractions():
    fired, lengths = _traces(rate_per_hour=0.3)
    result = simulate_alarm_queue(
        fired, lengths, arrival_rate=5.0, horizon_days=14,
        n_beds=20, n_nurses=2, mean_service_min=5.0, warmup_days=2, seed=3)
    assert result["n_alarms"] > 0
    assert 0.0 <= result["nurse_utilization"] <= 1.0
    assert 0.0 <= result["fraction_wait_over_5min"] <= 1.0
    assert result["p95_response_min"] >= result["mean_wait_min"] >= 0.0


def test_alarm_queue_without_alarms_is_empty():
    fired = [np.zeros(48, dtype=bool)]
    result = simulate_alarm_queue(
        fired, np.array([48]), arrival_rate=1.0, horizon_days=4,
        n_beds=4, n_nurses=1, mean_service_min=5.0, warmup_days=1, seed=4)
    assert result["n_alarms"] == 0
    assert result["mean_wait_min"] == 0.0
    assert result["fraction_wait_over_10min"] == 0.0
