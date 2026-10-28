import numpy as np
import pytest

from eaai.policy.alarm import (
    AlarmPolicy,
    alarms_per_day,
    max_alarms_per_day,
    run_policy,
    run_policy_batch,
)


def test_memoryless_matches_threshold():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    fired = run_policy(scores, AlarmPolicy(threshold=0.5, refractory=0))
    assert fired.tolist() == [False, True, True, False, True]


def test_refractory_suppresses_subsequent_alarms():
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    fired = run_policy(scores, AlarmPolicy(threshold=0.5, refractory=2))
    assert fired.tolist() == [True, False, False, True, False, False]


def test_escalation_overrides_refractory():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    fired = run_policy(scores, AlarmPolicy(threshold=0.5, refractory=3, escalation=2))
    assert fired.tolist() == [True, False, True, False]


def test_refractory_resets_after_quiet_period():
    scores = np.array([0.9, 0.1, 0.9])
    fired = run_policy(scores, AlarmPolicy(threshold=0.5, refractory=1))
    assert fired.tolist() == [True, False, True]


def test_batch_ignores_padding():
    batch = [np.array([0.9, 0.9, 0.9]), np.array([0.1, 0.9])]
    fired = run_policy_batch(batch, AlarmPolicy(threshold=0.5, refractory=1))
    assert fired[0].tolist() == [True, False, True]
    assert fired[1, :2].tolist() == [False, True]


def test_alarms_per_day_scaling():
    fired = np.array([True, False, False, False, True, False, False, False,
                      False, False, False, False])
    assert alarms_per_day(fired, length_hours=12) == 4.0


def test_max_alarms_per_day_bounds_every_admissible_stay():
    assert max_alarms_per_day(0) == 24.0
    assert max_alarms_per_day(11) == pytest.approx(3 * 24 / 25)

    l_max = max_alarms_per_day(4, min_stay_hours=24)
    assert l_max == pytest.approx(6 * 24 / 26)
    assert l_max > 5.0, "the bound must cover the 26-hour stay, not just the 24-hour one"

    for stay in range(24, 240):
        policy = AlarmPolicy(0.0, refractory=4)
        fired = run_policy(np.zeros(stay), policy)
        assert np.count_nonzero(fired) * 24.0 / stay <= l_max + 1e-12


def test_short_stays_raise_the_bound():
    assert max_alarms_per_day(4, min_stay_hours=6) == pytest.approx(8.0)
    assert max_alarms_per_day(4, min_stay_hours=6) > max_alarms_per_day(4, min_stay_hours=24)


def test_invalid_configuration_rejected():
    try:
        AlarmPolicy(threshold=0.5, refractory=-1)
    except ValueError:
        return
    raise AssertionError("negative refractory should raise")
