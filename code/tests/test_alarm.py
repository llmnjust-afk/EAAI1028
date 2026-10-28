import numpy as np

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


def test_max_alarms_per_day_is_tighter_than_sampling_rate():
    assert max_alarms_per_day(0) == 24.0
    assert max_alarms_per_day(4) == 5.0
    assert max_alarms_per_day(11) == 2.0


def test_invalid_configuration_rejected():
    try:
        AlarmPolicy(threshold=0.5, refractory=-1)
    except ValueError:
        return
    raise AssertionError("negative refractory should raise")
