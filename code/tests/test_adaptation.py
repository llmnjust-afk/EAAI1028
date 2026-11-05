import numpy as np
import pytest

from eaai.policy.adaptation import rate_matching_threshold


def _cohort(n=8, length=48):
    scores = [np.full(length, 0.9) for _ in range(n)]
    lengths = np.full(n, length, dtype=int)
    return scores, lengths


def test_rate_matching_chooses_a_threshold_below_the_target_rate():
    scores, lengths = _cohort()
    result = rate_matching_threshold(scores, lengths, [0.5, 0.95], target_rate=1.0)
    assert result.within_budget
    assert result.index == 1
    assert result.threshold == 0.95
    assert result.achieved_rate <= 1.0


def test_rate_matching_falls_back_to_the_most_aggressive_threshold():
    scores, lengths = _cohort()
    result = rate_matching_threshold(scores, lengths, [0.5, 0.8], target_rate=0.01)
    assert result.index == 0
    assert not result.within_budget


def test_rate_matching_rejects_nonpositive_targets():
    scores, lengths = _cohort()
    with pytest.raises(ValueError):
        rate_matching_threshold(scores, lengths, [0.5], target_rate=0.0)
