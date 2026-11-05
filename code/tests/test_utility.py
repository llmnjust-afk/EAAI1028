import numpy as np
import pytest

from eaai.eval.utility import net_benefit, normalized_utility, prediction_utility


def test_inaction_scores_zero_normalized_utility():
    labels = np.array([[0, 0, 0, 1, 1]])
    predictions = np.zeros_like(labels)
    assert prediction_utility(labels, predictions) < 0.0
    assert normalized_utility(labels, predictions) == 0.0


def test_oracle_scores_one():
    labels = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]])
    predictions = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    assert normalized_utility(labels, predictions) == pytest.approx(1.0)


def test_false_alarm_is_penalised():
    labels = np.zeros((1, 5))
    predictions = np.array([[0, 1, 0, 0, 0]])
    assert prediction_utility(labels, predictions) == pytest.approx(-0.05)


def test_net_benefit_matches_manual_value():
    value = net_benefit(0.5, 0.9, 0.1, 0.2)
    assert value == pytest.approx(0.5 * 0.1 - 0.1 * 0.9 * 0.25)
