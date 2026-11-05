import numpy as np
import pytest

from eaai.eval.utility import (net_benefit, normalized_utility,
                               normalized_utility_rows, prediction_utility,
                               prediction_utility_rows)
from eaai.experiments.e2_budget import patient_utility_matrix
from eaai.policy.alarm import AlarmPolicy, run_policy


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


def test_row_utility_sums_to_total_and_normalised_rows_sum_to_score():
    labels = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]])
    predictions = np.array([[0, 0, 1, 1, 0], [0, 1, 0, 0, 0]])
    rows = prediction_utility_rows(labels, predictions)
    assert rows.sum() == pytest.approx(prediction_utility(labels, predictions))
    norm_rows = normalized_utility_rows(labels, predictions)
    assert norm_rows.sum() == pytest.approx(normalized_utility(labels, predictions))


def test_row_utility_rejects_bad_shapes():
    with pytest.raises(ValueError, match="equal 2-D"):
        prediction_utility_rows(np.zeros((1, 3)), np.zeros((2, 3)))


def test_patient_utility_matrix_reproduces_normalized_utility():
    scores = [np.full(10, 0.9), np.full(10, 0.1)]
    onset = np.array([8, -1])
    lengths = np.array([10, 10])
    thresholds = np.array([0.5, 0.95])
    rows = patient_utility_matrix(scores, onset, lengths, thresholds)
    assert rows.shape == (2, 2)
    labels = np.zeros((2, 10))
    labels[0, 2:10] = 1.0
    predictions = np.zeros((2, 10))
    predictions[0, :] = run_policy(scores[0], AlarmPolicy(0.5, 4, 0))
    expected = normalized_utility(labels, predictions)
    assert rows[:, 0].sum() == pytest.approx(expected)
