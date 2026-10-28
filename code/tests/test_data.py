import numpy as np
import pandas as pd
import pytest

from eaai.data.harmonize import harmonise, pad_records
from eaai.data.splits import four_way, purge
from eaai.data.synthetic import make_scores


def test_clock_gap_becomes_a_missing_hour():
    frame = pd.DataFrame({
        "ICULOS": [1, 2, 5],
        "HR": [80.0, 81.0, 82.0],
        "SepsisLabel": [0, 0, 0],
    })
    record = harmonise(frame, variables=("HR",))
    assert record.length == 5
    assert record.mask[:, 0].tolist() == [True, True, False, False, True]
    assert record.values[2, 0] != record.values[2, 0]


def test_delta_counts_elapsed_steps_since_observation():
    frame = pd.DataFrame({
        "ICULOS": [1, 2, 5],
        "HR": [80.0, 81.0, 82.0],
    })
    record = harmonise(frame, variables=("HR",), label_column="SepsisLabel")
    assert record.delta[1, 0] == 1.0
    assert record.delta[4, 0] == 3.0


def test_onset_is_first_positive_label_on_the_clock():
    frame = pd.DataFrame({
        "ICULOS": [1, 2, 3, 4],
        "HR": [80.0] * 4,
        "SepsisLabel": [0, 1, 1, 1],
    })
    record = harmonise(frame, variables=("HR",))
    assert record.onset_index == 1


def test_padding_preserves_lengths_and_marks_validity():
    frames = [
        pd.DataFrame({"ICULOS": [1, 2], "HR": [80.0, 81.0]}),
        pd.DataFrame({"ICULOS": [1, 2, 3, 4], "HR": [70.0] * 4}),
    ]
    batch = pad_records([harmonise(f, variables=("HR",)) for f in frames])
    assert batch["valid"].shape == (2, 4)
    assert batch["lengths"].tolist() == [2, 4]
    assert batch["valid"][0].tolist() == [True, True, False, False]


def test_four_way_split_is_disjoint_and_complete():
    split = four_way(1000, seed=3)
    parts = split.as_dict()
    stacked = np.concatenate([parts[k] for k in parts])
    assert stacked.size == 1000
    assert np.unique(stacked).size == 1000


def test_purge_drops_events_too_close_to_record_end():
    onset = np.array([-1, 100, 5])
    lengths = np.array([50, 105, 60])
    kept = purge(np.array([0, 1, 2]), onset, lengths, gap_hours=12)
    assert kept.tolist() == [0, 2]


def test_synthetic_cohort_shapes():
    cohort = make_scores(50, seed=0)
    assert len(cohort["scores"]) == 50
    assert cohort["lengths"].min() >= 24
    assert cohort["event"].dtype == bool
    assert np.all((cohort["onset"][~cohort["event"]] == -1))
