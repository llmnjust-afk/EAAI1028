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


def test_labels_before_the_shipped_label_convention():
    frame = pd.DataFrame({
        "ICULOS": [1, 2, 3, 4, 5, 6, 7, 8],
        "HR": [80.0] * 8,
        "SepsisLabel": [0, 0, 1, 1, 1, 1, 1, 1],
    })
    shipped = harmonise(frame, variables=("HR",))
    clinical = harmonise(frame, variables=("HR",), label_lead_hours=6)
    assert shipped.onset_index == 2
    assert clinical.onset_index == 8


def test_record_keeps_the_variable_order_it_was_built_with():
    frame = pd.DataFrame({"ICULOS": [1, 2], "SBP": [90.0, 95.0], "HR": [70.0, 75.0]})
    record = harmonise(frame, variables=("SBP", "HR"))
    batch = pad_records([record])
    assert batch["variables"] == ["SBP", "HR"]
    assert batch["values"].shape == (1, 2, 2)


def test_build_2019_tensors_from_psv_files(tmp_path):
    from eaai.data.loaders import build_2019_tensors

    folder = tmp_path / "training_setA"
    folder.mkdir()
    (folder / "p001.psv").write_text(
        "ICULOS|HR|SepsisLabel\n1|80.0|0\n2|82.0|0\n3|85.0|1\n")
    (folder / "p002.psv").write_text(
        "ICULOS|HR|SepsisLabel\n1|70.0|0\n4|72.0|0\n")

    batch = build_2019_tensors(tmp_path, subset="A", variables=("HR",))
    assert batch["ids"] == ["p001", "p002"]
    assert batch["lengths"].tolist() == [3, 4]
    assert batch["onset"].tolist() == [2, -1]
    assert batch["mask"][1, :, 0].tolist() == [True, False, False, True]


def test_load_challenge2012_parses_the_header(tmp_path):
    from eaai.data.loaders import build_2012_tensors

    folder = tmp_path / "set-a"
    folder.mkdir()
    (folder / "132539.txt").write_text(
        "Time,Parameter,Value\n00:00,RecordID,132539\n00:00,Age,54\n"
        "00:00,Gender,0\n00:00,Height,-1\n00:00,ICUType,4\n"
        "00:07,HR,73\n00:37,HR,80\n00:07,NISysABP,147\n"
        "00:07,RespRate,19\n02:12,HCT,40\n47:37,HR,90\n")
    (tmp_path / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,6,1,5,-1,0\n")
    batch = build_2012_tensors(tmp_path, subset="a")
    assert batch["lengths"].tolist() == [49]
    assert batch["onset"].tolist() == [-1]
    assert batch["mortality"].tolist() == [0.0]
    assert not batch["mask"][0, 0, 0]  # 00:07 must not leak into admission time.
    assert batch["values"][0, 1, 0] == 80  # last available HR in hour one.
    assert batch["values"][0, 1, 1] == 147  # mapped NISysABP -> SBP.
    assert batch["values"][0, 3, 8] == 40  # HCT -> Hct, not available at hour two.
    assert batch["static"].shape == (1, 4)
    assert batch["values"][0, 48, 0] == 90
