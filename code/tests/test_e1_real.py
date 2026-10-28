"""Tests for the 2019 cohort adapter and real-cohort E1 driver."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eaai.data.cohort import prepare_2019, split_summary, subset_batch
from eaai.experiments.e1_real import run


def _write_cohort(root, subset="A", patients=24, hours=10):
    folder = root / f"training_set{subset}"
    folder.mkdir(parents=True)
    for i in range(patients):
        septic = i % 2 == 0
        labels = np.zeros(hours, dtype=int)
        if septic:
            labels[2:] = 1
        frame = pd.DataFrame({
            "HR": 70 + i % 5 + np.arange(hours) * 0.1,
            "O2Sat": 98 - (i % 3),
            "Temp": 37.0,
            "SBP": 120,
            "MAP": 80,
            "DBP": 60,
            "Resp": 18,
            "Age": 60 + i % 10,
            "Gender": i % 2,
            "Unit1": 1,
            "Unit2": 0,
            "HospAdmTime": -5,
            "ICULOS": np.arange(1, hours + 1),
            "SepsisLabel": labels,
        })
        frame.to_csv(folder / f"p{i:06d}.psv", sep="|", index=False, na_rep="NaN")


def test_2019_adapter_builds_prospective_targets_and_subset(tmp_path):
    _write_cohort(tmp_path)
    batch, labels = prepare_2019(tmp_path, subset="A", horizon=6)
    assert batch["values"].shape[0] == 24
    septic = np.flatnonzero((batch["onset"] > 0))
    assert septic.size == 12
    patient = septic[0]
    assert batch["onset"][patient] == 8
    assert np.all(labels[patient, 2:8] == 1)
    assert np.isnan(labels[patient, 8:]).all()
    assert np.all(labels[patient, :2] == 0)

    subset = subset_batch(batch, np.array([patient, patient + 1]))
    assert subset["values"].shape[0] == 2
    assert subset["ids"] == [batch["ids"][patient], batch["ids"][patient + 1]]
    summary = split_summary(batch, labels, np.array([patient]))
    assert summary["patients"] == 1
    assert summary["positive_patients"] == 1


def test_2019_adapter_rejects_missing_positive_class(tmp_path):
    _write_cohort(tmp_path, patients=4)
    for path in (tmp_path / "training_setA").glob("*.psv"):
        frame = pd.read_csv(path, sep="|")
        frame["SepsisLabel"] = 0
        frame.to_csv(path, sep="|", index=False)
    with pytest.raises(ValueError, match="no positive target timesteps"):
        prepare_2019(tmp_path)


def test_e1_real_driver_trains_caches_and_evaluates(tmp_path):
    data = tmp_path / "data"
    _write_cohort(data, patients=40)
    report = run(data, subset="A", model_name="prefix_lr", seed=0,
                 cache_root=tmp_path / "cache", results_dir=tmp_path / "results",
                 external_subset=None, chunk_size=7)
    assert set(report["partitions"]) == {"validation", "calibration", "test"}
    assert report["partitions"]["test"]["patients"] > 0
    assert np.isfinite(report["partitions"]["test"]["auroc"])
    assert report["cache"]["patients"] == 40
    assert (tmp_path / "cache" / "A_lead6__prefix_lr__s0.npz").exists()
    saved = json.loads((tmp_path / "results" / "e1_A_prefix_lr_s0.json").read_text())
    assert saved["data"]["horizon"] == 6
    assert (tmp_path / "results" / "splits_A_s0.npz").exists()
