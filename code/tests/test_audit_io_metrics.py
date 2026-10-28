import numpy as np
import pandas as pd
import pytest

from eaai.data.challenge2012 import hourly_record
from eaai.data.loaders import build_2012_tensors
from eaai.eval.metrics import (bootstrap_ci, discrimination, expected_calibration_error,
                               net_benefit, paired_wilcoxon, delong_test)
from eaai.scores.api import ScoreCache, align_lengths


def test_2012_outcomes_join_by_id_not_row_order(tmp_path):
    folder = tmp_path / "set-a"
    folder.mkdir()
    for pid in (100, 200):
        (folder / f"{pid}.txt").write_text(
            f"Time,Parameter,Value\n00:00,RecordID,{pid}\n00:01,HR,{pid}\n")
    (tmp_path / "Outcomes-a.txt").write_text(
        "RecordID,In-hospital_death\n200,1\n100,0\n")
    batch = build_2012_tensors(tmp_path)
    assert batch["mortality"].tolist() == [0., 1.]
    assert batch["onset"].tolist() == [-1, -1]
    assert np.isfinite(batch["delta"]).all()
    assert batch["mask"][:, 0].sum() == 0
    (tmp_path / "Outcomes-a.txt").write_text("RecordID,In-hospital_death\n200,1\n")
    with pytest.raises(ValueError, match="missing mortality"):
        build_2012_tensors(tmp_path)


def test_2012_conversion_is_causal_and_aliases_are_chronological():
    raw = pd.DataFrame({"Time": ["00:00", "00:30", "00:50", "01:01"],
        "Parameter": ["RecordID", "SysABP", "NISysABP", "SysABP"],
        "Value": [100, 120, 100, 80]})
    full = hourly_record(raw, "100")
    prefix = hourly_record(raw.iloc[:3], "100")
    assert full.loc[1, "SBP"] == 100 and full.loc[2, "SBP"] == 80
    pd.testing.assert_frame_equal(full.iloc[:2], prefix.iloc[:2])
    with pytest.raises(ValueError, match="RecordID"):
        hourly_record(raw, "999")
    raw.loc[1, "Time"] = "48:01"
    with pytest.raises(ValueError, match="48 hours"):
        hourly_record(raw, "100")


def test_old_cache_is_not_used_after_causal_model_fix(tmp_path):
    cache = ScoreCache(tmp_path)
    np.savez_compressed(tmp_path / "cohort__model__s0.npz", scores=np.ones((2, 3)),
                        onset=np.array([-1, 2]), lengths=np.array([3, 3]))
    assert not cache.exists("cohort", "model", 0)
    with pytest.raises(ValueError, match="obsolete"):
        cache.read("cohort", "model", 0)
    cache.write("cohort", "model", 0, np.ones((2, 3)), [-1, 2], np.array([3, 2]))
    assert cache.exists("cohort", "model", 0)
    assert np.isnan(cache.read("cohort", "model", 0)["scores"][1, 2])


def test_documented_2012_range_outlier_is_missing_and_counted():
    raw = pd.DataFrame({"Time": ["00:00", "00:30", "01:00"],
                       "Parameter": ["RecordID", "Temp", "Temp"],
                       "Value": [100., -17.8, 37.]})
    result = hourly_record(raw, "100")
    assert result.attrs["invalid_measurements"] == 1
    assert result.loc[1, "Temp"] == 37.


def test_cache_rejects_path_traversal_and_invalid_shapes(tmp_path):
    cache = ScoreCache(tmp_path)
    with pytest.raises(ValueError, match="safe names"):
        cache.exists("../cohort", "model", 0)
    with pytest.raises(ValueError):
        cache.write("c", "m", 0, np.ones((2, 3)), [-1], np.array([3, 3]))
    with pytest.raises(ValueError):
        align_lengths([np.ones(3)], np.array([3, 3]))


@pytest.mark.parametrize("function", [discrimination, expected_calibration_error, net_benefit])
def test_metrics_reject_empty_or_nonbinary_targets(function):
    kwargs = {"threshold": .5} if function == net_benefit else {}
    for targets, scores in (([], []), ([0, 2], [.1, .9]), ([0, 1], [.1, np.nan])):
        with pytest.raises(ValueError):
            function(targets, scores, **kwargs)


def test_probabilities_outside_unit_interval_and_small_delong_are_rejected():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [-.1, 1.1])
    with pytest.raises(ValueError, match="two observations"):
        delong_test([0, 1], [.1, .9], [.2, .8])
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_wilcoxon_zero_differences_do_not_dilute_effect_size():
    report = paired_wilcoxon([0, 1, 2, 3], [0, 0, 0, 0])
    assert report["effect_size"] == 1.
    assert paired_wilcoxon([0, 1e-9], [0, 0])["effect_size"] == 1.


def test_saved_diagnostic_comparison_normalises_json_intervals_without_hiding_errors():
    import json
    from scripts.verify_saved_e0 import compare
    result = {"interval": (0., .018846), "rate": .06, "threshold": None}
    compare(result, json.loads(json.dumps(result)))
    with pytest.raises(AssertionError, match="different numbers"):
        compare(result, {"interval": [0., .018846], "rate": .13, "threshold": None})
