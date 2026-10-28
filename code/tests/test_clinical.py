import numpy as np
import pandas as pd

from eaai.scores.clinical import MISSING_COMPONENTS, SCORERS, score_cohort


def frame(**values):
    return pd.DataFrame({k: [v] for k, v in values.items()})


def test_qsofa_counts_two_components():
    out = score_cohort(frame(Resp=24, SBP=90), "qsofa")
    assert out[0] == 2.0


def test_qsofa_nan_when_no_component_observed():
    out = score_cohort(frame(HR=80), "qsofa")
    assert np.isnan(out[0])


def test_news_expected_total():
    out = score_cohort(frame(Resp=25, O2Sat=90, Temp=37.0, SBP=90, HR=130), "news")
    assert out[0] == 11.0


def test_mews_expected_total():
    out = score_cohort(frame(Resp=25, Temp=37.0, SBP=90, HR=130), "mews")
    assert out[0] == 6.0


def test_sirs_expected_total():
    out = score_cohort(frame(Temp=39.0, HR=130, Resp=25, WBC=13.0), "sirs")
    assert out[0] == 4.0


def test_partial_sofa_expected_total():
    out = score_cohort(frame(Platelets=15, Bilirubin_total=13.0, MAP=65, Creatinine=5.5), "sofa")
    assert out[0] == 13.0


def test_forward_fill_carries_last_observation():
    data = pd.DataFrame({"Resp": [24.0, np.nan], "SBP": [90.0, np.nan]})
    out = SCORERS["qsofa"](data.ffill())
    assert out.tolist() == [2.0, 2.0]


def test_missing_components_are_documented():
    assert "glasgow coma scale" in MISSING_COMPONENTS["sofa"]
    assert MISSING_COMPONENTS["sirs"] == ()
