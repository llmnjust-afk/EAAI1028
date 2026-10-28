"""Clinical instruments computed from the Challenge 2019 variable set.

Each scorer returns a per-timestep integer score. Components whose source
variables are absent from the cohort are omitted and recorded in
``MISSING_COMPONENTS``; the affected scores are therefore partial and are
labelled as such wherever they are reported. Omitting a component is
conservative for threshold-based alerting because it can only lower the score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_COMPONENTS: dict[str, tuple[str, ...]] = {
    "qsofa": ("glasgow coma scale",),
    "mews": ("avpu consciousness",),
    "news": ("avpu consciousness",),
    "sirs": (),
    "sofa": ("pao2/fio2 ratio", "vasopressor dose", "glasgow coma scale", "urine output"),
}


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _points(condition: pd.Series, value: float) -> pd.Series:
    return condition.fillna(False).astype(float) * value


def _sum(parts: list[pd.Series], sources: list[pd.Series]) -> np.ndarray:
    total = np.zeros(len(parts[0]), dtype=float)
    for part in parts:
        total = total + part.to_numpy()
    observed = np.zeros(len(parts[0]), dtype=bool)
    for source in sources:
        observed |= source.notna().to_numpy()
    total[~observed] = np.nan
    return total


def qsofa(frame: pd.DataFrame) -> np.ndarray:
    """Quick SOFA, two of three components (GCS unavailable)."""
    resp = _series(frame, "Resp")
    sbp = _series(frame, "SBP")
    parts = [_points(resp >= 22, 1), _points(sbp <= 100, 1)]
    return _sum(parts, [resp, sbp])


def mews(frame: pd.DataFrame) -> np.ndarray:
    """Modified Early Warning Score, four of five components."""
    sbp = _series(frame, "SBP")
    hr = _series(frame, "HR")
    resp = _series(frame, "Resp")
    temp = _series(frame, "Temp")

    sbp_pts = (
        _points(sbp <= 70, 3) + _points((sbp > 70) & (sbp <= 80), 2)
        + _points((sbp > 80) & (sbp <= 100), 1)
        + _points((sbp > 100) & (sbp <= 199), 0)
        + _points(sbp >= 200, 2)
    )
    hr_pts = (
        _points(hr <= 40, 2) + _points((hr > 40) & (hr <= 50), 1)
        + _points((hr > 50) & (hr <= 100), 0)
        + _points((hr > 100) & (hr <= 110), 1)
        + _points((hr > 110) & (hr <= 129), 2)
        + _points(hr >= 130, 3)
    )
    resp_pts = (
        _points(resp < 9, 2) + _points((resp >= 9) & (resp <= 14), 0)
        + _points((resp >= 15) & (resp <= 20), 1)
        + _points((resp >= 21) & (resp <= 29), 2)
        + _points(resp >= 30, 3)
    )
    temp_pts = _points(temp < 35, 2) + _points((temp >= 35) & (temp <= 38.4), 0) + _points(temp >= 38.5, 2)
    return _sum([sbp_pts, hr_pts, resp_pts, temp_pts], [sbp, hr, resp, temp])


def news(frame: pd.DataFrame) -> np.ndarray:
    """National Early Warning Score, five of six components."""
    resp = _series(frame, "Resp")
    spo2 = _series(frame, "O2Sat")
    temp = _series(frame, "Temp")
    sbp = _series(frame, "SBP")
    hr = _series(frame, "HR")

    resp_pts = (
        _points(resp <= 8, 3) + _points((resp >= 9) & (resp <= 11), 1)
        + _points((resp >= 12) & (resp <= 20), 0)
        + _points((resp >= 21) & (resp <= 24), 2)
        + _points(resp >= 25, 3)
    )
    spo2_pts = (
        _points(spo2 <= 91, 3) + _points((spo2 >= 92) & (spo2 <= 93), 2)
        + _points((spo2 >= 94) & (spo2 <= 95), 1) + _points(spo2 >= 96, 0)
    )
    temp_pts = (
        _points(temp <= 35.0, 3) + _points((temp > 35.0) & (temp <= 36.0), 1)
        + _points((temp > 36.0) & (temp <= 38.0), 0)
        + _points((temp > 38.0) & (temp <= 39.0), 1) + _points(temp > 39.0, 2)
    )
    sbp_pts = (
        _points(sbp <= 90, 3) + _points((sbp > 90) & (sbp <= 100), 2)
        + _points((sbp > 100) & (sbp <= 110), 1)
        + _points((sbp > 110) & (sbp <= 219), 0) + _points(sbp >= 220, 3)
    )
    hr_pts = (
        _points(hr <= 40, 3) + _points((hr > 40) & (hr <= 50), 1)
        + _points((hr > 50) & (hr <= 90), 0) + _points((hr > 90) & (hr <= 110), 1)
        + _points((hr > 110) & (hr <= 130), 2) + _points(hr >= 131, 3)
    )
    return _sum([resp_pts, spo2_pts, temp_pts, sbp_pts, hr_pts],
                [resp, spo2, temp, sbp, hr])


def sirs(frame: pd.DataFrame) -> np.ndarray:
    """Systemic Inflammatory Response Syndrome criteria, four components."""
    temp = _series(frame, "Temp")
    hr = _series(frame, "HR")
    resp = _series(frame, "Resp")
    paco2 = _series(frame, "PaCO2")
    wbc = _series(frame, "WBC")

    temp_pts = _points((temp > 38.0) | (temp < 36.0), 1)
    hr_pts = _points(hr > 90, 1)
    resp_pts = _points((resp > 20) | (paco2 < 32), 1)
    wbc_pts = _points((wbc > 12.0) | (wbc < 4.0), 1)
    return _sum([temp_pts, hr_pts, resp_pts, wbc_pts], [temp, hr, resp, wbc])


def sofa(frame: pd.DataFrame) -> np.ndarray:
    """Partial SOFA: coagulation, liver, cardiovascular and renal components."""
    platelets = _series(frame, "Platelets")
    bilirubin = _series(frame, "Bilirubin_total")
    map_ = _series(frame, "MAP")
    creatinine = _series(frame, "Creatinine")

    plt_pts = (
        _points(platelets < 20, 4) + _points((platelets >= 20) & (platelets < 50), 3)
        + _points((platelets >= 50) & (platelets < 100), 2)
        + _points((platelets >= 100) & (platelets < 150), 1)
    )
    bili_pts = (
        _points(bilirubin >= 12.0, 4) + _points((bilirubin >= 6.0) & (bilirubin < 12.0), 3)
        + _points((bilirubin >= 2.0) & (bilirubin < 6.0), 2)
        + _points((bilirubin >= 1.2) & (bilirubin < 2.0), 1)
    )
    map_pts = _points(map_ < 70, 1)
    creat_pts = (
        _points(creatinine >= 5.0, 4) + _points((creatinine >= 3.5) & (creatinine < 5.0), 3)
        + _points((creatinine >= 2.0) & (creatinine < 3.5), 2)
        + _points((creatinine >= 1.2) & (creatinine < 2.0), 1)
    )
    return _sum([plt_pts, bili_pts, map_pts, creat_pts],
                [platelets, bilirubin, map_, creatinine])


SCORERS = {
    "qsofa": qsofa,
    "mews": mews,
    "news": news,
    "sirs": sirs,
    "sofa": sofa,
}


def forward_fill_vitals(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Carry the most recent observation forward, as clinical staff would."""
    out = frame.copy()
    present = [c for c in columns if c in out.columns]
    out[present] = out[present].ffill()
    return out


def score_cohort(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Compute a named instrument for a patient frame, with forward filling."""
    if name not in SCORERS:
        raise KeyError(f"unknown instrument {name!r}; available: {sorted(SCORERS)}")
    filled = forward_fill_vitals(frame, ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP",
                                         "Resp", "PaCO2", "WBC", "Platelets",
                                         "Bilirubin_total", "Creatinine"])
    return SCORERS[name](filled)
