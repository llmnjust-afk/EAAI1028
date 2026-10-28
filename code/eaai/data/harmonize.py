"""Convert a per-patient record into a padded tensor with observation metadata.

The representation matches what the irregularity-aware backbones expect:
values, an observation mask, and the elapsed time since the last observation
of each variable. ``ICULOS`` is used as the clock, so a record that skips
hours produces a gap rather than a missing row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import CHALLENGE_2019_DYNAMIC, CHALLENGE_2019_LABEL


@dataclass
class PatientRecord:
    """One patient's harmonised trajectory."""

    values: np.ndarray
    mask: np.ndarray
    delta: np.ndarray
    length: int
    onset_index: int | None
    static: np.ndarray
    variables: tuple[str, ...] = ()
    onset_left_censored: bool = False

    def loss_mask(self) -> np.ndarray:
        return np.ones(self.length, dtype=bool)


def _clock(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        raise ValueError("cannot harmonise an empty patient record")
    if frame.columns.duplicated().any():
        raise ValueError("duplicate variable names are ambiguous")
    if "ICULOS" in frame.columns:
        clock = pd.to_numeric(frame["ICULOS"], errors="coerce").to_numpy()
        if not np.isfinite(clock).all() or np.any(clock != np.floor(clock)):
            raise ValueError("ICULOS must be finite integer hours")
        if np.any(np.diff(clock) <= 0):
            raise ValueError("ICULOS must be strictly increasing without duplicates")
        clock = clock - clock[0]
    else:
        clock = np.arange(len(frame), dtype=float)
    return clock.astype(int)


def harmonise(frame: pd.DataFrame,
              variables: tuple[str, ...] = CHALLENGE_2019_DYNAMIC,
              label_column: str = CHALLENGE_2019_LABEL,
              label_lead_hours: int = 0,
              static_names: tuple[str, ...] = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")) -> PatientRecord:
    """Reindex a patient frame onto its hourly clock and build mask/delta tensors.

    ``onset_index`` is the first step at which ``label_column`` is positive. The
    2019 challenge sets that label six hours before clinical onset, so pass
    ``label_lead_hours=6`` to recover the clinical event time; the default keeps
    the label convention used by most published baselines so that lead times
    remain comparable across papers. A shifted onset outside observed follow-up
    is not a lead-time case; this does not imply that discharge equals onset.
    If the first row is already positive, onset is left-censored and adding six
    hours does not reconstruct a unique clinical time. Such records retain
    onset_index=0 and onset_left_censored=True: no identifiable pre-onset period
    is used for event-label training or lead-time evaluation.
    """
    if not isinstance(label_lead_hours, (int, np.integer)) or label_lead_hours < 0:
        raise ValueError("label_lead_hours must be a nonnegative integer")
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be nonempty and unique")
    clock = _clock(frame)
    horizon = int(clock.max()) + 1
    grid = np.full(horizon, -1, dtype=int)
    grid[clock] = np.arange(len(frame))

    raw = np.full((horizon, len(variables)), np.nan, dtype=float)
    for j, name in enumerate(variables):
        if name in frame.columns:
            column = pd.to_numeric(frame[name], errors="coerce").to_numpy()
            placed = np.where(grid >= 0, column[np.clip(grid, 0, None)], np.nan)
            raw[:, j] = placed

    raw[~np.isfinite(raw)] = np.nan
    mask = np.isfinite(raw)

    delta = np.zeros_like(raw)
    # Elapsed time just before incorporating the current observation. Before
    # the first observation, use time since the record origin. Missing hours
    # must retain elapsed time, rather than being reset to zero.
    last_seen = np.zeros(raw.shape[1])
    for t in range(horizon):
        seen = mask[t]
        delta[t] = t - last_seen
        last_seen[seen] = t

    onset = None
    left_censored = False
    if label_column in frame.columns:
        labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("supplied event labels must be complete binary values")
        hits = np.flatnonzero(np.nan_to_num(labels, nan=0.0) >= 1)
        if hits.size:
            left_censored = bool(hits[0] == 0)
            onset = 0 if left_censored else int(clock[hits[0]]) + int(label_lead_hours)

    static = np.array([float(pd.to_numeric(frame[c], errors="coerce").iloc[0])
                       if c in frame.columns else np.nan for c in static_names])
    static[~np.isfinite(static)] = np.nan

    return PatientRecord(values=raw, mask=mask, delta=delta, length=horizon,
                          onset_index=onset, static=static,
                          variables=tuple(variables), onset_left_censored=left_censored)


def pad_records(records: list[PatientRecord]) -> dict:
    """Stack records into padded arrays plus a validity mask."""
    n = len(records)
    if n == 0:
        raise ValueError("records must contain at least one patient")
    first = records[0]
    for r in records:
        if (r.length <= 0 or r.values.shape != (r.length, first.values.shape[1])
                or r.mask.shape != r.values.shape or r.delta.shape != r.values.shape
                or r.static.shape != first.static.shape or r.variables != first.variables):
            raise ValueError("records must have consistent shapes and variable order")
    horizon = max(r.length for r in records)
    d = records[0].values.shape[1]
    s = records[0].static.shape[0]

    values = np.zeros((n, horizon, d), dtype=float)
    mask = np.zeros((n, horizon, d), dtype=bool)
    delta = np.zeros((n, horizon, d), dtype=float)
    valid = np.zeros((n, horizon), dtype=bool)
    static = np.zeros((n, s), dtype=float)
    lengths = np.zeros(n, dtype=int)
    onset = np.full(n, -1, dtype=int)

    for i, r in enumerate(records):
        values[i, : r.length] = r.values
        mask[i, : r.length] = r.mask
        delta[i, : r.length] = r.delta
        valid[i, : r.length] = True
        static[i] = r.static
        lengths[i] = r.length
        onset[i] = -1 if r.onset_index is None else r.onset_index

    variables = records[0].variables if records else ()
    return {"values": values, "mask": mask, "delta": delta, "valid": valid,
             "static": static, "lengths": lengths, "onset": onset,
             "variables": list(variables),
             "onset_left_censored": np.array([r.onset_left_censored for r in records])}


def summarise(record: PatientRecord) -> dict:
    """Per-variable observed fraction, useful for cohort-level QC."""
    observed = record.mask.sum(axis=0) / max(record.length, 1)
    return {"length": record.length, "onset_index": record.onset_index,
            "observed_fraction": observed}
