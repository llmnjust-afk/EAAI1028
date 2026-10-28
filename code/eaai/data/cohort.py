"""Cohort-level adapters from official readers to experiment-ready arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .harmonize import pad_records
from .loaders import build_2019_tensors
from ..scores.labels import event_labels, valid_steps

BATCH_KEYS = ("values", "mask", "delta", "valid", "static", "lengths", "onset")


def subset_batch(batch: dict, index: np.ndarray) -> dict:
    """Return a patient subset of a padded batch without copying metadata."""
    index = np.asarray(index)
    if (index.ndim != 1 or not index.size or not np.isfinite(index).all() or
            np.any(index != np.floor(index))):
        raise ValueError("subset indices must be a nonempty integer vector")
    index = index.astype(int)
    n = len(batch["lengths"])
    if np.any(index < 0) or np.any(index >= n):
        raise ValueError("subset index out of range")
    out = {key: np.asarray(batch[key])[index] for key in BATCH_KEYS if key in batch}
    if "onset_left_censored" in batch:
        out["onset_left_censored"] = np.asarray(batch["onset_left_censored"])[index]
    if "ids" in batch:
        out["ids"] = [batch["ids"][i] for i in index]
    for key in ("subset", "label_lead_hours", "variables"):
        if key in batch:
            out[key] = batch[key]
    return out


def prepare_2019(root, subset: str = "A", limit: int | None = None,
                 label_lead_hours: int = 6, horizon: int = 6) -> tuple[dict, np.ndarray]:
    """Load a 2019 subset and construct prospective event labels.

    ``label_lead_hours=6`` converts the shipped label boundary back to the
    clinical sepsis time; ``horizon=6`` then labels exactly the six hours before
    that time. Records with a left-censored onset receive no training target,
    because their pre-onset window is unidentifiable.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    batch = build_2019_tensors(root, subset=subset, limit=limit,
                               label_lead_hours=label_lead_hours)
    labels = event_labels(batch["onset"], batch["lengths"], horizon)
    observed = valid_steps(batch["lengths"], labels.shape[1]) & np.isfinite(labels)
    if not observed.any():
        raise ValueError("cohort has no usable target timesteps")
    if not np.isin(labels[observed], [0, 1]).all():
        raise ValueError("constructed labels must be binary")
    if not np.any(labels[observed] == 1):
        raise ValueError("cohort has no positive target timesteps")
    batch["horizon"] = int(horizon)
    batch["targets"] = labels
    return batch, labels


def split_summary(batch: dict, labels: np.ndarray, index: np.ndarray) -> dict:
    """Counts and target prevalence for a patient partition."""
    index = np.asarray(index, dtype=int)
    selected = labels[index]
    usable = valid_steps(batch["lengths"][index], labels.shape[1]) & np.isfinite(selected)
    target = selected[usable]
    positive_patients = int(sum(
        np.any(selected[row][usable[row]] == 1) for row in range(len(index))))
    return {
        "patients": int(index.size),
        "timesteps": int(usable.sum()),
        "positive_timesteps": int(target.sum()),
        "positive_patients": positive_patients,
        "left_censored": int(np.sum(batch["onset_left_censored"][index])),
        "median_length": float(np.median(batch["lengths"][index])),
    }


def clinical_score_batch_multi(root, subset: str = "A",
                               instruments=("news", "sirs", "qsofa"),
                               limit: int | None = None,
                               label_lead_hours: int = 6) -> dict:
    """Build several padded clinical-instrument caches in one cohort pass.

    Loading and harmonising each PSV repeatedly dominates the cost of the
    single-instrument helper. This function reads each record once, builds the
    hourly frame once, and applies every requested instrument to that frame.
    Steps before the first observed component are backfilled with the first
    finite score; any remaining unobserved step receives zero points.
    """
    from .harmonize import harmonise
    from .loaders import load_challenge2019
    from .schema import CHALLENGE_2019_DYNAMIC
    from ..scores.clinical import score_cohort

    instruments = tuple(dict.fromkeys(instruments))
    if not instruments:
        raise ValueError("at least one clinical instrument is required")
    frames = load_challenge2019(root, subset=subset, limit=limit)
    if not frames:
        raise ValueError("cohort is empty")
    ids: list[str] = []
    lengths: list[int] = []
    onset: list[int] = []
    left_censored: list[bool] = []
    scores = {name: [] for name in instruments}
    for pid, frame in frames:
        if "ICULOS" in frame.columns:
            clock = pd.to_numeric(frame["ICULOS"], errors="coerce").to_numpy()
            if (not np.isfinite(clock).all() or np.any(clock != np.floor(clock))
                    or np.any(np.diff(clock) <= 0)):
                raise ValueError("ICULOS must be finite, integer and strictly increasing")
            clock = (clock - clock[0]).astype(int)
        else:
            clock = np.arange(len(frame), dtype=int)
        horizon = int(clock.max()) + 1
        hourly = pd.DataFrame(index=np.arange(horizon))
        for column in frame.columns:
            placed = np.full(horizon, np.nan, dtype=float)
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            placed[clock] = values
            hourly[column] = placed
        record = harmonise(frame, variables=CHALLENGE_2019_DYNAMIC,
                           label_lead_hours=label_lead_hours)
        for name in instruments:
            scored = np.asarray(score_cohort(hourly, name), dtype=float)
            if scored.shape != (record.length,):
                raise ValueError("clinical score length does not match harmonised record")
            finite = np.flatnonzero(np.isfinite(scored))
            if finite.size:
                scored[:finite[0]] = scored[finite[0]]
            scored[~np.isfinite(scored)] = 0.0
            scores[name].append(scored)
        ids.append(pid)
        lengths.append(record.length)
        onset.append(-1 if record.onset_index is None else record.onset_index)
        left_censored.append(record.onset_left_censored)
    width = max(lengths)
    output = {}
    for name in instruments:
        padded = np.full((len(scores[name]), width), np.nan)
        for i, seq in enumerate(scores[name]):
            padded[i, :seq.size] = seq
        output[name] = {
            "scores": padded,
            "ids": list(ids),
            "onset": np.asarray(onset, dtype=int),
            "lengths": np.asarray(lengths, dtype=int),
            "onset_left_censored": np.asarray(left_censored, dtype=bool),
            "subset": subset,
            "instrument": name,
            "label_lead_hours": int(label_lead_hours),
        }
    return output


def clinical_score_batch(root, subset: str = "A", instrument: str = "news",
                         limit: int | None = None,
                         label_lead_hours: int = 6) -> dict:
    """Build one padded per-timestep clinical-instrument score cache."""
    return clinical_score_batch_multi(
        root, subset=subset, instruments=(instrument,), limit=limit,
        label_lead_hours=label_lead_hours)[instrument]
