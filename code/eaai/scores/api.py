"""Score-model plumbing and the on-disk score cache.

Policy experiments never retrain anything. Once a score function has been
fitted, its per-timestep scores are written to ``cache/`` keyed by
``(cohort, model, seed)`` and every budget, ablation and validity experiment is
computed from those arrays. This is what keeps the whole programme inside the
stated compute envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import re

import numpy as np


class ScoreModel(Protocol):
    """Anything that turns a padded batch into per-timestep risk scores."""

    name: str

    def fit(self, batch: dict, labels: np.ndarray) -> "ScoreModel": ...

    def predict(self, batch: dict) -> np.ndarray: ...


@dataclass
class ScoreCache:
    """Read and write cached score matrices."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cohort: str, model: str, seed: int) -> Path:
        if (any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", s)
                for s in (cohort, model)) or not isinstance(seed, (int, np.integer))):
            raise ValueError("cache keys require safe names and an integer seed")
        return self.root / f"{cohort}__{model}__s{seed}.npz"

    def exists(self, cohort: str, model: str, seed: int) -> bool:
        if not self._path(cohort, model, seed).exists():
            return False
        try:
            self.read(cohort, model, seed)
        except ValueError:
            return False
        return True

    def write(self, cohort: str, model: str, seed: int,
              scores: np.ndarray, onset: np.ndarray, lengths: np.ndarray) -> Path:
        path = self._path(cohort, model, seed)
        scores, onset, lengths = _cache_arrays(scores, onset, lengths)
        np.savez_compressed(path, scores=scores, onset=onset, lengths=lengths,
                            format_version=np.array(2))
        return path

    def read(self, cohort: str, model: str, seed: int) -> dict:
        path = self._path(cohort, model, seed)
        if not path.exists():
            raise FileNotFoundError(f"no cached scores at {path}")
        with np.load(path, allow_pickle=False) as data:
            if "format_version" not in data or int(data["format_version"]) != 2:
                raise ValueError("obsolete score cache: regenerate with causal version 2 models")
            arrays = _cache_arrays(*(data[k] for k in ("scores", "onset", "lengths")))
            return dict(zip(("scores", "onset", "lengths"), arrays))


def align_lengths(scores, lengths: np.ndarray,
                  horizon: int | None = None) -> np.ndarray:
    """Pad or trim score sequences to a common horizon, filling with NaN.

    Accepts either a dense matrix or a list of per-patient sequences, which is
    what the policy layer produces and what the cache stores.
    """
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not lengths.size or not np.isfinite(lengths).all() or
            np.any(lengths <= 0) or np.any(lengths != np.floor(lengths))):
        raise ValueError("lengths must be nonempty positive integers")
    lengths = lengths.astype(int)
    if horizon is not None and (not isinstance(horizon, (int, np.integer)) or horizon <= 0):
        raise ValueError("horizon must be a positive integer")
    if isinstance(scores, (list, tuple)):
        if len(scores) != lengths.size:
            raise ValueError("one length per patient is required")
        width = int(horizon or (lengths.max() if lengths.size else 0))
        out = np.full((len(scores), width), np.nan)
        for i, seq in enumerate(scores):
            seq = np.asarray(seq, dtype=float).ravel()
            take = min(int(lengths[i]), width, seq.size)
            out[i, :take] = seq[:take]
        return out

    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[0] != lengths.size:
        raise ValueError("scores must have one row per patient")
    width = horizon or scores.shape[1]
    out = np.full((scores.shape[0], width), np.nan)
    take = min(width, scores.shape[1])
    out[:, :take] = scores[:, :take]
    for i, length in enumerate(lengths):
        out[i, int(length):] = np.nan
    return out


def _cache_arrays(scores, onset, lengths):
    scores = np.asarray(scores, dtype=float)
    onset, lengths = np.asarray(onset), np.asarray(lengths)
    if scores.ndim != 2 or onset.shape != (scores.shape[0],):
        raise ValueError("cache requires a dense score matrix and one onset per patient")
    from .labels import valid_steps
    valid = valid_steps(lengths, scores.shape[1])
    if (valid.shape != scores.shape or not np.isfinite(onset).all() or
            np.any(onset < -1) or np.any(onset != np.floor(onset)) or
            not np.isfinite(scores[valid]).all()):
        raise ValueError("invalid cache shapes/onsets or nonfinite valid scores")
    return np.where(valid, scores, np.nan), onset.astype(int), lengths.astype(int)
