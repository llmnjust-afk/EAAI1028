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
        return self.root / f"{cohort}__{model}__s{seed}.npz"

    def exists(self, cohort: str, model: str, seed: int) -> bool:
        return self._path(cohort, model, seed).exists()

    def write(self, cohort: str, model: str, seed: int,
              scores: np.ndarray, onset: np.ndarray, lengths: np.ndarray) -> Path:
        path = self._path(cohort, model, seed)
        np.savez_compressed(path, scores=scores, onset=onset, lengths=lengths)
        return path

    def read(self, cohort: str, model: str, seed: int) -> dict:
        path = self._path(cohort, model, seed)
        if not path.exists():
            raise FileNotFoundError(f"no cached scores at {path}")
        with np.load(path) as data:
            return {k: data[k] for k in ("scores", "onset", "lengths")}


def align_lengths(scores: np.ndarray, lengths: np.ndarray,
                  horizon: int | None = None) -> np.ndarray:
    """Pad or trim a score matrix to a common horizon, filling with NaN."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[None, :]
    width = horizon or scores.shape[1]
    out = np.full((scores.shape[0], width), np.nan)
    take = min(width, scores.shape[1])
    out[:, :take] = scores[:, :take]
    for i, length in enumerate(lengths):
        out[i, int(length):] = np.nan
    return out
