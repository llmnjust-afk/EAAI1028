"""Turn onset times into per-timestep training targets."""

from __future__ import annotations

import numpy as np


def event_labels(onset: np.ndarray, lengths: np.ndarray, horizon: int,
                 width: int | None = None) -> np.ndarray:
    """Label each timestep positive when onset falls within the next ``horizon`` steps.

    Timesteps at or after onset are left negative: the task is prediction, not
    detection of an event that has already happened.
    """
    onset = np.asarray(onset).astype(int).ravel()
    lengths = np.asarray(lengths).astype(int).ravel()
    width = int(width or lengths.max())
    labels = np.zeros((onset.size, width), dtype=float)

    for i, (t_event, length) in enumerate(zip(onset, lengths)):
        if t_event < 0:
            continue
        start = max(t_event - horizon, 0)
        labels[i, start:t_event] = 1.0
    return labels


def valid_steps(lengths: np.ndarray, width: int | None = None) -> np.ndarray:
    """Boolean mask of real (unpadded) timesteps."""
    lengths = np.asarray(lengths).astype(int).ravel()
    width = int(width or lengths.max())
    grid = np.arange(width)[None, :]
    return grid < lengths[:, None]
