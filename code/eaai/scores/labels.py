"""Turn onset times into per-timestep training targets."""

from __future__ import annotations

import numpy as np


def _lengths(lengths, width):
    values = np.asarray(lengths)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(values != np.floor(values))):
        raise ValueError("lengths must be a nonempty vector of positive integers")
    values = values.astype(int)
    if width is None:
        width = int(values.max())
    if not isinstance(width, (int, np.integer)) or width < values.max():
        raise ValueError("width must be an integer covering every real timestep")
    return values, int(width)


def event_labels(onset: np.ndarray, lengths: np.ndarray, horizon: int,
                 width: int | None = None) -> np.ndarray:
    """Label each timestep positive when onset falls within the next ``horizon`` steps.

    Padding and timesteps at or after onset are NaN (excluded from training and
    evaluation), not negative examples. This is a prospective prediction task.
    """
    lengths, width = _lengths(lengths, width)
    onset = np.asarray(onset)
    if (onset.shape != lengths.shape or not np.isfinite(onset).all()
            or np.any(onset < -1) or np.any(onset != np.floor(onset))):
        raise ValueError("onset must have one integer event time or -1 per patient")
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    onset = onset.astype(int)
    labels = np.zeros((onset.size, width), dtype=float)

    for i, (t_event, length) in enumerate(zip(onset, lengths)):
        labels[i, length:] = np.nan
        if t_event < 0:
            continue
        start = max(t_event - horizon, 0)
        labels[i, start:min(t_event, length)] = 1.0
        labels[i, t_event:length] = np.nan
    return labels


def valid_steps(lengths: np.ndarray, width: int | None = None) -> np.ndarray:
    """Boolean mask of real (unpadded) timesteps."""
    lengths, width = _lengths(lengths, width)
    grid = np.arange(width)[None, :]
    return grid < lengths[:, None]
