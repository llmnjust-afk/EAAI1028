"""Patient-disjoint partitions and an optional outcome-dependent exclusion.

Patient separation prevents within-patient leakage. ``purge`` is NOT necessary
for this property and changes the population; use only as a reported sensitivity
analysis, not as a routine preprocessing or a claim to eliminate censoring bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation,
                "calibration": self.calibration, "test": self.test}


def four_way(n: int, ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
             seed: int = 0) -> Split:
    """Partition patient indices into train/validation/calibration/test."""
    if (not isinstance(n, (int, np.integer)) or n <= 0 or len(ratios) != 4 or
            not np.isfinite(ratios).all() or np.any(np.asarray(ratios) < 0) or
            not np.isclose(sum(ratios), 1.0)):
        raise ValueError("positive n and four nonnegative ratios summing to one required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cuts = np.floor(np.cumsum(ratios[:-1]) * n).astype(int)
    return Split(*np.split(order, cuts))


def purge(index: np.ndarray, onset: np.ndarray, lengths: np.ndarray,
          gap_hours: int = 12) -> np.ndarray:
    """Drop patients whose event falls inside the final ``gap_hours`` of follow-up.

    This optional outcome-dependent exclusion changes the estimand and may
    introduce selection bias even when applied identically to all partitions.
    It does not establish absence of censoring bias or prevent leakage.

    ``onset`` holds -1 for patients without an event.
    """
    index = np.asarray(index)
    keep = (onset[index] < 0) | (onset[index] <= lengths[index] - gap_hours)
    return index[keep]


def stratified_by_site(site: np.ndarray, ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
                       seed: int = 0) -> Split:
    """Four-way split that preserves site proportions in every partition."""
    site = np.asarray(site)
    parts: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "calibration": [], "test": []}
    if site.ndim != 1 or not site.size:
        raise ValueError("site labels must be a nonempty one-dimensional array")
    for position, value in enumerate(np.unique(site)):
        idx = np.flatnonzero(site == value)
        local = four_way(idx.size, ratios=ratios, seed=seed + position)
        for name, local_idx in local.as_dict().items():
            parts[name].append(idx[local_idx])
    return Split(*(np.sort(np.concatenate(parts[k])) for k in
                   ("train", "validation", "calibration", "test")))
