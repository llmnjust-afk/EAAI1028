"""Patient-level partitioning with an onset purge gap.

Splitting by timestep leaks information: adjacent hours of the same patient are
near-duplicates, and the label turns positive six hours before clinical onset.
Both effects inflate AUROC, and the inflation does not appear in cross-site
experiments, which makes it easy to mistake for domain shift. Every partition
here is therefore made on patient identifiers with a purge window around onset.
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
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cuts = np.cumsum([int(round(r * n)) for r in ratios[:-1]])
    return Split(*np.split(order, cuts))


def purge(index: np.ndarray, onset: np.ndarray, lengths: np.ndarray,
          gap_hours: int = 12) -> np.ndarray:
    """Drop patients whose onset falls inside the purge gap of the partition.

    ``onset`` holds -1 for patients without an event; ``lengths`` is the record
    length in hours. A patient is kept when the event either does not occur or
    occurs at least ``gap_hours`` before the end of the record, so that the
    pre-onset window is fully observed.
    """
    index = np.asarray(index)
    keep = (onset[index] < 0) | (onset[index] <= lengths[index] - gap_hours)
    return index[keep]


def stratified_by_site(site: np.ndarray, ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
                       seed: int = 0) -> Split:
    """Four-way split that preserves site proportions in every partition."""
    site = np.asarray(site)
    parts: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "calibration": [], "test": []}
    for value in np.unique(site):
        idx = np.flatnonzero(site == value)
        local = four_way(idx.size, ratios=ratios, seed=seed + int(value))
        for name, local_idx in local.as_dict().items():
            parts[name].append(idx[local_idx])
    return Split(*(np.sort(np.concatenate(parts[k])) for k in
                   ("train", "validation", "calibration", "test")))
