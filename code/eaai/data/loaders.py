"""Readers for the two PhysioNet Challenge cohorts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .harmonize import PatientRecord, harmonise, pad_records


def _read_psv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|")


def load_challenge2019(root: str | Path, subset: str = "A",
                       limit: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    """Load one hospital of the 2019 cohort.

    ``subset="A"`` reads ``training_setA`` and ``subset="B"`` reads
    ``training_setB``. The two folders correspond to different institutions and
    are the unit of the cross-site transfer protocol, not interchangeable
    shards of a single sample.
    """
    folder = Path(root) / f"training_set{subset.upper()}"
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} not found; download the 2019 cohort from PhysioNet and "
            "point root at the directory containing training_setA/training_setB"
        )
    files = sorted(folder.glob("*.psv"))
    if limit is not None:
        files = files[:limit]
    return [(p.stem, _read_psv(p)) for p in files]


def load_challenge2012(root: str | Path, subset: str = "a",
                       limit: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    """Load one subset of the 2012 cohort (mortality task)."""
    folder = Path(root) / f"set-{subset.lower()}"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; download the 2012 cohort from PhysioNet")
    files = sorted(folder.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    frames = []
    for p in files:
        frame = pd.read_csv(p, sep=",", header=None, na_values=["-1", "-1.0"])
        frames.append((p.stem, frame))
    return frames


def build_2019_tensors(root: str | Path, subset: str = "A",
                       limit: int | None = None,
                       variables: tuple[str, ...] | None = None) -> dict:
    """Load, harmonise and pad a 2019 subset into model-ready arrays."""
    from .schema import CHALLENGE_2019_DYNAMIC

    variables = variables or CHALLENGE_2019_DYNAMIC
    records: list[PatientRecord] = []
    ids: list[str] = []
    for pid, frame in load_challenge2019(root, subset=subset, limit=limit):
        records.append(harmonise(frame, variables=variables))
        ids.append(pid)
    batch = pad_records(records)
    batch["ids"] = ids
    batch["subset"] = subset
    return batch
