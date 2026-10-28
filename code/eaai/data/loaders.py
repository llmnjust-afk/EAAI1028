"""Readers for the two PhysioNet Challenge cohorts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    if subset.upper() not in ("A", "B"):
        raise ValueError("only public 2019 training subsets A and B are supported")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
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
    """Read official long-form records and convert causally to hourly decisions.

    Outcomes are NOT present in patient files. They are joined separately by
    :func:`build_2012_tensors`; no onset is available for lead-time evaluation.
    """
    from .challenge2012 import hourly_record

    if subset.lower() not in ("a", "b", "c"):
        raise ValueError("2012 subset must be a, b or c")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    folder = Path(root) / f"set-{subset.lower()}"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; download the 2012 cohort from PhysioNet")
    files = sorted(folder.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    frames = []
    for p in files:
        frame = pd.read_csv(p, dtype={"Time": str})
        frame.columns = [str(c).strip() for c in frame.columns]
        frames.append((p.stem, hourly_record(frame, p.stem)))
    return frames


def build_2019_tensors(root: str | Path, subset: str = "A",
                       limit: int | None = None,
                       variables: tuple[str, ...] | None = None,
                       label_lead_hours: int = 0) -> dict:
    """Load, harmonise and pad a 2019 subset into model-ready arrays."""
    from .schema import CHALLENGE_2019_DYNAMIC

    variables = CHALLENGE_2019_DYNAMIC if variables is None else variables
    records: list[PatientRecord] = []
    ids: list[str] = []
    for pid, frame in load_challenge2019(root, subset=subset, limit=limit):
        records.append(harmonise(frame, variables=variables, label_lead_hours=label_lead_hours))
        ids.append(pid)
    batch = pad_records(records)
    batch["ids"] = ids
    batch["subset"] = subset
    batch["label_lead_hours"] = label_lead_hours
    return batch


def build_2012_tensors(root: str | Path, subset: str = "a",
                       limit: int | None = None,
                       variables: tuple[str, ...] | None = None) -> dict:
    """Load, harmonise and pad a 2012 subset into model-ready arrays.

    ``SepsisLabel`` does not exist in this cohort, so every ``onset`` entry is
    ``-1``. Outcomes-a/b/c.txt is required and joined by RecordID, never row
    order. Decision times are 0,...,48 h (49 points), not event follow-up.
    """
    from .schema import CHALLENGE_2012_VARIABLES
    from .challenge2012 import STATIC

    if subset.lower() not in ("a", "b", "c"):
        raise ValueError("2012 subset must be a, b or c")
    variables = CHALLENGE_2012_VARIABLES if variables is None else variables
    outcome_path = Path(root) / f"Outcomes-{subset.lower()}.txt"
    outcomes = pd.read_csv(outcome_path)
    if not {"RecordID", "In-hospital_death"}.issubset(outcomes.columns):
        raise ValueError("outcomes require RecordID and In-hospital_death")
    if outcomes["RecordID"].duplicated().any():
        raise ValueError("duplicate RecordID in outcomes")
    outcomes = outcomes.set_index("RecordID")["In-hospital_death"]
    if not outcomes.isin([0, 1]).all():
        raise ValueError("mortality outcomes must be binary and nonmissing")
    records: list[PatientRecord] = []
    ids: list[str] = []
    mortality: list[float] = []
    qc = {}
    for pid, frame in load_challenge2012(root, subset=subset, limit=limit):
        records.append(harmonise(frame, variables=variables,
                                 label_column="__absent__", static_names=STATIC))
        ids.append(pid)
        qc[pid] = frame.attrs.get("invalid_measurements", 0)
        if int(pid) not in outcomes.index:
            raise ValueError(f"missing mortality outcome for RecordID {pid}")
        mortality.append(float(outcomes.loc[int(pid)]))
    batch = pad_records(records)
    batch["ids"] = ids
    batch["mortality"] = np.asarray(mortality)
    batch["decision_hours"] = np.arange(49)
    batch["qc_invalid_measurements"] = qc
    # Names are aligned, source units are retained. Direct cross-cohort reuse
    # needs explicit unit conversion (notably Lactate) and a shared static schema.
    batch["source_units"] = {"Lactate": "mmol/L"}
    batch["subset"] = subset
    return batch
