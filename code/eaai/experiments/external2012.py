"""External-cohort stress test on the independent PhysioNet 2012 ICU data.

The 2012 cohort has no sepsis onset.  It is used here only to test whether the
budget-versus-threshold transport finding survives a change of cohort, outcome
and hospital system: the event is in-hospital mortality, and an alarm counts as
a detection when it fires in the last 12 observed hours before the end of the
48-hour record.  This is an outcome-shift stress test, not a sepsis replication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.loaders import build_2012_tensors
from ..data.splits import subject_four_way
from ..scores.baselines import PrefixFeatureModel
from .e2_budget import sweep

SUBSETS = ("a", "b", "c")


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _subset(root: str | Path, subset: str) -> dict:
    batch = build_2012_tensors(root, subset=subset)
    mortality = np.asarray(batch["mortality"], dtype=float)
    lengths = np.asarray(batch["lengths"], dtype=int)
    # All decision points are labelled with the patient outcome; the policy is
    # a terminal-risk alarm, not an onset predictor.
    labels = np.repeat(mortality[:, None], batch["values"].shape[1], axis=1)
    onset = np.where(mortality > 0.5, lengths - 1, -1).astype(int)
    batch["labels"] = labels
    batch["onset"] = onset
    return batch


def _predict(model: PrefixFeatureModel, batch: dict, chunk: int = 1500) -> np.ndarray:
    n = len(batch["lengths"])
    out = []
    for start in range(0, n, chunk):
        part = {k: np.asarray(v)[start:start + chunk]
                for k, v in batch.items()
                if k in ("values", "mask", "delta", "valid", "static", "lengths")}
        out.append(model.predict(part))
    return np.concatenate(out, axis=0)


def run(root: str | Path, seed: int = 0,
        budgets=(0.25, 0.5, 1.0, 2.0),
        thresholds: int = 200) -> dict:
    root = Path(root)
    source = _subset(root, "a")
    split = subject_four_way(source["ids"], seed=seed)
    model = PrefixFeatureModel(name="prefix_lr", backend="logistic", random_state=seed)
    model.fit({k: source[k][split.train] for k in
               ("values", "mask", "delta", "valid", "static", "lengths")},
              source["labels"][split.train])
    scores_a = _predict(model, source)
    grid = np.linspace(0.01, 0.99, thresholds)

    rows = []
    internal = sweep(scores_a, source["onset"], source["lengths"],
                     split.calibration, split.test, budgets=tuple(budgets),
                     thresholds=grid, procedure="blocked", min_stay_hours=24.0)
    rows.extend({"cohort": "2012-a (internal)", **row} for row in internal)

    for name in ("b", "c"):
        target = _subset(root, name)
        width = max(scores_a.shape[1], target["values"].shape[1])
        def pad(x):
            return np.pad(x, ((0, 0), (0, width - x.shape[1])), constant_values=np.nan)
        combined = np.concatenate([pad(scores_a), pad(_predict(model, target))], axis=0)
        onset = np.concatenate([source["onset"], target["onset"]])
        lengths = np.concatenate([source["lengths"], target["lengths"]])
        external_index = np.arange(len(source["lengths"]), len(lengths))
        external = sweep(combined, onset, lengths, split.calibration, external_index,
                         budgets=tuple(budgets), thresholds=grid,
                         procedure="blocked", min_stay_hours=24.0)
        rows.extend({"cohort": f"2012-{name} (external)", **row} for row in external)

    cohorts = {"a": {"patients": int(len(source["ids"])),
                     "deaths": int(source["labels"][:, 0].sum())}}
    for name in ("b", "c"):
        target = _subset(root, name)
        cohorts[name] = {"patients": int(len(target["ids"])),
                         "deaths": int(target["labels"][:, 0].sum())}
    return _clean({
        "config": {"seed": seed, "budgets": list(budgets),
                   "thresholds": thresholds, "min_stay_hours": 24.0,
                   "event": "in-hospital mortality", "detection_window_hours": 12.0},
        "cohorts": cohorts,
        "rows": rows,
    })
