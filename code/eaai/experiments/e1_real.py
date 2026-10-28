"""Train a real-cohort score model, evaluate partitions and cache trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.cohort import prepare_2019, split_summary, subset_batch
from ..data.splits import four_way
from ..scores.api import ScoreCache
from ..scores.baselines import MaskedGRU, PrefixFeatureModel
from .e1_train import evaluate_scores


def make_model(name: str, seed: int, **kwargs):
    if name in ("prefix_lr", "logistic"):
        return PrefixFeatureModel(name="prefix_lr", backend="logistic", random_state=seed)
    if name in ("lightgbm", "lgbm"):
        return PrefixFeatureModel(name="lightgbm", backend="lightgbm", random_state=seed)
    if name in ("gru", "masked_gru"):
        return MaskedGRU(name="masked_gru", random_state=seed, **kwargs)
    raise ValueError(f"unknown model {name!r}")


def predict_chunked(model, batch: dict, chunk_size: int = 2000) -> np.ndarray:
    n = len(batch["lengths"])
    if n == 0:
        raise ValueError("prediction requires at least one patient")
    outputs = []
    for start in range(0, n, chunk_size):
        part = subset_batch(batch, np.arange(start, min(start + chunk_size, n)))
        outputs.append(model.predict(part))
    return np.concatenate(outputs, axis=0)


def _partition_report(model, batch, labels, index, name):
    part = subset_batch(batch, index)
    scores = predict_chunked(model, part)
    report = evaluate_scores(scores, labels[index], part["lengths"])
    report.update(split_summary(batch, labels, index))
    report["partition"] = name
    return report


def run(root: str | Path, subset: str = "A", limit: int | None = None,
        model_name: str = "prefix_lr", seed: int = 0,
        cache_root: str | Path | None = None,
        results_dir: str | Path | None = None,
        external_subset: str | None = "B", external_limit: int | None = None,
        chunk_size: int = 2000, label_lead_hours: int = 6,
        horizon: int = 6, **model_kwargs) -> dict:
    """Fit on the training partition, evaluate held-out patients and cache scores."""
    batch, labels = prepare_2019(root, subset=subset, limit=limit,
                                 label_lead_hours=label_lead_hours, horizon=horizon)
    split = four_way(len(labels), seed=seed)
    train = subset_batch(batch, split.train)
    model = make_model(model_name, seed, **model_kwargs)
    model.fit(train, labels[split.train])

    reports = {name: _partition_report(model, batch, labels, index, name)
               for name, index in (("validation", split.validation),
                                   ("calibration", split.calibration),
                                   ("test", split.test))}

    cache_report = {}
    if cache_root is not None:
        cache = ScoreCache(Path(cache_root))
        scores = predict_chunked(model, batch, chunk_size=chunk_size)
        cohort = f"{subset}_lead{label_lead_hours}"
        path = cache.write(cohort, model.name, seed, scores,
                           batch["onset"], batch["lengths"])
        cache_report = {"cohort": cohort, "path": str(path), "patients": int(len(labels))}

    external = {}
    if external_subset:
        ext_batch, ext_labels = prepare_2019(
            root, subset=external_subset, limit=external_limit,
            label_lead_hours=label_lead_hours, horizon=horizon)
        ext_scores = predict_chunked(model, ext_batch, chunk_size=chunk_size)
        external = {
            "subset": external_subset,
            "report": evaluate_scores(ext_scores, ext_labels, ext_batch["lengths"]),
            "patients": int(len(ext_labels)),
        }
        external["report"].update(split_summary(
            ext_batch, ext_labels, np.arange(len(ext_labels))))
        if cache_root is not None:
            cohort = f"{external_subset}_lead{label_lead_hours}"
            path = ScoreCache(Path(cache_root)).write(
                cohort, model.name, seed, ext_scores,
                ext_batch["onset"], ext_batch["lengths"])
            external["cache"] = {"cohort": cohort, "path": str(path)}

    report = {
        "data": {"root": str(root), "subset": subset, "limit": limit,
                 "label_lead_hours": label_lead_hours, "horizon": horizon},
        "model": {"name": model.name, "seed": seed,
                  "params": {k: v for k, v in vars(model).items()
                             if k not in ("pipeline", "module")}},
        "split": {name: split_summary(batch, labels, index)
                  for name, index in split.as_dict().items()},
        "partitions": reports,
        "cache": cache_report,
        "external": external,
    }
    if results_dir is not None:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"e1_{subset}_{model.name}_s{seed}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(out / f"splits_{subset}_s{seed}.npz", **split.as_dict())
    return report
