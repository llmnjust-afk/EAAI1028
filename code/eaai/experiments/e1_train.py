"""Train one score function, cache its per-timestep outputs, report discrimination.

The point of caching is that every policy experiment downstream is post-hoc: the
backbone is trained once per (cohort, model, seed) and never again.
"""

from __future__ import annotations

import numpy as np

from ..eval.metrics import discrimination, expected_calibration_error
from ..scores.api import ScoreCache
from ..scores.labels import valid_steps


def evaluate_scores(scores: np.ndarray, labels: np.ndarray,
                    lengths: np.ndarray, n_bins: int = 10) -> dict:
    """Flatten valid timesteps and report AUROC, AUPRC and calibration error."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if scores.ndim != 2 or labels.shape != scores.shape or len(lengths) != scores.shape[0]:
        raise ValueError("scores, labels and lengths must have consistent shapes")
    valid = valid_steps(lengths, scores.shape[1]) & np.isfinite(labels)
    if not valid.any() or not np.isfinite(scores[valid]).all():
        raise ValueError("evaluation requires finite scores at usable target timesteps")
    y = np.asarray(labels, dtype=float)[valid]
    s = scores[valid]

    report = discrimination(y, s)
    report["ece"] = expected_calibration_error(y, s, n_bins=n_bins)
    report["n_timesteps"] = int(valid.sum())
    report["n_positive"] = int(y.sum())

    lengths = np.asarray(lengths, dtype=int)
    per_patient = np.array([np.nanmax(scores[i, :lengths[i]])
                            for i in range(scores.shape[0])])
    report["patient_max_mean"] = float(np.nanmean(per_patient))
    return report


def run(model, train_batch: dict, train_labels: np.ndarray,
        test_batch: dict, test_labels: np.ndarray,
        cache: ScoreCache | None = None, key: tuple[str, str, int] | None = None,
        test_onset: np.ndarray | None = None, n_bins: int = 10) -> dict:
    """Fit ``model``, score the test fold, optionally persist the scores."""
    fitted = model.fit(train_batch, train_labels)
    scores = fitted.predict(test_batch)
    report = evaluate_scores(scores, test_labels, np.asarray(test_batch["lengths"]),
                             n_bins=n_bins)
    report["model"] = getattr(model, "name", type(model).__name__)

    if cache is not None:
        if key is None:
            raise ValueError("a (cohort, model, seed) key is required to cache scores")
        if test_onset is None:
            raise ValueError("test_onset is required to cache scores")
        cohort, name, seed = key
        cache.write(cohort, name, seed, scores, np.asarray(test_onset),
                    np.asarray(test_batch["lengths"]))
        report["cached"] = True
    return report
