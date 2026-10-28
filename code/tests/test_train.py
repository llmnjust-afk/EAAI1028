import numpy as np

from eaai.experiments.e1_train import evaluate_scores, run
from eaai.scores.api import ScoreCache
from eaai.scores.baselines import PrefixFeatureModel


def _batch(n=64, seed=0, width=30):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(15, width + 1, size=n)
    values = rng.normal(size=(n, width, 2))
    mask = np.ones((n, width, 2), dtype=bool)
    mask[np.arange(width)[None, :] >= lengths[:, None]] = False
    return {"values": values, "mask": mask, "delta": np.ones_like(values),
            "static": np.zeros((n, 1)), "lengths": lengths}


def _labels(batch):
    return (batch["values"][:, :, 0] > 0.5).astype(float)


def test_evaluate_scores_reports_discrimination():
    batch = _batch()
    labels = _labels(batch)
    scores = np.clip(batch["values"][:, :, 0] * 0.2 + 0.5, 0, 1)
    report = evaluate_scores(scores, labels, batch["lengths"])
    assert 0.0 <= report["auroc"] <= 1.0
    assert report["n_timesteps"] == int(batch["lengths"].sum())
    assert report["n_positive"] > 0


def test_run_caches_scores_and_roundtrips(tmp_path):
    train, test = _batch(seed=1), _batch(seed=2)
    cache = ScoreCache(tmp_path)
    report = run(PrefixFeatureModel(), train, _labels(train), test, _labels(test),
                 cache=cache, key=("synthetic", "prefix_lr", 0),
                 test_onset=np.full(64, -1))
    assert report["cached"] and report["model"] == "prefix_lr"
    stored = cache.read("synthetic", "prefix_lr", 0)
    assert stored["scores"].shape == (64, 30)
