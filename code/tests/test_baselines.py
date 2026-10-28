import numpy as np
import pytest

from eaai.data.harmonize import pad_records
from eaai.data.synthetic import make_scores
from eaai.scores.baselines import MaskedGRU, PrefixFeatureModel, _prefix_features
from eaai.scores.api import align_lengths
from eaai.scores.labels import event_labels, valid_steps


def _batch(n=64, seed=0, width=40):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, width + 1, size=n)
    values = rng.normal(size=(n, width, 3))
    mask = rng.random((n, width, 3)) < 0.6
    mask[np.arange(width)[None, :] >= lengths[:, None]] = False
    delta = rng.exponential(size=(n, width, 3))
    static = rng.normal(size=(n, 2))
    return {"values": values, "mask": mask, "delta": delta, "static": static,
            "lengths": lengths}


def test_prefix_features_are_causal():
    batch = _batch()
    full = _prefix_features(batch)
    tampered = dict(batch)
    tampered["values"] = batch["values"].copy()
    tampered["values"][:, 20:, :] += 50.0
    partial = _prefix_features(tampered)
    assert np.allclose(full[:, :20], partial[:, :20])


def test_event_labels_mark_the_lead_window():
    onset = np.array([30, -1, 10])
    lengths = np.array([40, 40, 15])
    labels = event_labels(onset, lengths, horizon=6)
    assert labels[0, 24:30].tolist() == [1.0] * 6
    assert np.isnan(labels[0, 30:]).all()
    assert labels[1].sum() == 0
    assert labels[2, 4:10].sum() == 6


def test_valid_steps_masks_padding():
    lengths = np.array([5, 2])
    valid = valid_steps(lengths, width=7)
    assert valid[0].sum() == 5 and valid[1].sum() == 2
    assert not valid[1, 2]


def test_prefix_model_fits_and_scores_every_valid_step():
    batch = _batch(n=80)
    labels = (batch["values"][:, :, 0] > 0.5).astype(float)
    model = PrefixFeatureModel().fit(batch, labels)
    scores = model.predict(batch)
    assert scores.shape == (80, 40)
    valid = valid_steps(batch["lengths"], 40)
    assert np.isnan(scores[~valid]).all()
    assert ((scores[valid] >= 0) & (scores[valid] <= 1)).all()


def test_masked_gru_trains_and_predicts():
    pytest.importorskip("torch")
    batch = _batch(n=32, width=24)
    labels = (batch["values"][:, :, 0] > 0.5).astype(float)
    model = MaskedGRU(hidden=8, epochs=2, batch_size=16).fit(batch, labels)
    scores = model.predict(batch)
    assert scores.shape == (32, 24)
    assert np.isfinite(scores[valid_steps(batch["lengths"], 24)]).all()


def test_pad_records_roundtrip_feeds_the_baselines():
    cohort = make_scores(24, seed=3)
    lengths = np.asarray(cohort["lengths"])
    matrix = np.full((len(lengths), int(lengths.max())), np.nan)
    for i, (seq, length) in enumerate(zip(cohort["scores"], lengths)):
        matrix[i, :length] = np.asarray(seq, dtype=float)[:length]
    batch = {"values": matrix[:, :, None], "mask": ~np.isnan(matrix[:, :, None]),
             "delta": np.ones_like(matrix)[:, :, None],
             "static": np.zeros((matrix.shape[0], 1)),
             "lengths": np.asarray(cohort["lengths"])}
    labels = event_labels(cohort["onset"], cohort["lengths"], horizon=6, width=matrix.shape[1])
    assert labels.shape == matrix.shape
    model = PrefixFeatureModel().fit(batch, labels)
    assert np.isfinite(model.predict(batch)[0, :3]).all()


def test_align_lengths_accepts_ragged_sequences():
    seqs = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])]
    out = align_lengths(seqs, np.array([3, 2]))
    assert out.shape == (2, 3)
    assert out[0].tolist() == [0.1, 0.2, 0.3]
    assert np.isnan(out[1, 2])
