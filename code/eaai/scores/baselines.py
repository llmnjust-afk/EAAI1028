"""Reference score functions: a prefix-feature learner and a masked GRU.

Both consume the padded batch produced by :func:`eaai.data.harmonize.pad_records`
and emit one score per timestep, which is the interface the policy layer expects.
"""

from __future__ import annotations

import numpy as np

from .labels import valid_steps


def _prefix_features(batch: dict) -> np.ndarray:
    """Per-timestep features built only from information available at that time."""
    values = np.asarray(batch["values"], dtype=float)
    mask = np.asarray(batch["mask"], dtype=bool)
    static = np.asarray(batch["static"], dtype=float)
    if (values.ndim != 3 or mask.shape != values.shape or static.ndim != 2 or
            static.shape[0] != values.shape[0] or
            np.asarray(batch["lengths"]).shape != (values.shape[0],)):
        raise ValueError("inconsistent batch shapes")
    valid_steps(batch["lengths"], values.shape[1])
    if np.any(mask & ~np.isfinite(values)) or np.isinf(static).any():
        raise ValueError("observed values and nonmissing static features must be finite")

    running = np.zeros_like(values)
    last = np.zeros((values.shape[0], values.shape[2]))
    for t in range(values.shape[1]):
        last = np.where(mask[:, t, :], values[:, t, :], last)
        running[:, t, :] = last

    observed = np.cumsum(mask, axis=1) / np.arange(1, mask.shape[1] + 1)[None, :, None]
    static_rep = np.repeat(static[:, None, :], values.shape[1], axis=1)
    return np.concatenate([running, observed, static_rep], axis=2)


class PrefixFeatureModel:
    """Logistic regression (or LightGBM) over prefix-summary features."""

    def __init__(self, name: str = "prefix_lr", backend: str = "logistic",
                 random_state: int = 0) -> None:
        self.name = name
        self.backend = backend
        self.random_state = random_state
        self.pipeline = None
        self.n_features_: int | None = None

    def _make(self):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.backend not in ("logistic", "lightgbm"):
            raise ValueError("backend must be 'logistic' or 'lightgbm'")
        if self.backend == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("pip install lightgbm to use backend='lightgbm'") from exc
            estimator = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                       num_leaves=31, random_state=self.random_state)
            return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])

        estimator = LogisticRegression(max_iter=2000, C=1.0, random_state=self.random_state)
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("model", estimator)])

    def fit(self, batch: dict, labels: np.ndarray) -> "PrefixFeatureModel":
        self.pipeline, self.n_features_ = None, None
        features = _prefix_features(batch)
        labels = np.asarray(labels, dtype=float)
        if labels.shape != features.shape[:2]:
            raise ValueError("labels must match the batch patient/time dimensions")
        valid = valid_steps(batch["lengths"], features.shape[1]) & np.isfinite(labels)
        x = features[valid]
        y = np.asarray(labels, dtype=float)[valid]

        if not y.size or not np.isin(y, [0, 1]).all() or y.min() == y.max():
            raise ValueError("training fold contains a single class")
        pipeline = self._make()
        pipeline.fit(x, y.astype(int))
        self.pipeline = pipeline
        self.n_features_ = x.shape[1]
        return self

    def predict(self, batch: dict) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        features = _prefix_features(batch)
        flat = features.reshape(-1, features.shape[2])
        scores = self.pipeline.predict_proba(flat)[:, 1]
        scores = scores.reshape(features.shape[0], features.shape[1])
        valid = valid_steps(batch["lengths"], features.shape[1])
        return np.where(valid, scores, np.nan)


class MaskedGRU:
    """A gated recurrent baseline that consumes values, masks and elapsed time."""

    def __init__(self, name: str = "masked_gru", hidden: int = 64, epochs: int = 30,
                 learning_rate: float = 1e-3, batch_size: int = 64,
                 random_state: int = 0) -> None:
        self.name = name
        self.hidden = hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.module = None
        if (any(not isinstance(v, (int, np.integer)) or v <= 0
                for v in (hidden, epochs, batch_size)) or
                not np.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("positive integer hidden/epochs/batch_size and learning rate required")

    def _build(self, n_features: int):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, d_in: int, hidden: int) -> None:
                super().__init__()
                self.gru = nn.GRU(d_in, hidden, batch_first=True, bidirectional=False)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out).squeeze(-1)

        return Net(n_features, self.hidden)

    def _tensorise(self, batch: dict):
        import torch

        values = np.asarray(batch["values"], dtype=np.float32)
        mask = np.asarray(batch["mask"], dtype=np.float32)
        delta = np.asarray(batch["delta"], dtype=np.float32)
        static = np.asarray(batch["static"], dtype=np.float32)

        if (values.ndim != 3 or mask.shape != values.shape or delta.shape != values.shape or
                static.ndim != 2 or static.shape[0] != values.shape[0] or
                np.asarray(batch["lengths"]).shape != (values.shape[0],) or
                not np.isin(mask, [0, 1]).all()):
            raise ValueError("inconsistent batch shapes or nonbinary observation mask")
        valid_steps(batch["lengths"], values.shape[1])
        if np.any((mask > 0) & ~np.isfinite(values)):
            raise ValueError("observed values must be finite")
        if not np.isfinite(delta).all() or np.any(delta < 0):
            raise ValueError("elapsed times must be finite and nonnegative")
        static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
        delta = np.log1p(np.clip(delta, 0, None))
        filled = np.where(mask > 0, values, 0.0)
        static_rep = np.repeat(static[:, None, :], values.shape[1], axis=1)
        x = np.concatenate([filled, mask, delta, static_rep], axis=2)
        return torch.from_numpy(x)

    def fit(self, batch: dict, labels: np.ndarray) -> "MaskedGRU":
        self.module = None
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        x = self._tensorise(batch)
        labels = np.asarray(labels, dtype=np.float32)
        if labels.shape != tuple(x.shape[:2]):
            raise ValueError("labels must match the batch patient/time dimensions")
        usable = valid_steps(batch["lengths"], x.shape[1]) & np.isfinite(labels)
        if not usable.any() or not np.isin(labels[usable], [0, 1]).all():
            raise ValueError("training labels must contain usable binary targets")
        y = torch.from_numpy(np.where(usable, labels, 0.0))
        valid = torch.from_numpy(usable)

        self.module = self._build(x.shape[2])
        optimiser = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        n = x.shape[0]
        for _ in range(self.epochs):
            self.module.train()
            order = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                if not valid[idx].any():
                    continue
                logits = self.module(x[idx])
                loss = loss_fn(logits, y[idx])
                weight = valid[idx].float()
                loss = (loss * weight).sum() / weight.sum().clamp(min=1.0)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        return self

    def predict(self, batch: dict) -> np.ndarray:
        import torch

        if self.module is None:
            raise RuntimeError("model is not fitted")
        self.module.eval()
        if np.asarray(batch["lengths"]).size == 0:
            raise ValueError("prediction requires at least one patient")
        outputs = []
        with torch.no_grad():
            for start in range(0, len(batch["lengths"]), self.batch_size):
                part = {k: np.asarray(batch[k])[start:start+self.batch_size]
                        for k in ("values", "mask", "delta", "static", "lengths")}
                outputs.append(torch.sigmoid(self.module(self._tensorise(part))).numpy())
        scores = np.concatenate(outputs, axis=0)
        valid = valid_steps(batch["lengths"], scores.shape[1])
        return np.where(valid, scores, np.nan)
