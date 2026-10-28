"""Audit the official 2019 A/B record format and label convention.

This checks ingestion properties only. It does not estimate discrimination,
alarm burden or clinical effectiveness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaai.data.harmonize import harmonise
from eaai.data.loaders import load_challenge2019
from eaai.data.schema import CHALLENGE_2019_DYNAMIC


def audit(root: Path, subset: str, limit: int | None = None) -> dict:
    frames = load_challenge2019(root, subset=subset, limit=limit)
    if not frames:
        raise ValueError("no records found")
    lengths, onsets, left_censored, positive = [], [], [], []
    observed = np.zeros(len(CHALLENGE_2019_DYNAMIC), dtype=float)
    total_steps = 0
    for pid, frame in frames:
        record = harmonise(frame, variables=CHALLENGE_2019_DYNAMIC,
                           label_lead_hours=6)
        if record.values.shape != (record.length, len(CHALLENGE_2019_DYNAMIC)):
            raise AssertionError(f"{pid}: inconsistent record shape")
        if np.any(record.mask & ~np.isfinite(record.values)):
            raise AssertionError(f"{pid}: nonfinite observed value")
        if not np.isfinite(record.delta).all() or np.any(record.delta < 0):
            raise AssertionError(f"{pid}: invalid elapsed time")
        lengths.append(record.length)
        onsets.append(-1 if record.onset_index is None else record.onset_index)
        left_censored.append(record.onset_left_censored)
        positive.append(record.onset_index is not None)
        observed += record.mask.sum(axis=0)
        total_steps += record.length

    lengths = np.asarray(lengths)
    onsets = np.asarray(onsets)
    positive = np.asarray(positive)
    left_censored = np.asarray(left_censored)
    detected = positive & ~left_censored
    return {
        "source": "https://physionet.org/content/challenge-2019/1.0.0/",
        "subset": subset,
        "records": int(len(frames)),
        "positive_records": int(positive.sum()),
        "left_censored_records": int(left_censored.sum()),
        "negative_records": int((~positive).sum()),
        "records_with_identifiable_onset": int(detected.sum()),
        "onset_steps": {
            "min": int(onsets[detected].min()) if detected.any() else None,
            "median": float(np.median(onsets[detected])) if detected.any() else None,
            "max": int(onsets[detected].max()) if detected.any() else None,
        },
        "length_hours": {
            "min": int(lengths.min()), "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max()),
        },
        "observed_fraction": {
            name: float(observed[i] / total_steps)
            for i, name in enumerate(CHALLENGE_2019_DYNAMIC)},
        "interpretation": "data ingestion and label-boundary QC, not model performance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset", default="A", choices=("A", "B"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.data_root, args.subset, args.limit)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
