"""
evaluation/data.py

Loads the team's ECG_Splits CSVs (train.csv / val.csv / test.csv: 360
z-scored samples per row, then label, label_enc, record_id, split) and
caches them as .npz next to the CSVs, since parsing the 500 MB train CSV
takes a while.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass
class ECGSplit:
    name: str
    signals: np.ndarray      # (N, 360) float32
    labels: np.ndarray       # (N,) int, label_enc
    patient_ids: np.ndarray  # (N,) MIT-BIH record index the beat came from
    beat_ids: list[str]      # unique per beat: "<split>_<row>"


def load_split(name: str, data_dir: str | Path = ARTIFACTS_DIR) -> ECGSplit:
    data_dir = Path(data_dir)
    cache = data_dir / f"{name}.npz"
    if not cache.exists():
        import pandas as pd

        df = pd.read_csv(data_dir / f"{name}.csv")
        np.savez(cache,
                 signals=df.iloc[:, :360].to_numpy(dtype=np.float32),
                 labels=df["label_enc"].to_numpy(dtype=np.int64),
                 patient_ids=df["record_id"].to_numpy())
    with np.load(cache) as npz:
        signals, labels, patient_ids = npz["signals"], npz["labels"], npz["patient_ids"]
    return ECGSplit(name, signals, labels, patient_ids, [f"{name}_{i}" for i in range(len(labels))])


def stratified_sample(split: ECGSplit, n: int, seed: int = 0) -> np.ndarray:
    """Row indices of a class-stratified sample of ~n beats, keeping at least
    a few beats of every class (A is rare in val/test)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(split.labels, return_counts=True)
    picks = []
    for c, count in zip(classes, counts):
        k = min(count, max(20, int(round(n * count / len(split.labels)))))
        picks.append(rng.choice(np.flatnonzero(split.labels == c), size=k, replace=False))
    return rng.permutation(np.concatenate(picks))
