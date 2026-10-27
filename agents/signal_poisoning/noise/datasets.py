"""Separate, leakage-safe signal-corruption artifact generation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def add_gaussian_noise(signals: np.ndarray, sigma: float = 0.01, seed: int = 42) -> np.ndarray:
    """Matches the supplied attack: z-score-space additive Gaussian noise."""
    return (np.asarray(signals, dtype=np.float32) + np.random.default_rng(seed).normal(0, sigma, np.asarray(signals).shape)).astype(np.float32)


def save_corrupted_signal_set(path: str | Path, signals: np.ndarray, labels: np.ndarray, record_ids: np.ndarray,
                              is_corrupted: np.ndarray, noise_type: str | None, severity: float | None) -> None:
    """Writes detector inputs and evaluation-only truth in separate arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signals=np.asarray(signals, dtype=np.float32), labels=np.asarray(labels), record_ids=np.asarray(record_ids))
    truth = [{"is_corrupted": bool(flag), "noise_type": noise_type if flag else None,
              "severity": severity if flag else None} for flag in is_corrupted]
    path.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
