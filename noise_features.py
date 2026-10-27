"""Sriram's signal-only ECG noise feature extractor.

Features cover the agreed spectral, wavelet-residual and morphology tracks.
They accept one normalized 360-sample beat and never read labels or attack
ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
import pywt

FS = 360.0
EPS = 1e-12


def as_beat(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.shape == (360, 1):
        x = x[:, 0]
    if x.shape != (360,):
        raise ValueError(f"Expected signal shape (360, 1) or (360,), got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or infinity.")
    return x


def _bandpower(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq <= high)
    return float(trapezoid(psd[mask], freq[mask])) if mask.any() else 0.0


@dataclass
class MorphologyTemplateBank:
    """Class-agnostic clean templates; fit only on known-clean training beats."""
    templates: np.ndarray

    @classmethod
    def fit(cls, clean_beats: Iterable[np.ndarray], max_templates: int = 32) -> "MorphologyTemplateBank":
        beats = np.asarray([as_beat(x) for x in clean_beats], dtype=np.float64)
        if len(beats) == 0:
            raise ValueError("At least one clean beat is required to fit templates.")
        indices = np.linspace(0, len(beats) - 1, min(max_templates, len(beats)), dtype=int)
        return cls(templates=beats[indices])

    def best_correlation(self, signal: np.ndarray) -> float:
        x = as_beat(signal)
        correlations = [np.corrcoef(x, template)[0, 1] for template in self.templates]
        return float(np.nanmax(np.nan_to_num(correlations, nan=0.0)))

    def save(self, path: str) -> None:
        np.savez_compressed(path, templates=self.templates)

    @classmethod
    def load(cls, path: str) -> "MorphologyTemplateBank":
        with np.load(path) as artifact:
            return cls(templates=artifact["templates"])


def extract_noise_features(signal: np.ndarray, templates: MorphologyTemplateBank | None = None) -> dict[str, float]:
    x = as_beat(signal)
    freq, psd = welch(x, fs=FS, nperseg=min(256, len(x)), detrend="constant")
    total = _bandpower(freq, psd, 0.05, 100.0) + EPS
    baseline = _bandpower(freq, psd, 0.05, 0.7) / total
    powerline = _bandpower(freq, psd, 49.0, 51.0) / total
    emg = _bandpower(freq, psd, 20.0, 100.0) / total

    coeffs = pywt.wavedec(x, "db4", level=4, mode="symmetric")
    detail = np.concatenate(coeffs[1:])
    mad_sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    residual_kurtosis = float(np.mean((detail - detail.mean()) ** 4) / (detail.var() ** 2 + EPS))
    template_correlation = templates.best_correlation(x) if templates is not None else 1.0
    return {
        "baseline_power_ratio": float(baseline),
        "powerline_50hz_ratio": float(powerline),
        "emg_power_ratio": float(emg),
        "wavelet_mad_sigma": float(mad_sigma),
        "wavelet_residual_kurtosis": residual_kurtosis,
        "template_correlation": template_correlation,
    }
