"""
core/signal_filters.py

Filters shared by the signal-poisoning agent (signal-quality metrics) and
the Recovery agent (denoising), so both use the same definition of "the
clean part of the beat".
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, welch

FS = 360.0


def butterworth_bandpass(x: np.ndarray, low_hz: float = 0.5, high_hz: float = 40.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass — the same filter the MIT-BIH
    preprocessing notebook applies to raw records."""
    nyquist = 0.5 * FS
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Soft-threshold wavelet denoising, universal threshold with the noise
    level estimated from the finest detail coefficients."""
    x = np.asarray(x, dtype=np.float64)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="symmetric")[: len(x)]


def estimated_snr_db(x: np.ndarray) -> float:
    """SNR of the beat against its own wavelet-denoised version."""
    x = np.asarray(x, dtype=np.float64)
    clean = wavelet_denoise(x)
    noise_power = float(np.var(x - clean)) + 1e-12
    return float(10 * np.log10((float(np.var(clean)) + 1e-12) / noise_power))


def spectral_entropy(x: np.ndarray) -> float:
    """Normalized Shannon entropy of the Welch power spectrum, in [0, 1].
    Clean beats concentrate power in a few low-frequency bands; broadband
    noise pushes this toward 1."""
    freq, psd = welch(np.asarray(x, dtype=np.float64), fs=FS, nperseg=min(256, len(x)))
    p = psd / (psd.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
