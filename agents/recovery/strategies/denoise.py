"""
agents/recovery/strategies/denoise.py

Classical signal-poisoning mitigation: Butterworth band-pass + wavelet
denoising (the pipeline spec's "Butterworth or wavelet filtering"). Used
when the autoencoder reconstruction isn't trustworthy enough to substitute
(see signal_repair.py) — denoising keeps the original beat's morphology and
only strips out-of-band and small-coefficient high-frequency content.

    1. zero-phase Butterworth band-pass, 0.5-40 Hz (the same filter the
       MIT-BIH preprocessing notebook applies to raw records)
    2. wavelet soft-thresholding (db4, level 4, universal threshold with
       the noise level estimated from the finest detail coefficients)
    3. restore the input beat's mean and standard deviation, so the
       denoised beat stays on the scale the classifier was trained on

This module does not decide whether the result is good enough: the
orchestrator re-validates the denoised beat and re-runs signal-poisoning
detection on it, and refers it for expert review if it is still flagged.
"""
from __future__ import annotations

import logging

import numpy as np

from core.signal_filters import butterworth_bandpass, wavelet_denoise
from shared.contracts import ECGRecord, RecoveryAction, RecoveryResult

logger = logging.getLogger(__name__)

AGENT_NAME = "recovery"
AGENT_VERSION = "0.1.0"


def denoise_beat(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    y = wavelet_denoise(butterworth_bandpass(x))
    y_std = y.std()
    if y_std < 1e-9:
        return y
    return (y - y.mean()) / y_std * x.std() + x.mean()


def repair_signal_by_denoising(record: ECGRecord) -> RecoveryResult:
    original = np.asarray(record.signal, dtype=np.float64)
    denoised = denoise_beat(original)
    removed = float(np.sqrt(np.mean((original - denoised) ** 2)))
    repaired = record.model_copy(update={"signal": denoised.tolist(), "source": "repaired"})
    logger.info("record_id=%s denoised (Butterworth 0.5-40 Hz + db4 wavelet), removed RMS=%.4f",
                record.record_id, removed)
    return RecoveryResult(
        record_id=record.record_id,
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        action=RecoveryAction.DENOISE_SIGNAL,
        repaired_record=repaired,
        repair_confidence=None,
        notes=f"Butterworth 0.5-40 Hz + db4 wavelet denoising (removed RMS {removed:.3f}); "
              "pending re-check by signal poisoning detection",
    )
