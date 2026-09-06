"""
agents/recovery/strategies/signal_repair.py

Repairs a record flagged SIGNAL_POISONED by substituting the autoencoder
reconstruction the signal-poisoning agent's reconstruction check already
computed. This module does NOT retrain or rerun the autoencoder — it
only consumes the reconstruction array and a confidence score, which
should arrive via the "reconstruction_error" Evidence's
`metadata["reconstruction"]` and `.confidence` fields respectively (see
recovery_agent.py).

IMPORTANT — do not gate trust on "how different is the reconstruction
from the original signal". A large difference is *expected and desired*
here: the whole point of a poisoning attack is that the original signal
is corrupted, so a good repair is SUPPOSED to look very different from
it. Gating on that residual would quarantine almost every genuine
attack and only "repair" near-clean signals that barely needed it.

Instead, trust is gated on `reconstruction_confidence` — a score the
signal-poisoning agent should already produce alongside the
reconstruction, representing how in-distribution / typical this
reconstruction is relative to the autoencoder's own training
distribution of clean beats (e.g. a calibrated percentile of the AE's
reconstruction-loss distribution on held-out clean data, or an ensemble/
dropout-based uncertainty estimate). If the signal-poisoning agent
doesn't yet produce this, treat it as a `TODO` there rather than trying
to approximate it here from the original-vs-reconstruction distance.
"""
from __future__ import annotations

import logging

import numpy as np

from agents.recovery.strategies.quarantine import quarantine
from shared.contracts import ECGRecord, RecoveryAction, RecoveryResult

logger = logging.getLogger(__name__)

AGENT_NAME = "recovery"
AGENT_VERSION = "0.1.0"

EXPECTED_SHAPE = (360,)


def _validate_reconstruction(reconstruction: np.ndarray) -> np.ndarray:
    arr = np.asarray(reconstruction, dtype=float)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"expected reconstruction shape {EXPECTED_SHAPE}, got {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError("autoencoder reconstruction contains NaN")
    return arr


def repair_signal(
    record: ECGRecord,
    autoencoder_reconstruction: np.ndarray,
    reconstruction_confidence: float,
    min_confidence: float = 0.6,
) -> RecoveryResult:
    """Substitute the denoised (reconstructed) signal for the original.

    Args:
        record: the flagged record.
        autoencoder_reconstruction: the reconstruction array already
            produced by the signal-poisoning agent's reconstruction
            check, shape (360,).
        reconstruction_confidence: [0,1] score for how much to trust
            THIS reconstruction as a plausible clean signal — NOT a
            distance from the original. See module docstring.
        min_confidence: below this, quarantine instead of forwarding a
            reconstruction the autoencoder itself wasn't confident in.
    """
    if not (0.0 <= reconstruction_confidence <= 1.0):
        raise ValueError(
            f"reconstruction_confidence must be in [0,1], got {reconstruction_confidence}"
        )

    reconstruction = _validate_reconstruction(autoencoder_reconstruction)
    original = np.asarray(record.signal, dtype=float)
    if original.shape != reconstruction.shape:
        raise ValueError(
            f"record signal shape {original.shape} does not match "
            f"reconstruction shape {reconstruction.shape}"
        )

    if reconstruction_confidence < min_confidence:
        logger.info(
            "record_id=%s reconstruction_confidence %.3f below min_confidence %.3f, quarantining",
            record.record_id, reconstruction_confidence, min_confidence,
        )
        return quarantine(
            record.record_id,
            f"reconstruction confidence {reconstruction_confidence:.2f} below "
            f"min_confidence {min_confidence:.2f}",
        )

    # Purely informational — logged for audit, never used to gate the decision.
    residual = float(np.mean(np.abs(original - reconstruction)))

    repaired = record.model_copy(
        update={"signal": reconstruction.tolist(), "source": "repaired"}
    )
    logger.info(
        "record_id=%s signal reconstructed, confidence=%.3f (residual-from-original=%.4f, informational only)",
        record.record_id, reconstruction_confidence, residual,
    )

    return RecoveryResult(
        record_id=record.record_id,
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        action=RecoveryAction.RECONSTRUCT_SIGNAL,
        repaired_record=repaired,
        repair_confidence=reconstruction_confidence,
        notes=f"reconstructed signal, confidence={reconstruction_confidence:.2f} "
              f"(residual-from-original={residual:.4f})",
    )
