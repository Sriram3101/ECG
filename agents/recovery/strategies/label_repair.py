"""
agents/recovery/strategies/label_repair.py

Repairs a record flagged LABEL_POISONED by re-labeling it with the
k-fold out-of-fold (OOF) consensus probabilities the label-poisoning
agent's confident-learning check already computed. This module does NOT
recompute those probabilities — it only consumes them. If you don't have
them threaded through yet, see the note in recovery_agent.py about how
they should reach this agent via the orchestrator's `artifacts` dict.
"""
from __future__ import annotations

import logging

import numpy as np

from agents.recovery.strategies.quarantine import quarantine
from shared.contracts import ECGRecord, RecoveryAction, RecoveryResult

logger = logging.getLogger(__name__)

AGENT_NAME = "recovery"
AGENT_VERSION = "0.1.0"


def _validate_probabilities(oof_probabilities: np.ndarray) -> np.ndarray:
    arr = np.asarray(oof_probabilities, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"oof_probabilities must be 1-D, got shape {arr.shape}")
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("oof_probabilities must all be in [0,1]")
    if not np.isclose(arr.sum(), 1.0, atol=1e-2):
        raise ValueError(f"oof_probabilities must sum to ~1.0, got {arr.sum():.4f}")
    return arr


def repair_label(
    record: ECGRecord,
    oof_probabilities: np.ndarray,
    min_confidence: float = 0.7,
) -> RecoveryResult:
    """Re-label using the consensus (argmax) of the OOF probabilities.

    Args:
        record: the flagged record (its current `.label` is the
            suspected-poisoned one).
        oof_probabilities: per-class consensus probabilities from the
            label-poisoning agent's confident-learning check, shape
            (num_classes,).
        min_confidence: if the consensus probability doesn't clear this
            bar, the relabel isn't trustworthy enough to act on
            automatically — quarantine for manual review instead of
            silently writing a low-confidence label into the pipeline.
    """
    arr = _validate_probabilities(oof_probabilities)
    consensus_label = int(np.argmax(arr))
    consensus_confidence = float(np.max(arr))

    if consensus_confidence < min_confidence:
        logger.info(
            "record_id=%s relabel confidence %.3f below min_confidence %.3f, quarantining",
            record.record_id, consensus_confidence, min_confidence,
        )
        return quarantine(
            record.record_id,
            f"relabel confidence {consensus_confidence:.2f} below "
            f"min_confidence {min_confidence:.2f}",
        )

    unchanged = record.label is not None and consensus_label == record.label
    notes = (
        "consensus label matches original — detector may be a false positive, "
        "review before trusting downstream diagnosis"
        if unchanged
        else f"relabeled {record.label} -> {consensus_label}"
    )

    repaired = record.model_copy(update={"label": consensus_label, "source": "repaired"})
    logger.info("record_id=%s %s (confidence %.3f)", record.record_id, notes, consensus_confidence)

    return RecoveryResult(
        record_id=record.record_id,
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        action=RecoveryAction.RELABEL,
        repaired_record=repaired,
        repair_confidence=consensus_confidence,
        notes=notes,
    )
