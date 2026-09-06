"""
agents/recovery/strategies/quarantine.py

The two "give up cleanly" outcomes. Quarantine means "a human should
look at this"; reject means "don't even hold onto it for review"
(e.g. validation-agent-level garbage that slipped through, or a repair
attempt whose own output failed sanity checks). Most callers want
quarantine, not reject — reject is reserved for callers that pass an
explicit reason indicating the record itself is unusable.
"""
from __future__ import annotations

from shared.contracts import RecoveryAction, RecoveryResult

AGENT_NAME = "recovery"
AGENT_VERSION = "0.1.0"


def quarantine(record_id: str, reason: str) -> RecoveryResult:
    """Route a record to manual clinician/reviewer queue. No repaired
    signal or label is produced — the orchestrator should stop the
    pipeline here rather than forwarding to Diagnosis."""
    return RecoveryResult(
        record_id=record_id,
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        action=RecoveryAction.QUARANTINE,
        notes=reason,
    )


def reject(record_id: str, reason: str) -> RecoveryResult:
    """Discard the record outright. Use sparingly and only when you have
    an explicit reason the record cannot even be safely queued for
    manual review (e.g. a repair produced an invalid signal)."""
    return RecoveryResult(
        record_id=record_id,
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        action=RecoveryAction.REJECT,
        notes=reason,
    )
