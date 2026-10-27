"""
agents/recovery/recovery_agent.py

Dispatches a flagged record to the right repair strategy based on the
Decision agent's verdict. Only ever called when
`decision.recommended_action != "proceed"` — the orchestrator should not
call this agent on a CLEAN verdict (see `run()`'s guard below, which
raises rather than silently no-op'ing).

Artifact threading
------------------
Label repair and signal repair both need data that the *detection*
agents already computed (OOF probabilities, autoencoder reconstruction).
Rather than recomputing them here, this agent expects them to be passed
in one of two ways:

1. Explicitly, via keyword args:
       recovery_agent.run(decision, record,
                           oof_probabilities=...,
                           autoencoder_reconstruction=...,
                           reconstruction_confidence=...)

2. Attached on the DecisionRecord's evidence list by the detection
   agents: the "reconstruction_error" Evidence should carry
   `metadata={"reconstruction": [...]}` for the array, and its own
   `.confidence` field is read as `reconstruction_confidence` (see
   module docstring in strategies/signal_repair.py for why this must be
   "how trustworthy is this reconstruction", not "how close is it to
   the original"). Any evidence with `metadata["oof_probabilities"]` is
   read for the label side. Use `RecoveryAgent.extract_artifacts(decision)`
   to pull these out automatically.

Pick one convention and use it consistently across your pipeline; don't
mix both silently, since (1) always wins if both are present, which can
mask a broken (2).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agents.recovery.strategies import label_repair, signal_repair
from agents.recovery.strategies.denoise import repair_signal_by_denoising
from agents.recovery.strategies.quarantine import quarantine as _quarantine
from shared.contracts import DecisionRecord, ECGRecord, RecoveryAction, RecoveryResult, Verdict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "repair_thresholds.yaml"

# Verdicts the Recovery agent is allowed to be called on. CLEAN should
# never reach here — the orchestrator's job, not this agent's, to skip
# straight to Diagnosis for clean records.
REPAIRABLE_VERDICTS = {
    Verdict.LABEL_POISONED,
    Verdict.SIGNAL_POISONED,
    Verdict.BOTH_POISONED,
    Verdict.UNCERTAIN,
}


SIGNAL_REPAIR_FALLBACKS = ("quarantine", "denoise")


class RecoveryAgentConfig:
    def __init__(self, min_label_confidence: float, min_signal_confidence: float,
                 signal_repair_fallback: str = "quarantine"):
        """`signal_repair_fallback`: what to do with a SIGNAL_POISONED record
        whose autoencoder reconstruction is below `min_signal_confidence` —
        "quarantine" (send to review) or "denoise" (Butterworth + wavelet
        filtering, which the orchestrator then re-checks)."""
        if signal_repair_fallback not in SIGNAL_REPAIR_FALLBACKS:
            raise ValueError(f"signal_repair_fallback must be one of {SIGNAL_REPAIR_FALLBACKS}")
        self.signal_repair_fallback = signal_repair_fallback
        if not (0.0 <= min_label_confidence <= 1.0):
            raise ValueError("min_label_confidence must be in [0,1]")
        if not (0.0 <= min_signal_confidence <= 1.0):
            raise ValueError("min_signal_confidence must be in [0,1]")
        self.min_label_confidence = min_label_confidence
        self.min_signal_confidence = min_signal_confidence

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "RecoveryAgentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"recovery agent config not found at {path}. "
                "Copy agents/recovery/config/repair_thresholds.yaml into place, "
                "or pass an explicit path to RecoveryAgentConfig.from_yaml()."
            )
        raw = yaml.safe_load(path.read_text()) or {}
        return cls(
            min_label_confidence=raw["min_label_confidence"],
            min_signal_confidence=raw["min_signal_confidence"],
            signal_repair_fallback=raw.get("signal_repair_fallback", "quarantine"),
        )


class RecoveryAgent:
    version = "0.1.0"
    name = "recovery"

    def __init__(self, config: RecoveryAgentConfig | None = None):
        self.config = config or RecoveryAgentConfig.from_yaml()

    def run(
        self,
        decision: DecisionRecord,
        record: ECGRecord,
        *,
        oof_probabilities: np.ndarray | None = None,
        autoencoder_reconstruction: np.ndarray | None = None,
        reconstruction_confidence: float | None = None,
    ) -> RecoveryResult:
        if decision.verdict not in REPAIRABLE_VERDICTS:
            raise ValueError(
                f"RecoveryAgent should not be called for verdict "
                f"'{decision.verdict.value}' — the orchestrator should route "
                "CLEAN records straight to Diagnosis without invoking Recovery."
            )
        if decision.record_id != record.record_id:
            raise ValueError(
                f"decision.record_id ({decision.record_id!r}) does not match "
                f"record.record_id ({record.record_id!r})"
            )

        if decision.verdict == Verdict.LABEL_POISONED:
            oof = self._require(oof_probabilities, "oof_probabilities", decision)
            return label_repair.repair_label(record, oof, self.config.min_label_confidence)

        if decision.verdict == Verdict.SIGNAL_POISONED:
            recon = self._require(autoencoder_reconstruction, "autoencoder_reconstruction", decision)
            recon_conf = self._require(reconstruction_confidence, "reconstruction_confidence", decision)
            result = signal_repair.repair_signal(record, recon, recon_conf, self.config.min_signal_confidence)
            if result.action == RecoveryAction.QUARANTINE and self.config.signal_repair_fallback == "denoise":
                logger.info("record_id=%s reconstruction not trusted, falling back to denoising", record.record_id)
                return repair_signal_by_denoising(record)
            return result

        if decision.verdict == Verdict.BOTH_POISONED:
            logger.info("record_id=%s both label and signal poisoning detected, quarantining", record.record_id)
            return _quarantine(
                record.record_id,
                "both label and signal poisoning detected — compounding risk, routed to manual review",
            )

        # UNCERTAIN
        logger.info("record_id=%s uncertain verdict, quarantining", record.record_id)
        return _quarantine(
            record.record_id,
            f"uncertain verdict (label_score={decision.label_score:.2f}, "
            f"signal_score={decision.signal_score:.2f}) — routed to manual review",
        )

    def _require(self, value: Any, name: str, decision: DecisionRecord) -> Any:
        if value is not None:
            return value
        extracted = self.extract_artifacts(decision).get(name)
        if extracted is not None:
            return extracted
        raise KeyError(
            f"RecoveryAgent needs '{name}' to repair a "
            f"{decision.verdict.value} record (record_id={decision.record_id!r}), "
            f"but it was not passed explicitly and could not be found in "
            f"decision.evidence[*].metadata. Either pass it as a keyword "
            f"argument to run(), or have the detection agent attach it via "
            f"Evidence.metadata (see this module's docstring)."
        )

    @staticmethod
    def extract_artifacts(decision: DecisionRecord) -> dict[str, Any]:
        """Best-effort extraction of repair artifacts from
        `decision.evidence`, keyed by the same names `run()` accepts as
        keyword arguments:

          - "reconstruction_error" evidence -> metadata["reconstruction"]
            -> returned as "autoencoder_reconstruction"
          - "reconstruction_error" evidence -> its own `.confidence` field
            -> returned as "reconstruction_confidence"
            (this is intentionally NOT derived from the reconstruction
            array itself — see strategies/signal_repair.py's docstring
            for why original-vs-reconstruction distance is the wrong
            signal to gate repair trust on)
          - any evidence -> metadata["oof_probabilities"]
            -> returned as "oof_probabilities"
            (label-side evidence isn't on DecisionRecord.evidence by
            default — see the build-plan doc's note on threading
            oof_probabilities through the orchestrator's `artifacts` dict
            instead, if your label-poisoning agent's evidence doesn't
            flow through here.)
        """
        artifacts: dict[str, Any] = {}
        for e in decision.evidence:
            if e.check_name == "reconstruction_error":
                if "reconstruction" in e.metadata:
                    artifacts["autoencoder_reconstruction"] = e.metadata["reconstruction"]
                artifacts["reconstruction_confidence"] = e.confidence
            if "oof_probabilities" in e.metadata:
                artifacts["oof_probabilities"] = e.metadata["oof_probabilities"]
        return artifacts
