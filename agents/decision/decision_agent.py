"""
agents/decision/decision_agent.py

Fuses label-poisoning and signal-poisoning evidence into one verdict.

Inputs it expects (see orchestrator/pipeline.py in the build-plan doc):
  - label_score:      float in [0,1], already fused/calibrated by the
                       label-poisoning agent's own "Score fusion" step.
  - signal_evidence:   list[Evidence] — the four *raw* signal-poisoning
                       checks (reconstruction_error, feature_space_distance,
                       prediction_fragility, gradient_sensitivity). This
                       agent fuses them itself (see fusion.py).

Output: a DecisionRecord (shared/contracts.py) carrying the verdict,
confidence, both component scores, the raw evidence for audit, and a
recommended_action that the orchestrator uses to decide whether to call
the Recovery agent at all.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import yaml

from agents.decision.fusion import (
    DEFAULT_SIGNAL_CHECK_WEIGHTS,
    fuse_signal_evidence,
    fuse_signal_evidence_noisy_or,
)
from shared.contracts import DecisionRecord, Evidence, Verdict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "thresholds.yaml"

# Signal check that fired hardest -> the attack mechanism it points to.
_SIGNAL_ATTACK_TYPES = {
    "reconstruction_error": "waveform_corruption",
    "feature_space_distance": "out_of_distribution_signal",
    "prediction_fragility": "adversarial_perturbation",
    "gradient_sensitivity": "adversarial_perturbation",
}

_FUSION_STRATEGIES: dict[str, Callable[[list[Evidence], dict], float]] = {
    "weighted_mean": fuse_signal_evidence,
    "noisy_or": fuse_signal_evidence_noisy_or,
}


class DecisionAgentConfig:
    """Typed wrapper around thresholds.yaml so bad config fails at load
    time, not three calls deep into a pipeline run."""

    def __init__(
        self,
        label_threshold: float,
        signal_threshold: float,
        uncertain_margin: float = 0.0,
        fusion_strategy: str = "weighted_mean",
        signal_check_weights: dict[str, float] | None = None,
    ):
        if not (0.0 <= label_threshold <= 1.0):
            raise ValueError("label_threshold must be in [0,1]")
        if not (0.0 <= signal_threshold <= 1.0):
            raise ValueError("signal_threshold must be in [0,1]")
        if uncertain_margin < 0.0:
            raise ValueError("uncertain_margin must be >= 0")
        if fusion_strategy not in _FUSION_STRATEGIES:
            raise ValueError(
                f"unknown fusion_strategy '{fusion_strategy}', "
                f"expected one of {sorted(_FUSION_STRATEGIES)}"
            )

        self.label_threshold = label_threshold
        self.signal_threshold = signal_threshold
        self.uncertain_margin = uncertain_margin
        self.fusion_strategy = fusion_strategy
        self.signal_check_weights = signal_check_weights or DEFAULT_SIGNAL_CHECK_WEIGHTS

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "DecisionAgentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"decision agent config not found at {path}. "
                "Copy agents/decision/config/thresholds.yaml into place, "
                "or pass an explicit path to DecisionAgentConfig.from_yaml()."
            )
        raw = yaml.safe_load(path.read_text()) or {}
        return cls(
            label_threshold=raw["label_threshold"],
            signal_threshold=raw["signal_threshold"],
            uncertain_margin=raw.get("uncertain_margin", 0.0),
            fusion_strategy=raw.get("fusion_strategy", "weighted_mean"),
            signal_check_weights=raw.get("signal_check_weights"),
        )


class DecisionAgent:
    """Deterministic, config-driven evidence fusion. No learned model here
    by default — see the build-plan doc, Section 4, for when/how to swap
    in a trained meta-classifier once you have labeled attack-simulation
    data to train and validate it against."""

    version = "0.1.0"
    name = "decision"

    def __init__(self, config: DecisionAgentConfig | None = None):
        self.config = config or DecisionAgentConfig.from_yaml()
        self._fuse = _FUSION_STRATEGIES[self.config.fusion_strategy]

    def run(
        self,
        record_id: str,
        label_score: float,
        signal_evidence: list[Evidence],
        label_evidence: list[Evidence] | None = None,
    ) -> DecisionRecord:
        """`label_evidence` (the label-poisoning agent's per-check evidence)
        is optional and only carried on the DecisionRecord for audit and
        for the Recovery agent's artifact extraction — the label side is
        already fused into `label_score` and is not re-fused here."""
        if not (0.0 <= label_score <= 1.0):
            raise ValueError(f"label_score must be in [0,1], got {label_score}")

        signal_score = self._fuse(signal_evidence, self.config.signal_check_weights)

        verdict, action = self._decide(label_score, signal_score)
        confidence = self._confidence(label_score, signal_score, verdict)
        attack_type = self._attack_type(verdict, signal_evidence)
        severity = self._severity(verdict, label_score, signal_score)

        logger.info(
            "decision record_id=%s label_score=%.3f signal_score=%.3f "
            "verdict=%s attack_type=%s severity=%s action=%s confidence=%.3f",
            record_id, label_score, signal_score, verdict.value, attack_type, severity, action, confidence,
        )

        return DecisionRecord(
            record_id=record_id,
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            label_score=label_score,
            signal_score=signal_score,
            evidence=list(label_evidence or []) + list(signal_evidence),
            recommended_action=action,
            attack_type=attack_type,
            severity=severity,
        )

    @staticmethod
    def _attack_type(verdict: Verdict, signal_evidence: list[Evidence]) -> str | None:
        """Names the most likely attack mechanism from which evidence fired."""
        if verdict == Verdict.CLEAN:
            return None
        if verdict == Verdict.LABEL_POISONED:
            return "label_flip"
        weighted = [e for e in signal_evidence if e.check_name in _SIGNAL_ATTACK_TYPES]
        signal_type = (_SIGNAL_ATTACK_TYPES[max(weighted, key=lambda e: e.score).check_name]
                       if weighted else "signal_tampering")
        if verdict == Verdict.BOTH_POISONED:
            return f"label_flip+{signal_type}"
        if verdict == Verdict.UNCERTAIN:
            return "undetermined"
        return signal_type

    def _severity(self, verdict: Verdict, label_score: float, signal_score: float) -> str:
        """How far past its threshold the triggering score is: low within
        the first third of the remaining range, moderate within the second,
        high beyond that or when both channels are poisoned."""
        if verdict == Verdict.CLEAN:
            return "none"
        if verdict in (Verdict.BOTH_POISONED,):
            return "high"
        if verdict == Verdict.UNCERTAIN:
            return "low"
        score, threshold = ((label_score, self.config.label_threshold) if verdict == Verdict.LABEL_POISONED
                            else (signal_score, self.config.signal_threshold))
        excess = (score - threshold) / max(1.0 - threshold, 1e-9)
        return "low" if excess < 1 / 3 else "moderate" if excess < 2 / 3 else "high"

    def _decide(self, label_score: float, signal_score: float) -> tuple[Verdict, str]:
        cfg = self.config
        label_flag = label_score >= cfg.label_threshold
        signal_flag = signal_score >= cfg.signal_threshold

        if label_flag and signal_flag:
            return Verdict.BOTH_POISONED, "quarantine"
        if label_flag:
            return Verdict.LABEL_POISONED, "repair"
        if signal_flag:
            return Verdict.SIGNAL_POISONED, "repair"

        # Uncertain band: neither threshold was crossed, but at least one
        # score is close enough to it that "confidently clean" would be
        # overstating it. Disabled by default (uncertain_margin: 0).
        if cfg.uncertain_margin > 0:
            near_label = cfg.label_threshold - cfg.uncertain_margin <= label_score < cfg.label_threshold
            near_signal = cfg.signal_threshold - cfg.uncertain_margin <= signal_score < cfg.signal_threshold
            if near_label or near_signal:
                return Verdict.UNCERTAIN, "repair"

        return Verdict.CLEAN, "proceed"

    @staticmethod
    def _confidence(label_score: float, signal_score: float, verdict: Verdict) -> float:
        if verdict == Verdict.CLEAN:
            return 1.0 - max(label_score, signal_score)
        if verdict == Verdict.UNCERTAIN:
            # Confidence in "uncertain" is lowest right at the midpoint of
            # whichever score triggered it and highest near either edge.
            return 1.0 - abs(0.5 - max(label_score, signal_score)) * 2
        return max(label_score, signal_score)
