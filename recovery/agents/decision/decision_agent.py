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
    ) -> DecisionRecord:
        if not (0.0 <= label_score <= 1.0):
            raise ValueError(f"label_score must be in [0,1], got {label_score}")

        signal_score = self._fuse(signal_evidence, self.config.signal_check_weights)

        verdict, action = self._decide(label_score, signal_score)
        confidence = self._confidence(label_score, signal_score, verdict)

        logger.info(
            "decision record_id=%s label_score=%.3f signal_score=%.3f "
            "verdict=%s action=%s confidence=%.3f",
            record_id, label_score, signal_score, verdict.value, action, confidence,
        )

        return DecisionRecord(
            record_id=record_id,
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            label_score=label_score,
            signal_score=signal_score,
            evidence=signal_evidence,
            recommended_action=action,
        )

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
