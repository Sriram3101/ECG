import numpy as np
import pytest

from agents.recovery.recovery_agent import RecoveryAgent, RecoveryAgentConfig
from shared.contracts import DecisionRecord, ECGRecord, Evidence, Verdict

SIGNAL_LENGTH = 360


def make_signal(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, size=SIGNAL_LENGTH).tolist()


def make_record(record_id: str = "rec_1", label: int | None = 2, signal: list[float] | None = None) -> ECGRecord:
    return ECGRecord(record_id=record_id, signal=signal or make_signal(), label=label)


_DUMMY_SIGNAL_EVIDENCE = [
    Evidence(check_name=name, channel="signal", score=0.5)
    for name in [
        "reconstruction_error",
        "feature_space_distance",
        "prediction_fragility",
        "gradient_sensitivity",
    ]
]


def make_decision(
    record_id: str = "rec_1",
    verdict: Verdict = Verdict.LABEL_POISONED,
    label_score: float = 0.9,
    signal_score: float = 0.1,
    evidence: list[Evidence] | None = None,
) -> DecisionRecord:
    action = "proceed" if verdict == Verdict.CLEAN else "repair"
    return DecisionRecord(
        record_id=record_id,
        agent_name="decision",
        agent_version="0.1.0",
        verdict=verdict,
        confidence=max(label_score, signal_score),
        label_score=label_score,
        signal_score=signal_score,
        evidence=evidence if evidence is not None else _DUMMY_SIGNAL_EVIDENCE,
        recommended_action=action,
    )


@pytest.fixture
def default_config() -> RecoveryAgentConfig:
    return RecoveryAgentConfig(min_label_confidence=0.7, min_signal_confidence=0.6)


@pytest.fixture
def agent(default_config) -> RecoveryAgent:
    return RecoveryAgent(default_config)
