import pytest

from agents.decision.decision_agent import DecisionAgent, DecisionAgentConfig
from shared.contracts import Evidence

SIGNAL_CHECK_NAMES = [
    "reconstruction_error",
    "feature_space_distance",
    "prediction_fragility",
    "gradient_sensitivity",
]


def make_signal_evidence(scores: dict[str, float]) -> list[Evidence]:
    """scores: check_name -> score. Missing checks default to 0.05
    (i.e. "looks clean") so tests can override just the check they care
    about."""
    return [
        Evidence(check_name=name, channel="signal", score=scores.get(name, 0.05))
        for name in SIGNAL_CHECK_NAMES
    ]


@pytest.fixture
def default_config() -> DecisionAgentConfig:
    return DecisionAgentConfig(label_threshold=0.55, signal_threshold=0.55)


@pytest.fixture
def agent(default_config) -> DecisionAgent:
    return DecisionAgent(default_config)
