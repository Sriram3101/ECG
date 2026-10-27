import pytest

from agents.decision.decision_agent import DecisionAgent, DecisionAgentConfig
from agents.decision.tests.conftest import make_signal_evidence
from shared.contracts import Verdict


class TestDecisionAgentConfig:
    def test_rejects_out_of_range_thresholds(self):
        with pytest.raises(ValueError, match="label_threshold"):
            DecisionAgentConfig(label_threshold=1.5, signal_threshold=0.5)

    def test_rejects_unknown_fusion_strategy(self):
        with pytest.raises(ValueError, match="unknown fusion_strategy"):
            DecisionAgentConfig(label_threshold=0.5, signal_threshold=0.5, fusion_strategy="vibes")

    def test_rejects_negative_uncertain_margin(self):
        with pytest.raises(ValueError, match="uncertain_margin"):
            DecisionAgentConfig(label_threshold=0.5, signal_threshold=0.5, uncertain_margin=-0.1)

    def test_loads_from_shipped_yaml(self):
        # Exercises the actual config/thresholds.yaml shipped with the agent.
        config = DecisionAgentConfig.from_yaml()
        assert 0.0 <= config.label_threshold <= 1.0
        assert 0.0 <= config.signal_threshold <= 1.0

    def test_missing_yaml_raises_helpful_error(self):
        with pytest.raises(FileNotFoundError, match="decision agent config not found"):
            DecisionAgentConfig.from_yaml("/nonexistent/path/thresholds.yaml")


class TestDecisionAgentVerdicts:
    def test_clean_when_both_scores_low(self, agent):
        evidence = make_signal_evidence({})  # all default to 0.05
        decision = agent.run("rec_1", label_score=0.1, signal_evidence=evidence)
        assert decision.verdict == Verdict.CLEAN
        assert decision.recommended_action == "proceed"
        assert decision.confidence > 0.5

    def test_label_poisoned_when_only_label_score_high(self, agent):
        evidence = make_signal_evidence({})
        decision = agent.run("rec_2", label_score=0.9, signal_evidence=evidence)
        assert decision.verdict == Verdict.LABEL_POISONED
        assert decision.recommended_action == "repair"

    def test_signal_poisoned_when_only_signal_score_high(self, agent):
        evidence = make_signal_evidence({
            "reconstruction_error": 0.9,
            "feature_space_distance": 0.9,
            "prediction_fragility": 0.9,
            "gradient_sensitivity": 0.9,
        })
        decision = agent.run("rec_3", label_score=0.1, signal_evidence=evidence)
        assert decision.verdict == Verdict.SIGNAL_POISONED
        assert decision.recommended_action == "repair"

    def test_both_poisoned_when_both_scores_high(self, agent):
        evidence = make_signal_evidence({
            "reconstruction_error": 0.9,
            "feature_space_distance": 0.9,
            "prediction_fragility": 0.9,
            "gradient_sensitivity": 0.9,
        })
        decision = agent.run("rec_4", label_score=0.9, signal_evidence=evidence)
        assert decision.verdict == Verdict.BOTH_POISONED
        assert decision.recommended_action == "quarantine"

    def test_exact_threshold_counts_as_flagged(self, agent):
        # >= threshold, not > threshold
        evidence = make_signal_evidence({})
        decision = agent.run("rec_5", label_score=0.55, signal_evidence=evidence)
        assert decision.verdict == Verdict.LABEL_POISONED

    def test_invalid_label_score_raises(self, agent):
        evidence = make_signal_evidence({})
        with pytest.raises(ValueError, match="label_score must be in"):
            agent.run("rec_6", label_score=1.5, signal_evidence=evidence)

    def test_output_is_deterministic(self, agent):
        evidence = make_signal_evidence({"reconstruction_error": 0.6})
        first = agent.run("rec_7", label_score=0.3, signal_evidence=evidence)
        second = agent.run("rec_7", label_score=0.3, signal_evidence=evidence)
        assert first.verdict == second.verdict
        assert first.signal_score == pytest.approx(second.signal_score)


class TestUncertainBand:
    @pytest.fixture
    def agent_with_uncertain_band(self):
        config = DecisionAgentConfig(label_threshold=0.55, signal_threshold=0.55, uncertain_margin=0.1)
        return DecisionAgent(config)

    def test_score_just_below_threshold_is_uncertain(self, agent_with_uncertain_band):
        evidence = make_signal_evidence({})
        decision = agent_with_uncertain_band.run("rec_8", label_score=0.50, signal_evidence=evidence)
        assert decision.verdict == Verdict.UNCERTAIN
        assert decision.recommended_action == "repair"

    def test_score_well_below_threshold_is_clean_even_with_band_enabled(self, agent_with_uncertain_band):
        evidence = make_signal_evidence({})
        decision = agent_with_uncertain_band.run("rec_9", label_score=0.1, signal_evidence=evidence)
        assert decision.verdict == Verdict.CLEAN

    def test_uncertain_band_disabled_by_default(self, agent):
        # default_config fixture has uncertain_margin=0.0
        evidence = make_signal_evidence({})
        decision = agent.run("rec_10", label_score=0.50, signal_evidence=evidence)
        assert decision.verdict == Verdict.CLEAN


class TestDecisionRecordContract:
    def test_record_carries_original_evidence_for_audit(self, agent):
        evidence = make_signal_evidence({"reconstruction_error": 0.7})
        decision = agent.run("rec_11", label_score=0.2, signal_evidence=evidence)
        assert decision.evidence == evidence

    def test_record_has_agent_metadata(self, agent):
        evidence = make_signal_evidence({})
        decision = agent.run("rec_12", label_score=0.1, signal_evidence=evidence)
        assert decision.agent_name == "decision"
        assert decision.agent_version == DecisionAgent.version
        assert decision.record_id == "rec_12"
