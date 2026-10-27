import numpy as np
import pytest

from agents.recovery.recovery_agent import RecoveryAgent, RecoveryAgentConfig
from agents.recovery.tests.conftest import make_decision, make_record, make_signal
from shared.contracts import Evidence, RecoveryAction, Verdict


class TestRecoveryAgentConfig:
    def test_rejects_out_of_range_label_confidence(self):
        with pytest.raises(ValueError, match="min_label_confidence"):
            RecoveryAgentConfig(min_label_confidence=1.5, min_signal_confidence=0.1)

    def test_rejects_out_of_range_signal_confidence(self):
        with pytest.raises(ValueError, match="min_signal_confidence"):
            RecoveryAgentConfig(min_label_confidence=0.5, min_signal_confidence=-0.1)

    def test_loads_from_shipped_yaml(self):
        config = RecoveryAgentConfig.from_yaml()
        assert 0.0 <= config.min_label_confidence <= 1.0
        assert 0.0 <= config.min_signal_confidence <= 1.0

    def test_missing_yaml_raises_helpful_error(self):
        with pytest.raises(FileNotFoundError, match="recovery agent config not found"):
            RecoveryAgentConfig.from_yaml("/nonexistent/repair_thresholds.yaml")


class TestDispatch:
    def test_label_poisoned_dispatches_to_label_repair(self, agent):
        decision = make_decision(verdict=Verdict.LABEL_POISONED)
        record = make_record(label=3)
        oof = [0.02, 0.90, 0.05, 0.02, 0.01]

        result = agent.run(decision, record, oof_probabilities=oof)

        assert result.action == RecoveryAction.RELABEL
        assert result.repaired_record.label == 1

    def test_signal_poisoned_dispatches_to_signal_repair(self, agent):
        original = make_signal(seed=2)
        decision = make_decision(verdict=Verdict.SIGNAL_POISONED)
        record = make_record(signal=original)
        reconstruction = (np.asarray(original) + 0.01).tolist()

        result = agent.run(
            decision, record,
            autoencoder_reconstruction=reconstruction,
            reconstruction_confidence=0.9,
        )

        assert result.action == RecoveryAction.RECONSTRUCT_SIGNAL

    def test_both_poisoned_always_quarantines(self, agent):
        decision = make_decision(verdict=Verdict.BOTH_POISONED)
        record = make_record()

        # Even with valid artifacts available, BOTH_POISONED should not
        # attempt an automatic repair.
        result = agent.run(
            decision, record,
            oof_probabilities=[0.02, 0.90, 0.05, 0.02, 0.01],
            autoencoder_reconstruction=make_signal(seed=3),
            reconstruction_confidence=0.9,
        )

        assert result.action == RecoveryAction.QUARANTINE
        assert "compounding risk" in result.notes

    def test_uncertain_quarantines(self, agent):
        decision = make_decision(verdict=Verdict.UNCERTAIN, label_score=0.5, signal_score=0.5)
        record = make_record()

        result = agent.run(decision, record)

        assert result.action == RecoveryAction.QUARANTINE
        assert "uncertain verdict" in result.notes

    def test_clean_verdict_raises(self, agent):
        decision = make_decision(verdict=Verdict.CLEAN, label_score=0.05, signal_score=0.05)
        record = make_record()

        with pytest.raises(ValueError, match="should not be called"):
            agent.run(decision, record)

    def test_mismatched_record_ids_raises(self, agent):
        decision = make_decision(record_id="rec_A", verdict=Verdict.LABEL_POISONED)
        record = make_record(record_id="rec_B")

        with pytest.raises(ValueError, match="does not match"):
            agent.run(decision, record, oof_probabilities=[0.9, 0.1])


class TestArtifactRequirements:
    def test_missing_oof_probabilities_raises_helpful_keyerror(self, agent):
        decision = make_decision(verdict=Verdict.LABEL_POISONED)
        record = make_record()

        with pytest.raises(KeyError, match="oof_probabilities"):
            agent.run(decision, record)

    def test_missing_reconstruction_raises_helpful_keyerror(self, agent):
        decision = make_decision(verdict=Verdict.SIGNAL_POISONED)
        record = make_record()

        with pytest.raises(KeyError, match="autoencoder_reconstruction"):
            agent.run(decision, record)

    def test_falls_back_to_evidence_metadata_and_confidence(self, agent):
        reconstruction = make_signal(seed=4)
        evidence = [
            Evidence(
                check_name="reconstruction_error",
                channel="signal",
                score=0.9,
                confidence=0.85,  # read as reconstruction_confidence
                metadata={"reconstruction": reconstruction},
            ),
            Evidence(check_name="feature_space_distance", channel="signal", score=0.5),
            Evidence(check_name="prediction_fragility", channel="signal", score=0.5),
            Evidence(check_name="gradient_sensitivity", channel="signal", score=0.5),
        ]
        decision = make_decision(verdict=Verdict.SIGNAL_POISONED, evidence=evidence)
        record = make_record(signal=make_signal(seed=99))  # unrelated original signal — irrelevant to trust now

        # No explicit kwargs — both must come from evidence
        result = agent.run(decision, record)

        assert result.action == RecoveryAction.RECONSTRUCT_SIGNAL
        assert result.repair_confidence == pytest.approx(0.85)

    def test_low_evidence_confidence_falls_back_to_quarantine(self, agent):
        reconstruction = make_signal(seed=4)
        evidence = [
            Evidence(
                check_name="reconstruction_error", channel="signal", score=0.9,
                confidence=0.2,  # below default min_signal_confidence (0.6)
                metadata={"reconstruction": reconstruction},
            ),
            Evidence(check_name="feature_space_distance", channel="signal", score=0.5),
            Evidence(check_name="prediction_fragility", channel="signal", score=0.5),
            Evidence(check_name="gradient_sensitivity", channel="signal", score=0.5),
        ]
        decision = make_decision(verdict=Verdict.SIGNAL_POISONED, evidence=evidence)
        record = make_record()

        result = agent.run(decision, record)

        assert result.action == RecoveryAction.QUARANTINE

    def test_explicit_kwargs_take_priority_over_evidence(self, agent):
        metadata_reconstruction = [0.0] * 360
        explicit_reconstruction = make_signal(seed=5)
        record = make_record()
        evidence = [
            Evidence(
                check_name="reconstruction_error", channel="signal", score=0.9,
                confidence=0.1,  # would quarantine if used
                metadata={"reconstruction": metadata_reconstruction},
            ),
            Evidence(check_name="feature_space_distance", channel="signal", score=0.5),
            Evidence(check_name="prediction_fragility", channel="signal", score=0.5),
            Evidence(check_name="gradient_sensitivity", channel="signal", score=0.5),
        ]
        decision = make_decision(verdict=Verdict.SIGNAL_POISONED, evidence=evidence)

        result = agent.run(
            decision, record,
            autoencoder_reconstruction=explicit_reconstruction,
            reconstruction_confidence=0.9,  # explicit kwarg wins over evidence.confidence=0.1
        )

        assert result.action == RecoveryAction.RECONSTRUCT_SIGNAL


class TestExtractArtifacts:
    def test_extracts_reconstruction_and_confidence_from_evidence(self):
        reconstruction = [0.1] * 360
        evidence = [
            Evidence(
                check_name="reconstruction_error", channel="signal", score=0.9,
                confidence=0.77,
                metadata={"reconstruction": reconstruction},
            )
        ]
        decision = make_decision(verdict=Verdict.SIGNAL_POISONED, evidence=evidence)

        artifacts = RecoveryAgent.extract_artifacts(decision)

        assert artifacts["autoencoder_reconstruction"] == reconstruction
        assert artifacts["reconstruction_confidence"] == pytest.approx(0.77)

    def test_returns_empty_dict_when_nothing_relevant_present(self):
        # dummy evidence has no "reconstruction_error" check name at all
        decision = make_decision(
            verdict=Verdict.LABEL_POISONED,
            evidence=[Evidence(check_name="some_other_check", channel="signal", score=0.1)],
        )
        assert RecoveryAgent.extract_artifacts(decision) == {}
