"""End-to-end tests: every agent in the architecture, wired by the orchestrator.

Uses synthetic beats and a small PyTorch DenseNet+BiLSTM+Attention trained
on the fly, so it runs without MIT-BIH or the Keras checkpoint.
"""
import numpy as np
import pytest

from core.ecg_state import make_ecg_state
from evaluation.synthetic_beats import flip_labels, make_beats, pgd_attack
from models.backend import TorchBackend
from models.densenet_bilstm import build_and_train_classifier
from orchestrator.pipeline import ECGDefensePipeline
from shared.contracts import RecoveryAction, Verdict


@pytest.fixture(scope="module")
def setup():
    signals, labels = make_beats(n_per_class=80, seed=0)
    backend = TorchBackend(build_and_train_classifier(signals, labels, epochs=8))
    audit_labels, poisoned = flip_labels(labels, 0.08, seed=1)
    record_ids = [f"beat_{i}" for i in range(len(labels))]
    pipeline = ECGDefensePipeline.from_reference_data(
        backend, clean_signals=signals, audit_signals=signals, audit_labels=audit_labels,
        audit_record_ids=record_ids, shap_permutations=8)
    return pipeline, backend, signals, labels, audit_labels, poisoned, record_ids


def test_clean_beat_flows_through_every_agent(setup):
    pipeline, _, signals, _, audit_labels, poisoned, record_ids = setup
    i = int(np.flatnonzero(~poisoned)[0])
    state = pipeline.run(signals[i], record_ids[i], label=int(audit_labels[i]))

    assert state["validation"]["status"] == "PASS"
    assert state["decision"].verdict == Verdict.CLEAN
    assert state["recovery"] is None
    assert state["diagnosis"]["predicted_class"] is not None
    assert len(state["explainability"]["attention_weights"]) == 360
    assert len(state["explainability"]["shap_values"]) == 12
    assert state["report"]["alert_level"] == "INFO"
    # decision keeps both channels' evidence for audit
    assert {e.channel for e in state["decision"].evidence} == {"label", "signal"}


def test_label_detector_separates_flipped_labels(setup):
    pipeline, _, signals, _, audit_labels, poisoned, record_ids = setup
    scores = np.array([pipeline.label_agent.run(r, s, int(y)).score
                       for r, s, y in zip(record_ids, signals, audit_labels)])
    flagged = scores >= pipeline.decision_agent.config.label_threshold
    assert flagged[poisoned].mean() >= 0.7
    assert flagged[~poisoned].mean() <= 0.05


def test_label_poisoned_beat_is_relabeled_before_diagnosis(setup):
    pipeline, _, signals, labels, audit_labels, poisoned, record_ids = setup
    # The flipped beat the label detector is most sure about — this test is
    # about routing through Recovery; detection rates are tested above.
    flipped = np.flatnonzero(poisoned)
    i = int(max(flipped, key=lambda j: pipeline.label_agent.run(record_ids[j], signals[j], int(audit_labels[j])).score))
    state = pipeline.run(signals[i], record_ids[i], label=int(audit_labels[i]))

    assert state["decision"].verdict == Verdict.LABEL_POISONED
    assert state["recovery"].action in (RecoveryAction.RELABEL, RecoveryAction.QUARANTINE)
    if state["recovery"].action == RecoveryAction.RELABEL:
        assert state["active_label"] == labels[i]
        assert state["report"]["alert_level"] == "WARNING"
        assert state["diagnosis"]["predicted_class"] is not None
    else:
        assert state["diagnosis"] is None
        assert state["report"]["alert_level"] == "CRITICAL"


def test_adversarial_evidence_reaches_decision_and_routes_to_recovery(setup):
    pipeline, backend, signals, labels, _, poisoned, _ = setup
    for i in np.flatnonzero(~poisoned)[:10]:
        attacked = pgd_attack(backend, signals[i], int(labels[i]), epsilon=0.1, step_size=0.02)
        state = pipeline.run(attacked, f"incoming_{i}", label=int(labels[i]))
        if state["validation"]["status"] == "REJECT":
            assert state["decision"] is None  # validation is the first line of defense
            continue
        signal_checks = {e.check_name: e for e in state["decision"].evidence if e.channel == "signal"}
        assert set(signal_checks) >= {"reconstruction_error", "feature_space_distance",
                                      "prediction_fragility", "gradient_sensitivity"}
        assert signal_checks["reconstruction_error"].score >= 0.9
        assert (state["recovery"] is None) == (state["decision"].verdict == Verdict.CLEAN)
    # How many of these end up CLEAN depends on the Decision agent's fusion
    # policy and thresholds, not on wiring — see README "Known limitations".


def test_invalid_beat_skips_detection_and_raises_critical_alert(setup):
    pipeline, *_ = setup
    state = pipeline.run(np.zeros(360), "flatline")

    assert state["validation"]["status"] == "REJECT"
    assert state["decision"] is None and state["diagnosis"] is None
    assert state["report"]["alert_level"] == "CRITICAL"


def test_langgraph_graph_matches_plain_runner(setup):
    pipeline, _, signals, _, audit_labels, poisoned, record_ids = setup
    graph = pipeline.build_graph()
    for i in (int(np.flatnonzero(~poisoned)[1]), int(np.flatnonzero(poisoned)[1])):
        via_graph = graph.invoke(make_ecg_state(record_ids[i], signals[i], label=int(audit_labels[i])))
        plain = pipeline.run(signals[i], record_ids[i], label=int(audit_labels[i]))
        assert via_graph["decision"].verdict == plain["decision"].verdict
        assert via_graph["report"]["alert_level"] == plain["report"]["alert_level"]
