r"""
orchestrator/pipeline.py

Wires every agent together in the order of the architecture diagram:

                         Validation agent
                (format check, signal sanity, triage)
                 |  REJECT ------------------------------------+
                 v                                             |
     Label poisoning detection    Signal poisoning detection   |
     (AUM, confident learning,    (reconstruction, Mahalanobis,|
      latent+attention, fusion)    fragility, gradient)        |
                 \                       /                     |
                  v                     v                      |
                        Decision agent                         |
                 proceed |          | repair / quarantine      |
                         |          v                          |
                         |    Recovery agent                   |
                         |     repaired | quarantined/rejected |
                         v          v   +----------------------+
                        Diagnosis agent                        |
                              v                                |
                      Explainability agent                     |
                              v                                v
                         Alert & reporting  <------------------+

Each node takes the ECGState and returns only the keys it writes, so the
same node functions run in the plain-Python runner (`run`) and in the
LangGraph graph (`build_graph`).
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from agents.decision.decision_agent import DecisionAgent
from agents.diagnosis.diagnosis_agent import run_diagnosis
from agents.explainability.explainability_agent import run_explainability
from agents.label_poisoning.label_poisoning_agent import LabelPoisoningAgent, LabelPoisoningConfig
from agents.recovery.recovery_agent import RecoveryAgent
from agents.recovery.strategies.quarantine import quarantine, reject
from agents.reporting.reporting_agent import run_reporting
from agents.signal_poisoning.noise.noise_agent import NoiseFeatureAgent
from agents.signal_poisoning.noise.noise_features import MorphologyTemplateBank
from agents.signal_poisoning.signal_poisoning_agent import SignalPoisoningAgent, SignalPoisoningConfig
from agents.validation.validation_agent import ValidationConfig, validate_beat
from core.ecg_state import ECGState, make_ecg_state
from models.backend import ClassifierBackend, as_batch
from shared.contracts import ECGRecord, RecoveryAction, RecoveryResult, Verdict

logger = logging.getLogger(__name__)

VALIDATION, LABEL, SIGNAL, DECISION, RECOVERY, DIAGNOSIS, EXPLAIN, REPORT = (
    "validation", "label_poisoning", "signal_poisoning", "decision",
    "recovery", "diagnosis", "explainability", "reporting",
)


class ECGDefensePipeline:
    def __init__(
        self,
        backend: ClassifierBackend,
        label_agent: LabelPoisoningAgent,
        signal_agent: SignalPoisoningAgent,
        decision_agent: Optional[DecisionAgent] = None,
        recovery_agent: Optional[RecoveryAgent] = None,
        explain_background: Optional[np.ndarray] = None,
        shap_permutations: int = 64,
        validation_config: Optional[ValidationConfig] = None,
    ):
        """`validation_config` defaults to the preset for per-beat z-scored
        beats (the team's ECG_Splits format); pass `ValidationConfig()` for
        raw-amplitude beats."""
        self.backend = backend
        self.label_agent = label_agent
        self.signal_agent = signal_agent
        self.decision_agent = decision_agent or DecisionAgent()
        self.recovery_agent = recovery_agent or RecoveryAgent()
        self.explain_background = explain_background
        self.shap_permutations = shap_permutations
        self.validation_config = validation_config or ValidationConfig.zscored_beats()

    @classmethod
    def from_reference_data(
        cls,
        backend: ClassifierBackend,
        clean_signals: np.ndarray,
        audit_signals: np.ndarray,
        audit_labels: np.ndarray,
        audit_record_ids: Sequence[str],
        calibration_signals: Optional[np.ndarray] = None,
        calibration_labels: Optional[np.ndarray] = None,
        noise_agent="default",
        label_config: Optional[LabelPoisoningConfig] = None,
        signal_config: Optional[SignalPoisoningConfig] = None,
        **kwargs,
    ) -> "ECGDefensePipeline":
        """Fit both detection agents and build the pipeline.

        clean_signals: beats known to be clean (the classifier's clean
            training split) — reference for signal poisoning detection and
            the SHAP background.
        audit_*: the labelled dataset whose annotations are being checked
            for label poisoning; beats later passed to `run` with one of
            these record ids reuse their audit statistics.
        calibration_*: clean, correctly labelled beats from patients in
            neither set above (e.g. the validation split). Both detection
            agents calibrate their scores for new beats on these; without
            them, beats from unseen patients are heavily over-flagged.
        noise_agent: spectral noise/quality agent run alongside the four
            signal checks. "default" builds the threshold-based
            NoiseFeatureAgent with morphology templates from the clean beats;
            pass a NoiseClassifierAgent to use the trained noise classifier,
            or None to skip it.
        """
        clean = as_batch(clean_signals)
        if isinstance(noise_agent, str) and noise_agent == "default":
            noise_agent = NoiseFeatureAgent(MorphologyTemplateBank.fit(clean[:256]))
        signal_agent = SignalPoisoningAgent(backend, signal_config, noise_agent=noise_agent).fit(
            clean, calibration_signals)
        label_agent = LabelPoisoningAgent(backend, label_config).fit(
            audit_signals, audit_labels, list(audit_record_ids), calibration_signals, calibration_labels)
        return cls(backend, label_agent, signal_agent, explain_background=clean.mean(axis=0), **kwargs)

    # ------------------------------------------------------------------ nodes

    def validation_node(self, state: ECGState) -> dict:
        result = validate_beat(state["raw_signal"], state.get("metadata"), self.validation_config)
        active = None if result["signal"] is None else np.asarray(result["signal"], dtype=np.float64).flatten()
        return {"validation": result, "active_signal": active}

    def label_poisoning_node(self, state: ECGState) -> dict:
        return {"label_poisoning": self.label_agent.run(state["beat_id"], state["active_signal"], state.get("label"))}

    def signal_poisoning_node(self, state: ECGState) -> dict:
        return {"signal_poisoning": self.signal_agent.run(state["beat_id"], state["active_signal"])}

    def decision_node(self, state: ECGState) -> dict:
        label_report = state["label_poisoning"]
        signal_report = state["signal_poisoning"]
        decision = self.decision_agent.run(
            record_id=state["beat_id"],
            label_score=label_report.score,
            signal_evidence=signal_report.evidence,
            label_evidence=label_report.evidence,
        )
        return {"decision": decision}

    def recovery_node(self, state: ECGState) -> dict:
        record = ECGRecord(record_id=state["beat_id"], signal=state["active_signal"].tolist(), label=state.get("label"))
        label_artifacts = state["label_poisoning"].artifacts
        signal_artifacts = state["signal_poisoning"].artifacts
        result: RecoveryResult = self.recovery_agent.run(
            state["decision"], record,
            oof_probabilities=label_artifacts.get("oof_probabilities"),
            autoencoder_reconstruction=signal_artifacts.get("autoencoder_reconstruction"),
            reconstruction_confidence=signal_artifacts.get("reconstruction_confidence"),
        )

        if result.repaired_record is None:
            return {"recovery": result, "active_signal": None, "active_label": None}

        # A repaired signal is a new signal: it has to clear the same
        # validation checks as any input before Diagnosis may use it.
        repaired_signal = np.asarray(result.repaired_record.signal, dtype=np.float64)
        revalidation = validate_beat(repaired_signal, config=self.validation_config)
        if revalidation["status"] == "REJECT":
            rejected = reject(state["beat_id"], f"repaired signal failed re-validation: {revalidation['reason']}")
            return {"recovery": rejected, "active_signal": None, "active_label": None}
        active_signal = np.asarray(revalidation["signal"], dtype=np.float64).flatten()

        # A repaired *signal* must also stop looking poisoned; if signal
        # poisoning detection still flags it, recovery wasn't possible and
        # the record is referred for expert review.
        if result.action in (RecoveryAction.RECONSTRUCT_SIGNAL, RecoveryAction.DENOISE_SIGNAL):
            recheck = self.signal_agent.run(state["beat_id"], active_signal)
            recheck_decision = self.decision_agent.run(state["beat_id"], 0.0, recheck.evidence)
            if recheck_decision.verdict != Verdict.CLEAN:
                referred = quarantine(state["beat_id"], f"{result.action.value} did not remove the attack "
                                      f"(signal score after repair {recheck_decision.signal_score:.2f}) — "
                                      "referred for expert review")
                return {"recovery": referred, "active_signal": None, "active_label": None}
            result = result.model_copy(update={
                "notes": f"{result.notes}; signal score after repair {recheck_decision.signal_score:.2f}"})

        return {
            "recovery": result,
            "active_signal": active_signal,
            "active_label": result.repaired_record.label,
        }

    def diagnosis_node(self, state: ECGState) -> dict:
        return {"diagnosis": run_diagnosis(dict(state), self.backend)["diagnosis"]}

    def explainability_node(self, state: ECGState) -> dict:
        updated = run_explainability(dict(state), self.backend, self.explain_background, self.shap_permutations)
        return {"explainability": updated["explainability"]}

    def reporting_node(self, state: ECGState) -> dict:
        return {"report": run_reporting(dict(state))["report"]}

    def audit_dataset(self, threshold: Optional[float] = None) -> list[dict]:
        """Records in the audited training set whose labels look poisoned —
        candidates to remove or relabel before retraining. Defaults to the
        Decision agent's label threshold."""
        return self.label_agent.suspicious_records(
            self.decision_agent.config.label_threshold if threshold is None else threshold)

    # ---------------------------------------------------------------- routing

    @staticmethod
    def route_after_validation(state: ECGState):
        if state["validation"]["status"] == "REJECT":
            return REPORT
        return [LABEL, SIGNAL]

    @staticmethod
    def route_after_decision(state: ECGState) -> str:
        return DIAGNOSIS if state["decision"].recommended_action == "proceed" else RECOVERY

    @staticmethod
    def route_after_recovery(state: ECGState) -> str:
        return DIAGNOSIS if state.get("active_signal") is not None else REPORT

    # ---------------------------------------------------------------- runners

    def _nodes(self) -> dict:
        return {
            VALIDATION: self.validation_node,
            LABEL: self.label_poisoning_node,
            SIGNAL: self.signal_poisoning_node,
            DECISION: self.decision_node,
            RECOVERY: self.recovery_node,
            DIAGNOSIS: self.diagnosis_node,
            EXPLAIN: self.explainability_node,
            REPORT: self.reporting_node,
        }

    def run_state(self, state: ECGState) -> ECGState:
        """Plain-Python execution of the graph (no LangGraph dependency)."""
        nodes = self._nodes()
        state = dict(state)

        def step(name: str) -> None:
            state.update(nodes[name](state))
            logger.debug("beat %s: %s done", state.get("beat_id"), name)

        step(VALIDATION)
        if self.route_after_validation(state) == REPORT:
            step(REPORT)
            return state
        step(LABEL)
        step(SIGNAL)
        step(DECISION)
        if self.route_after_decision(state) == RECOVERY:
            step(RECOVERY)
            if self.route_after_recovery(state) == REPORT:
                step(REPORT)
                return state
        step(DIAGNOSIS)
        step(EXPLAIN)
        step(REPORT)
        return state

    def run(self, signal: np.ndarray, beat_id: str, label: Optional[int] = None,
            metadata: Optional[dict] = None) -> ECGState:
        return self.run_state(make_ecg_state(beat_id, np.asarray(signal), metadata, label))

    def run_batch(self, signals: np.ndarray, beat_ids: Sequence[str],
                  labels: Optional[Sequence[Optional[int]]] = None) -> list[ECGState]:
        labels = labels if labels is not None else [None] * len(beat_ids)
        return [self.run(s, b, None if y is None else int(y)) for s, b, y in zip(signals, beat_ids, labels)]

    def build_graph(self):
        """The same pipeline as a compiled LangGraph graph (label and signal
        poisoning detection run as parallel branches that join at Decision).
        Invoke with `graph.invoke(make_ecg_state(beat_id, signal, label=...))`."""
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(ECGState)
        for name, fn in self._nodes().items():
            builder.add_node(name, fn)
        builder.add_edge(START, VALIDATION)
        builder.add_conditional_edges(VALIDATION, self.route_after_validation, [LABEL, SIGNAL, REPORT])
        builder.add_edge([LABEL, SIGNAL], DECISION)
        builder.add_conditional_edges(DECISION, self.route_after_decision, [DIAGNOSIS, RECOVERY])
        builder.add_conditional_edges(RECOVERY, self.route_after_recovery, [DIAGNOSIS, REPORT])
        builder.add_edge(DIAGNOSIS, EXPLAIN)
        builder.add_edge(EXPLAIN, REPORT)
        builder.add_edge(REPORT, END)
        return builder.compile()
