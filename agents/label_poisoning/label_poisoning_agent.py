"""
agents/label_poisoning/label_poisoning_agent.py

Label poisoning detection: is the *annotation* on this beat trustworthy?

Workflow
--------
1. `fit(signals, labels, record_ids)` audits a labelled dataset (the one
   the beats being checked belong to — label poisoning is a property of a
   dataset, not of a lone beat): embeds every beat with the trained
   classifier, then runs the three detectors over the whole set
   (learning dynamics / confident learning / latent+attention) and fits
   the per-class calibration.
2. `run(record_id, signal, claimed_label)` returns a DetectionReport with
   one Evidence per check plus the fused `score` (= label_score for the
   Decision agent). Records that were part of the audited dataset reuse
   their audit statistics (true AUM, true out-of-fold probabilities);
   unseen records fall back to final-epoch AUM margin and fold-ensemble
   probabilities.

The OOF probability vector is returned as `artifacts["oof_probabilities"]`
for the Recovery agent's relabel strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.label_poisoning.confident_learning import ConfidentLearningScorer, label_disagreement
from agents.label_poisoning.latent_attention import LatentAttentionScorer
from agents.label_poisoning.learning_dynamics import AUMScorer
from agents.label_poisoning.score_fusion import DEFAULT_LABEL_CHECK_WEIGHTS, PerClassScoreFusion
from models.backend import ClassifierBackend, as_batch
from shared.contracts import DetectionReport, Evidence

logger = logging.getLogger(__name__)


@dataclass
class LabelPoisoningConfig:
    aum_epochs: int = 15
    cl_folds: int = 5
    k_neighbors: int = 10
    z_offset: float = 3.0
    check_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LABEL_CHECK_WEIGHTS))
    seed: int = 0


@dataclass
class _AuditEntry:
    label: int
    statistics: dict[str, float]
    oof_probabilities: np.ndarray


class LabelPoisoningAgent:
    name = "label_poisoning"
    version = "0.1.0"

    def __init__(self, backend: ClassifierBackend, config: Optional[LabelPoisoningConfig] = None):
        self.backend = backend
        self.config = config or LabelPoisoningConfig()
        num_classes = backend.num_classes
        self._aum = AUMScorer(num_classes, epochs=self.config.aum_epochs, seed=self.config.seed)
        self._cl = ConfidentLearningScorer(num_classes, n_folds=self.config.cl_folds, seed=self.config.seed)
        self._latent = LatentAttentionScorer(num_classes, k_neighbors=self.config.k_neighbors)
        self._fusion = PerClassScoreFusion(self.config.check_weights, z_offset=self.config.z_offset)
        # Calibration for records outside the audited dataset, whose
        # statistics come from a different path (final-epoch margin, fold
        # ensemble, k-NN against other patients) and so a different scale.
        self._fresh_fusion: PerClassScoreFusion | None = None
        self._audit: dict[str, _AuditEntry] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._audit)

    def fit(self, signals: np.ndarray, labels: np.ndarray, record_ids: list[str],
            calibration_signals: Optional[np.ndarray] = None,
            calibration_labels: Optional[np.ndarray] = None) -> "LabelPoisoningAgent":
        """`calibration_*`: labelled beats from patients *not* in the audited
        set (e.g. the validation split), used to calibrate scores for new
        records passed to `run`. Without them, new records are scored on the
        audit calibration, which over-flags unseen patients."""
        x = as_batch(signals)
        labels = np.asarray(labels, dtype=int)
        if not (len(x) == len(labels) == len(record_ids)):
            raise ValueError("signals, labels and record_ids must have the same length")

        embeddings = self.backend.embed(x)
        probs = self.backend.predict_proba(x)

        aum = self._aum.fit(embeddings, labels)
        oof = self._cl.fit(embeddings, labels)
        self._latent.fit(embeddings, labels)
        latent = self._latent.statistics(embeddings, labels, probs, exclude_self=True)

        statistics = {"negative_aum": -aum, "oof_disagreement": label_disagreement(oof, labels), **latent}
        self._fusion.fit(statistics, labels)

        self._audit = {
            rid: _AuditEntry(int(labels[i]), {k: float(v[i]) for k, v in statistics.items()}, oof[i])
            for i, rid in enumerate(record_ids)
        }
        if calibration_signals is not None:
            cal_labels = np.asarray(calibration_labels, dtype=int)
            cal_stats, _ = self._fresh_statistics_batch(as_batch(calibration_signals), cal_labels)
            self._fresh_fusion = PerClassScoreFusion(self.config.check_weights, z_offset=self.config.z_offset)
            self._fresh_fusion.fit(cal_stats, cal_labels)
        logger.info("label poisoning audit fitted on %d records", len(record_ids))
        return self

    def suspicious_records(self, threshold: float) -> list[dict]:
        """Dataset-level audit: every audited record whose fused label score
        is >= threshold, most suspicious first, with the out-of-fold consensus
        label. This is the "remove suspicious training samples" recovery
        option — drop or review these before retraining."""
        if not self.is_fitted:
            raise RuntimeError("LabelPoisoningAgent.suspicious_records() called before fit()")
        flagged = []
        for record_id, entry in self._audit.items():
            score = self._fusion.fuse(self._fusion.check_scores(entry.statistics, entry.label))
            if score >= threshold:
                flagged.append({"record_id": record_id, "label_score": score, "claimed_label": entry.label,
                                "consensus_label": int(np.argmax(entry.oof_probabilities)),
                                "consensus_confidence": float(np.max(entry.oof_probabilities))})
        return sorted(flagged, key=lambda r: -r["label_score"])

    def _fresh_statistics_batch(self, x: np.ndarray, labels: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        embedding = self.backend.embed(x)
        probs = self.backend.predict_proba(x)
        oof = self._cl.predict_proba(embedding)
        latent = self._latent.statistics(embedding, labels, probs)
        statistics = {
            "negative_aum": -self._aum.margin(embedding, labels),
            "oof_disagreement": label_disagreement(oof, labels),
            **latent,
        }
        return statistics, oof

    def _fresh_statistics(self, signal: np.ndarray, label: int) -> tuple[dict[str, float], np.ndarray]:
        statistics, oof = self._fresh_statistics_batch(as_batch(signal), np.array([label]))
        return {k: float(v[0]) for k, v in statistics.items()}, oof[0]

    def run(self, record_id: str, signal: np.ndarray, claimed_label: Optional[int]) -> DetectionReport:
        if not self.is_fitted:
            raise RuntimeError("LabelPoisoningAgent.run() called before fit()")

        if claimed_label is None:
            # Nothing to audit: an unlabeled beat can't carry a poisoned label.
            return DetectionReport(record_id=record_id, agent_name=self.name, agent_version=self.version,
                                   channel="label", score=0.0, evidence=[],
                                   artifacts={"note": "no claimed label — label checks skipped"})

        entry = self._audit.get(record_id)
        in_audit = entry is not None and entry.label == int(claimed_label)
        if in_audit:
            statistics, oof = entry.statistics, entry.oof_probabilities
        else:
            statistics, oof = self._fresh_statistics(signal, int(claimed_label))

        fusion = self._fusion if in_audit or self._fresh_fusion is None else self._fresh_fusion
        check_scores = fusion.check_scores(statistics, claimed_label)
        label_score = fusion.fuse(check_scores)

        evidence = [
            Evidence(check_name="learning_dynamics", channel="label", score=check_scores["learning_dynamics"],
                     raw_value=-statistics["negative_aum"],
                     metadata={"aum": -statistics["negative_aum"], "from_training_trajectory": in_audit}),
            Evidence(check_name="confident_learning", channel="label", score=check_scores["confident_learning"],
                     raw_value=statistics["oof_disagreement"],
                     metadata={"oof_probabilities": oof.tolist(), "out_of_fold": in_audit}),
            Evidence(check_name="latent_attention", channel="label", score=check_scores["latent_attention"],
                     metadata={k: statistics[k] for k in ("knn_disagreement", "mahalanobis_margin",
                                                          "label_cross_entropy", "predictive_entropy")}),
        ]
        return DetectionReport(record_id=record_id, agent_name=self.name, agent_version=self.version,
                               channel="label", score=label_score, evidence=evidence,
                               artifacts={"oof_probabilities": oof, "check_scores": check_scores})
