"""Noise-detection LangGraph node using transparent, calibratable features."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import ECGState
from .noise_classifier import TrainedNoiseClassifier
from .noise_features import MorphologyTemplateBank, extract_noise_features


@dataclass
class NoiseThresholds:
    baseline_power_ratio: float = 0.10
    powerline_50hz_ratio: float = 0.03
    emg_power_ratio: float = 0.35
    wavelet_mad_sigma: float = 0.12
    template_correlation: float = 0.92


class NoiseFeatureAgent:
    """Replace thresholds with Nihar's trained classifier when it is ready."""
    def __init__(self, templates: MorphologyTemplateBank | None = None, thresholds: NoiseThresholds | None = None):
        self.templates = templates
        self.thresholds = thresholds or NoiseThresholds()

    def __call__(self, state: ECGState) -> dict:
        signal = state["validation_result"]["signal"]
        features = extract_noise_features(signal, self.templates)
        t = self.thresholds
        components = {
            "baseline_wander": features["baseline_power_ratio"] / t.baseline_power_ratio,
            "powerline": features["powerline_50hz_ratio"] / t.powerline_50hz_ratio,
            "em": features["emg_power_ratio"] / t.emg_power_ratio,
            "wavelet": features["wavelet_mad_sigma"] / t.wavelet_mad_sigma,
            "morphology": max(0.0, (t.template_correlation - features["template_correlation"]) / t.template_correlation),
        }
        noise_type, raw = max(components.items(), key=lambda item: item[1])
        # Monotonic bounded score; calibrate these thresholds on held-out corruptions.
        score = float(1.0 - math.exp(-max(raw - 1.0, 0.0)))
        return {"noise_evidence": {"score": score, "predicted_type": noise_type,
                                    "severity": float(raw), "features": features}}


class NoiseClassifierAgent:
    """Nihar's trained replacement for NoiseFeatureAgent.

    Same call signature, same top-level `noise_evidence` output contract
    (score/predicted_type/severity/features) -- swap this in for
    `NoiseFeatureAgent` in the graph and nothing downstream (Neha S's
    scaffold, the decision agent) needs to change.

    `score`/`predicted_type` answer "is this the poisoning attack we
    trained on" -- that's the only thing that should ever count as
    `is_attack` for the Decision Agent. They deliberately do NOT get
    merged with ordinary residual sensor noise (that conflation was the
    original bug this reframe fixed).

    `features` additionally carries the threshold agent's own
    `residual_noise_type` / `residual_noise_ratio` -- these say "does this
    beat still look noisy at all, regardless of whether it's an attack."
    That's a real, separate question worth answering (e.g. some sensor
    noise slips past preprocessing), and the old `NoiseFeatureAgent`
    already computed exactly this -- rather than retiring it, it keeps
    running here as an always-on secondary signal a Recovery Agent can act
    on independently of the attack decision (e.g. "clean this up" even
    when `predicted_type == 'none'`).

    `severity` has no ground truth at inference time (that's only in the
    eval-only `*.truth.json`), so it's still derived from the raw feature
    strength the same way the threshold agent computes it -- this keeps the
    field's meaning consistent across both agents for anyone comparing them.
    """
    def __init__(self, classifier: TrainedNoiseClassifier, templates: MorphologyTemplateBank | None = None,
                 thresholds: NoiseThresholds | None = None):
        self.classifier = classifier
        self.templates = templates
        self._severity_agent = NoiseFeatureAgent(templates, thresholds)

    def __call__(self, state: ECGState) -> dict:
        signal = state["validation_result"]["signal"]
        features = extract_noise_features(signal, self.templates)
        prediction = self.classifier.predict_one(features)
        threshold_evidence = self._severity_agent(state)["noise_evidence"]

        enriched_features = {
            **features,
            "residual_noise_type": threshold_evidence["predicted_type"],   # e.g. "baseline_wander" -- NOT an attack label
            "residual_noise_ratio": threshold_evidence["severity"],        # how far past the quality threshold, independent of attack decision
        }
        return {"noise_evidence": {"score": prediction["score"], "predicted_type": prediction["predicted_type"],
                                    "severity": threshold_evidence["severity"], "features": enriched_features}}
