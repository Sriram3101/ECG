"""
agents/signal_poisoning/signal_poisoning_agent.py

Signal poisoning detection: has the *waveform* itself been tampered with?

Emits exactly the four Evidence objects the Decision agent fuses
(agents/decision/fusion.py, DEFAULT_SIGNAL_CHECK_WEIGHTS):

    reconstruction_error     autoencoder error vs clean
    feature_space_distance   Mahalanobis on embeddings
    prediction_fragility     stability under perturbation
    gradient_sensitivity     local Lipschitz estimate

Each raw statistic is calibrated against a held-out slice of clean beats
(core/calibration.py), so a score of 0.5 means "3 robust standard
deviations further into the tail than typical clean beats".

The "reconstruction_error" Evidence carries `metadata["reconstruction"]`
and a `.confidence` that the Recovery agent reads for signal repair.

`artifacts["signal_quality"]` reports signal quality independently of the
attack verdict: estimated SNR, spectral entropy and — when the spectral
noise agent is attached — the noise type and severity.

Optionally, the spectral noise agent (agents/signal_poisoning/noise —
NoiseFeatureAgent or the trained NoiseClassifierAgent) runs alongside and
adds a fifth, "spectral_noise" Evidence. It is reported for audit and
residual-noise alerts; the Decision agent only fuses checks listed in its
`signal_check_weights`, so it does not affect the verdict unless you add a
weight for it there.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from agents.signal_poisoning.feature_distance import EmbeddingMahalanobis
from agents.signal_poisoning.fragility import gradient_sensitivity, prediction_fragility
from agents.signal_poisoning.reconstruction import ReconstructionAutoencoder
from core.calibration import RobustTailCalibrator
from core.signal_filters import estimated_snr_db, spectral_entropy
from models.backend import ClassifierBackend, as_batch
from shared.contracts import DetectionReport, Evidence

logger = logging.getLogger(__name__)

SIGNAL_CHECKS = ("reconstruction_error", "feature_space_distance", "prediction_fragility", "gradient_sensitivity")


@dataclass
class SignalPoisoningConfig:
    holdout_fraction: float = 0.2
    max_calibration_beats: int = 300
    fragility_perturbations: int = 16
    fragility_sigma: float = 0.02
    lipschitz_points: int = 8
    lipschitz_radius: float = 0.01
    z_offset: float = 3.0
    seed: int = 0


class SignalPoisoningAgent:
    name = "signal_poisoning"
    version = "0.1.0"

    def __init__(self, backend: ClassifierBackend, config: Optional[SignalPoisoningConfig] = None,
                 noise_agent: Optional[Callable[[dict], dict]] = None):
        self.backend = backend
        self.config = config or SignalPoisoningConfig()
        self.noise_agent = noise_agent
        self._autoencoder = ReconstructionAutoencoder(seed=self.config.seed)
        self._mahalanobis = EmbeddingMahalanobis()
        self._calibrators: dict[str, RobustTailCalibrator] = {}
        self._fixed_point_calibrator: RobustTailCalibrator | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self._calibrators)

    def _raw_statistics(self, signal: np.ndarray) -> tuple[dict[str, float], dict[str, dict]]:
        cfg = self.config
        x = as_batch(signal)
        recon_error, recon = self._autoencoder.error(x)
        fragility, fragility_meta = prediction_fragility(self.backend, x[0], cfg.fragility_perturbations,
                                                         cfg.fragility_sigma, cfg.seed)
        lipschitz, lipschitz_meta = gradient_sensitivity(self.backend, x[0], cfg.lipschitz_points,
                                                         cfg.lipschitz_radius, cfg.seed)
        raw = {
            "reconstruction_error": float(recon_error[0]),
            "feature_space_distance": float(self._mahalanobis.distance(self.backend.embed(x))[0]),
            "prediction_fragility": fragility,
            "gradient_sensitivity": lipschitz,
        }
        meta = {
            "reconstruction_error": {"reconstruction": recon[0]},
            "prediction_fragility": fragility_meta,
            "gradient_sensitivity": lipschitz_meta,
        }
        return raw, meta

    def fit(self, clean_signals: np.ndarray, calibration_signals: Optional[np.ndarray] = None) -> "SignalPoisoningAgent":
        """Fit on beats known to be clean (e.g. the clean training split the
        classifier was trained on).

        `calibration_signals`: clean beats from *other patients* than
        `clean_signals` (e.g. the validation split). Strongly recommended:
        calibrating on held-out beats of the same patients makes every new
        patient look anomalous — on the patient-wise MIT-BIH splits that
        flagged ~50% of clean test beats. Without it, a random holdout of
        `clean_signals` is used.
        """
        x = as_batch(clean_signals)
        if calibration_signals is not None:
            train, holdout = x, as_batch(calibration_signals)
        else:
            rng = np.random.default_rng(self.config.seed)
            order = rng.permutation(len(x))
            n_holdout = max(10, int(len(x) * self.config.holdout_fraction))
            if len(x) - n_holdout < 10:
                raise ValueError(f"need more clean beats to fit the signal poisoning agent (got {len(x)})")
            holdout, train = x[order[:n_holdout]], x[order[n_holdout:]]

        self._autoencoder.fit(train)
        self._mahalanobis.fit(self.backend.embed(train))

        calibration = holdout[: self.config.max_calibration_beats]
        raw_rows = [self._raw_statistics(beat)[0] for beat in calibration]
        self._calibrators = {
            check: RobustTailCalibrator(z_offset=self.config.z_offset).fit([r[check] for r in raw_rows])
            for check in SIGNAL_CHECKS
        }
        _, recon = self._autoencoder.error(calibration)
        self._fixed_point_calibrator = RobustTailCalibrator(z_offset=self.config.z_offset).fit(
            self._autoencoder.fixed_point_error(recon))
        logger.info("signal poisoning agent fitted on %d clean beats (%d for calibration)",
                    len(train), len(calibration))
        return self

    def run(self, record_id: str, signal: np.ndarray) -> DetectionReport:
        if not self.is_fitted:
            raise RuntimeError("SignalPoisoningAgent.run() called before fit()")
        raw, meta = self._raw_statistics(signal)

        reconstruction = meta["reconstruction_error"]["reconstruction"]
        fixed_point_error = float(self._autoencoder.fixed_point_error(reconstruction[None, :])[0])
        reconstruction_confidence = 1.0 - self._fixed_point_calibrator.score(fixed_point_error)

        evidence = []
        for check in SIGNAL_CHECKS:
            metadata = dict(meta.get(check, {}))
            confidence = 1.0
            if check == "reconstruction_error":
                metadata = {"reconstruction": reconstruction.tolist(), "fixed_point_error": fixed_point_error}
                confidence = reconstruction_confidence
            evidence.append(Evidence(check_name=check, channel="signal", score=self._calibrators[check].score(raw[check]),
                                     raw_value=raw[check], confidence=confidence, metadata=metadata))

        beat = as_batch(signal)[0]
        quality = {"snr_db": estimated_snr_db(beat), "spectral_entropy": spectral_entropy(beat)}
        artifacts: dict = {"autoencoder_reconstruction": reconstruction,
                           "reconstruction_confidence": reconstruction_confidence,
                           "signal_quality": quality}
        if self.noise_agent is not None:
            noise = self.noise_agent({"validation": {"signal": beat.reshape(-1, 1)}})["noise_evidence"]
            evidence.append(Evidence(check_name="spectral_noise", channel="signal", score=noise["score"],
                                     raw_value=noise["severity"],
                                     metadata={"predicted_type": noise["predicted_type"], "features": noise["features"]}))
            artifacts["noise_evidence"] = noise
            quality.update(noise_type=noise["predicted_type"], noise_severity=noise["severity"],
                           noise_score=noise["score"])

        return DetectionReport(record_id=record_id, agent_name=self.name, agent_version=self.version,
                               channel="signal", score=None, evidence=evidence, artifacts=artifacts)
