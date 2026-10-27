"""
agents/signal_poisoning/reconstruction.py

"Reconstruction check — autoencoder error vs clean" box.

An undercomplete autoencoder fitted on clean beats only. A poisoned beat
carries structure the autoencoder never learned to reproduce, so its
reconstruction error sits in the tail of the clean error distribution.

The reconstruction itself is also the Recovery agent's signal repair, so
this module reports how trustworthy it is (`reconstruction_confidence`).
Per the Recovery agent's contract that must NOT be "distance from the
original" (a good repair of an attacked beat is supposed to differ from
it). Instead it measures whether the reconstruction is a plausible clean
beat: pass it through the autoencoder again — clean-looking beats are
near fixed points — and compare that error with the clean fixed-point
error distribution.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class ReconstructionAutoencoder:
    def __init__(self, hidden_layers: tuple[int, ...] = (128, 32, 128), max_iter: int = 300, seed: int = 0):
        self._ae = MLPRegressor(hidden_layer_sizes=hidden_layers, activation="relu", max_iter=max_iter,
                                early_stopping=True, random_state=seed)

    def fit(self, clean_signals: np.ndarray) -> "ReconstructionAutoencoder":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._ae.fit(clean_signals, clean_signals)
        return self

    def reconstruct(self, signals: np.ndarray) -> np.ndarray:
        return np.asarray(self._ae.predict(np.atleast_2d(signals)), dtype=np.float64)

    def error(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mse per beat, reconstruction)."""
        recon = self.reconstruct(signals)
        return np.mean((np.atleast_2d(signals) - recon) ** 2, axis=1), recon

    def fixed_point_error(self, reconstructions: np.ndarray) -> np.ndarray:
        return self.error(reconstructions)[0]
