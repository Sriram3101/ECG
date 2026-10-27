"""
agents/signal_poisoning/feature_distance.py

"Feature-space distance — Mahalanobis on embeddings" box.

Mahalanobis distance of a beat's DenseNet+BiLSTM embedding from the clean
embedding distribution (shrinkage covariance, since embedding dimensions
are strongly correlated and the plain sample covariance is near-singular).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class EmbeddingMahalanobis:
    def __init__(self):
        self._cov: LedoitWolf | None = None

    def fit(self, clean_embeddings: np.ndarray) -> "EmbeddingMahalanobis":
        self._cov = LedoitWolf().fit(clean_embeddings)
        return self

    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        if self._cov is None:
            raise RuntimeError("EmbeddingMahalanobis.distance() called before fit()")
        return np.sqrt(np.maximum(self._cov.mahalanobis(np.atleast_2d(embeddings)), 0.0))
