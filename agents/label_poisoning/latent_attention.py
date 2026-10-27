"""
agents/label_poisoning/latent_attention.py

"Latent + attention — k-NN, Mahalanobis, entropy" box. Ported from the
team's PoisonDetection notebook, made label-aware: the notebook's
statistics measured "is this beat unusual"; for *label* poisoning the
question is "is this beat unusual *for the class it claims to be*".

All three statistics are computed in the embedding space of the
attention-pooled DenseNet+BiLSTM features:

  knn_disagreement   fraction of the k nearest audited beats whose label
                     differs from the claimed label
  mahalanobis_margin distance to the claimed class's centroid minus the
                     distance to the nearest class centroid (0 when the
                     claimed class *is* the nearest), tied covariance
  label_cross_entropy  -log p_model(claimed label); the notebook's
                     entropy / label-agreement signal, made label-aware.
                     Plain predictive entropy is kept in metadata.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-10


class LatentAttentionScorer:
    def __init__(self, num_classes: int, k_neighbors: int = 10):
        self.num_classes = num_classes
        self.k = k_neighbors
        self._knn: NearestNeighbors | None = None
        self._labels: np.ndarray | None = None
        self._centroids: dict[int, np.ndarray] = {}
        self._inv_cov: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "LatentAttentionScorer":
        self._labels = np.asarray(labels)
        self._knn = NearestNeighbors(n_neighbors=min(self.k + 1, len(embeddings))).fit(embeddings)
        self._centroids = {int(c): embeddings[labels == c].mean(axis=0) for c in np.unique(labels)}
        centered = embeddings - np.stack([self._centroids[int(c)] for c in labels])
        self._inv_cov = np.linalg.pinv(np.cov(centered, rowvar=False))
        return self

    def _mahalanobis(self, z: np.ndarray, centroid: np.ndarray) -> float:
        d = z - centroid
        return float(np.sqrt(max(d @ self._inv_cov @ d, 0.0)))

    def statistics(self, embeddings: np.ndarray, labels: np.ndarray, probs: np.ndarray,
                   exclude_self: bool = False) -> dict[str, np.ndarray]:
        """`exclude_self=True` when scoring the audited dataset itself, so a
        beat doesn't count as its own nearest neighbour."""
        if self._knn is None:
            raise RuntimeError("LatentAttentionScorer.statistics() called before fit()")
        n_query = self._knn.n_neighbors if exclude_self else self._knn.n_neighbors - 1
        _, idx = self._knn.kneighbors(embeddings, n_neighbors=max(n_query, 1))
        if exclude_self:
            idx = idx[:, 1:]
        knn_disagreement = (self._labels[idx] != labels[:, None]).mean(axis=1)

        mahalanobis_margin = np.empty(len(embeddings))
        for i, (z, y) in enumerate(zip(embeddings, labels)):
            distances = {c: self._mahalanobis(z, mu) for c, mu in self._centroids.items()}
            claimed = distances.get(int(y), max(distances.values()))
            mahalanobis_margin[i] = claimed - min(distances.values())

        label_cross_entropy = -np.log(probs[np.arange(len(labels)), labels] + _EPS)
        predictive_entropy = -np.sum(probs * np.log(probs + _EPS), axis=1)
        return {
            "knn_disagreement": knn_disagreement,
            "mahalanobis_margin": mahalanobis_margin,
            "label_cross_entropy": label_cross_entropy,
            "predictive_entropy": predictive_entropy,
        }
