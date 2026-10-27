"""
agents/label_poisoning/confident_learning.py

"Confident learning — k-fold OOF probabilities" box.

Every sample is scored by a model that never saw it during training
(k-fold out-of-fold prediction), so a mislabeled sample can't memorize its
own wrong label. The OOF probability vector is also exactly what the
Recovery agent's label repair consumes (`strategies/label_repair.py`), so
it is returned as an artifact rather than recomputed downstream.

Suspicion statistic: max_{j != y} p_j - p_y, i.e. how much more the
out-of-fold model believes some *other* class than the claimed one.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def label_disagreement(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.atleast_2d(probs)
    labels = np.atleast_1d(labels)
    assigned = probs[np.arange(len(labels)), labels]
    others = probs.copy()
    others[np.arange(len(labels)), labels] = -np.inf
    return others.max(axis=1) - assigned


class ConfidentLearningScorer:
    def __init__(self, num_classes: int, n_folds: int = 5, seed: int = 0):
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.seed = seed
        self._fold_models: list = []

    def _new_model(self):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def _full_proba(self, model, z: np.ndarray) -> np.ndarray:
        out = np.zeros((len(z), self.num_classes))
        out[:, model.classes_] = model.predict_proba(z)
        return out

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Return (N, num_classes) out-of-fold probabilities."""
        counts = np.bincount(labels)
        # StratifiedKFold needs every class present in every fold.
        n_folds = max(2, min(self.n_folds, int(counts[counts > 0].min())))
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(labels), self.num_classes))
        self._fold_models = []
        for train_idx, held_out_idx in folds.split(embeddings, labels):
            model = self._new_model().fit(embeddings[train_idx], labels[train_idx])
            oof[held_out_idx] = self._full_proba(model, embeddings[held_out_idx])
            self._fold_models.append(model)
        return oof

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Fold-ensemble probabilities for records outside the audited
        dataset — none of the fold models has seen them, so this is still
        an out-of-sample estimate."""
        if not self._fold_models:
            raise RuntimeError("ConfidentLearningScorer.predict_proba() called before fit()")
        return np.mean([self._full_proba(m, embeddings) for m in self._fold_models], axis=0)
