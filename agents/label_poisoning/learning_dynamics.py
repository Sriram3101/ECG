"""
agents/label_poisoning/learning_dynamics.py

"Learning dynamics — per-epoch AUM margin" box.

Area Under the Margin (Pleiss et al., 2020): train a small classifier on
the dataset under audit and, after every epoch, record each sample's
margin = p(assigned label) - max p(any other label). Correctly labeled
samples quickly reach a large positive margin; mislabeled samples are
pulled toward their true class by their neighbours, so their margin stays
low or negative across training. AUM is the mean margin over epochs.

The classifier is trained on the frozen DenseNet embeddings rather than
raw signal: it's fast enough to rerun on every audit, and it measures how
hard the *claimed label* is to fit given what the clean model learned.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _margins(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    assigned = probs[np.arange(len(labels)), labels]
    others = probs.copy()
    others[np.arange(len(labels)), labels] = -np.inf
    return assigned - others.max(axis=1)


def _full_proba(clf: MLPClassifier, z: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(z), num_classes))
    out[:, clf.classes_] = clf.predict_proba(z)
    return out


class AUMScorer:
    def __init__(self, num_classes: int, epochs: int = 15, hidden_units: int = 64, seed: int = 0):
        self.num_classes = num_classes
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._clf: MLPClassifier | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Train on (embeddings, claimed labels); return per-sample AUM."""
        self._scaler = StandardScaler().fit(embeddings)
        z = self._scaler.transform(embeddings)
        self._clf = MLPClassifier(hidden_layer_sizes=(self.hidden_units,), random_state=self.seed,
                                  learning_rate_init=1e-3, batch_size=min(128, len(z)))
        classes = np.arange(self.num_classes)
        margin_sum = np.zeros(len(z))
        for _ in range(self.epochs):
            self._clf.partial_fit(z, labels, classes=classes)
            margin_sum += _margins(_full_proba(self._clf, z, self.num_classes), labels)
        return margin_sum / self.epochs

    def margin(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Margin under the final-epoch model, for records that weren't part
        of the audited dataset (true AUM needs the training trajectory)."""
        if self._clf is None:
            raise RuntimeError("AUMScorer.margin() called before fit()")
        z = self._scaler.transform(embeddings)
        return _margins(_full_proba(self._clf, z, self.num_classes), labels)
