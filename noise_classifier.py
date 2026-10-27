"""Nihar's trained signal-poisoning (adversarial) classifier.

Consumes the exact feature dict Sriram's `extract_noise_features` produces
and predicts adversarial-perturbation evidence. Deliberately built on his
features (not raw signal) so it's a drop-in replacement for the
threshold-based `NoiseFeatureAgent` -- same input, same `noise_evidence`
output contract.

Reframed from a noise-*type* classifier to a signal-poisoning *detector*:
only PGD-adversarial beats are the positive class now. Real NSTDB sensor
noise (baseline_wander / ma / em) is treated as a held-out "distractor" set
used only for a specificity check (`evaluate_distractor_specificity`), never
as a training label. See `noise_datagen.sample_role_from_noise_type`, which
is the single source of truth for turning a `noise_type` into one of
"clean" / "distractor" / "attack".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Must match the keys extract_noise_features() returns, in a fixed order.
FEATURE_ORDER = [
    "baseline_power_ratio", "powerline_50hz_ratio", "emg_power_ratio",
    "wavelet_mad_sigma", "wavelet_residual_kurtosis", "template_correlation",
]

# Sensor-noise categories (baseline_wander/ma/em) are deliberately NOT
# prediction targets any more -- see module docstring. There used to be a
# multiclass `type_classifier` distinguishing bw/ma/em/pgd_adversarial as
# co-equal categories; that's gone, since asking "which sensor-noise type"
# was the wrong question for a signal-poisoning detector.
PREDICTED_TYPES = ["none", "adversarial_perturbation"]


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_ORDER], dtype=np.float64)


@dataclass
class TrainedNoiseClassifier:
    scaler: StandardScaler
    detector: CalibratedClassifierCV   # binary clean-vs-attack, calibrated -> score in [0,1]
    threshold: float = 0.5

    def save(self, path: str | Path) -> None:
        joblib.dump({"scaler": self.scaler, "detector": self.detector, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedNoiseClassifier":
        blob = joblib.load(path)
        return cls(**blob)

    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        vec = features_to_vector(features).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        score = float(self.detector.predict_proba(vec_scaled)[0, 1])
        predicted_type = "adversarial_perturbation" if score >= self.threshold else "none"
        return {"score": score, "predicted_type": predicted_type}


def split_distractor_for_training(sample_role: list[str] | np.ndarray, train_fraction: float = 0.7,
                                   seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Relabels a `train_fraction` share of "distractor" rows as "clean" so
    `train_noise_classifier` folds them in as explicit NEGATIVE ("not
    attack") examples, alongside real clean beats -- never as positive. This
    is different from leaving distractor out of training entirely: it
    teaches the decision boundary to specifically exclude broadband sensor
    noise, rather than just never encountering it and generalizing "any
    large deviation = attack" (which real-data runs showed happens
    otherwise -- see `evaluate_distractor_specificity`).

    Returns (role_for_training, distractor_holdout_mask). Pass the first to
    `train_noise_classifier`; pass the second as `restrict_to` to
    `evaluate_distractor_specificity`, so specificity is measured only on
    distractor beats the model never saw -- scoring against all "distractor"
    rows here would be measuring the model on its own training data.
    """
    sample_role = np.asarray(sample_role, dtype=object)
    role_for_training = sample_role.copy()
    distractor_idx = np.flatnonzero(sample_role == "distractor")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(distractor_idx)
    n_train = int(len(shuffled) * train_fraction)
    train_idx, holdout_idx = shuffled[:n_train], shuffled[n_train:]

    role_for_training[train_idx] = "clean"  # explicit negative, never "attack"
    holdout_mask = np.zeros(len(sample_role), dtype=bool)
    holdout_mask[holdout_idx] = True
    return role_for_training, holdout_mask


def train_noise_classifier(feature_matrix: np.ndarray, sample_role: list[str] | np.ndarray,
                            threshold: float = 0.5, seed: int = 42) -> TrainedNoiseClassifier:
    """`feature_matrix` rows follow FEATURE_ORDER. `sample_role[i]` is one of
    "clean" / "distractor" / "attack" -- get this from
    `noise_datagen.sample_role_from_noise_type`, don't derive it inline.

    Trains on "clean" vs "attack" rows ONLY. "distractor" rows (real NSTDB
    sensor noise) are excluded from training entirely; score them
    afterward with `evaluate_distractor_specificity` instead of folding
    them into the positive class the way the old `is_corrupted` flag did.
    """
    sample_role = np.asarray(sample_role, dtype=object)
    train_mask = np.isin(sample_role, ["clean", "attack"])
    if not train_mask.any():
        raise ValueError("No 'clean' or 'attack' rows to train on -- check sample_role values.")

    X = feature_matrix[train_mask]
    y = sample_role[train_mask] == "attack"

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    # cv folds can't exceed the smaller class's count; keep this robust on small demo sets.
    min_class_count = min(np.bincount(y_train.astype(int)))
    cv_folds = max(2, min(5, min_class_count))
    detector = CalibratedClassifierCV(GradientBoostingClassifier(random_state=seed), method="isotonic", cv=cv_folds)
    detector.fit(X_train_s, y_train)

    val_score = detector.score(X_val_s, y_val)
    print(f"[noise_classifier] held-out attack-detection accuracy: {val_score:.3f}")
    return TrainedNoiseClassifier(scaler=scaler, detector=detector, threshold=threshold)


def evaluate_distractor_specificity(classifier: TrainedNoiseClassifier, feature_matrix: np.ndarray,
                                     sample_role: list[str] | np.ndarray,
                                     restrict_to: np.ndarray | None = None) -> float:
    """Specificity check: what fraction of held-out NSTDB sensor-noise beats
    (never seen in training) does the detector wrongly flag as an attack?

    This is the number that actually tells you whether the detector learned
    "adversarial structure" versus just "signal looks unusual" -- run it
    every time you retrain, not just once. Lower is better.

    If you used `split_distractor_for_training`, pass its `distractor_holdout_mask`
    as `restrict_to` -- otherwise this scores against every "distractor" row,
    including ones the model may have already trained on, which would
    overstate specificity.
    """
    sample_role = np.asarray(sample_role, dtype=object)
    mask = sample_role == "distractor"
    if restrict_to is not None:
        mask = mask & np.asarray(restrict_to, dtype=bool)
    if not mask.any():
        print("[noise_classifier] no distractor rows found -- skipping specificity check.")
        return float("nan")
    X_scaled = classifier.scaler.transform(feature_matrix[mask])
    scores = classifier.detector.predict_proba(X_scaled)[:, 1]
    fpr = float(np.mean(scores >= classifier.threshold))
    print(f"[noise_classifier] distractor false-positive rate: {fpr:.3f} (n={int(mask.sum())})")
    return fpr
