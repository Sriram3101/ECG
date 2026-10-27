"""
agents/signal_poisoning/fragility.py

"Prediction fragility — stability under perturbation" box, and
"Gradient sensitivity — local Lipschitz estimate" box.

Adversarially perturbed beats are pushed just across a decision boundary,
so the classifier's output around them is unstable: small random noise
flips or reshapes the prediction (fragility), and the output changes
steeply with the input (high local Lipschitz constant). Clean beats sit
well inside their class region and are stable on both counts.
"""
from __future__ import annotations

import numpy as np

from models.backend import ClassifierBackend


def prediction_fragility(backend: ClassifierBackend, signal: np.ndarray, n_perturbations: int = 16,
                         sigma: float = 0.02, seed: int = 0) -> tuple[float, dict]:
    """Mean total-variation distance between the prediction on the beat and
    on randomly perturbed copies, plus the fraction of copies whose
    predicted class flips."""
    x = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=(n_perturbations, x.shape[1]))
    probs = backend.predict_proba(np.vstack([x, x + noise.astype(np.float32)]))
    base, perturbed = probs[0], probs[1:]
    tv_distance = 0.5 * np.abs(perturbed - base).sum(axis=1)
    flip_rate = float(np.mean(perturbed.argmax(axis=1) != base.argmax()))
    raw = float(tv_distance.mean() + flip_rate)
    return raw, {"mean_tv_distance": float(tv_distance.mean()), "flip_rate": flip_rate}


def gradient_sensitivity(backend: ClassifierBackend, signal: np.ndarray, n_points: int = 8,
                         radius: float = 0.01, seed: int = 0) -> tuple[float, dict]:
    """Local Lipschitz estimate of p(predicted class) around the beat: the
    largest input-gradient L2 norm over the beat and a few random points
    in a small ball around it."""
    x = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    predicted = int(backend.predict_proba(x)[0].argmax())
    offsets = np.random.default_rng(seed).normal(0.0, radius, size=(n_points - 1, x.shape[1]))
    points = np.vstack([x, x + offsets.astype(np.float32)])
    grads = backend.input_gradient(points, np.full(len(points), predicted))
    norms = np.linalg.norm(grads, axis=1)
    return float(norms.max()), {"gradient_norm_at_beat": float(norms[0]), "predicted_class": predicted}
