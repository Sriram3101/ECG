"""
evaluation/synthetic_beats.py

Synthetic single-lead beats with four distinguishable morphologies standing
in for the A / B / N / V classes, normalized per beat to [0, 1] exactly like
the MIT-BIH preprocessing notebook. Used by the end-to-end tests and
scripts/run_pipeline_demo.py so the full pipeline can run without the
MIT-BIH download or the trained Keras checkpoint.
"""
from __future__ import annotations

import numpy as np

T = np.arange(360)

# (center, width, amplitude) of P, QRS, T waves per class
_MORPHOLOGY = {
    0: [(120, 10, 0.35), (180, 5, 1.0), (260, 18, 0.3)],    # A: early P wave
    1: [(140, 10, 0.15), (180, 14, 1.0), (265, 22, -0.35)],  # B: wide QRS, inverted T
    2: [(140, 10, 0.15), (180, 5, 1.0), (260, 18, 0.3)],     # N: normal
    3: [(180, 22, -1.0), (250, 25, 0.6)],                    # V: no P, wide inverted complex
}


def _beat(cls: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(len(T))
    for center, width, amp in _MORPHOLOGY[cls]:
        center += rng.normal(0, 3)
        width *= rng.uniform(0.9, 1.1)
        x += amp * rng.uniform(0.85, 1.15) * np.exp(-0.5 * ((T - center) / width) ** 2)
    x += 0.05 * np.sin(2 * np.pi * T / 360 * rng.uniform(0.2, 0.6) + rng.uniform(0, 2 * np.pi))
    x += rng.normal(0, 0.01, len(T))
    return (x - x.min()) / (x.max() - x.min())


def make_beats(n_per_class: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(_MORPHOLOGY)), n_per_class)
    signals = np.stack([_beat(int(c), rng) for c in labels]).astype(np.float32)
    order = rng.permutation(len(labels))
    return signals[order], labels[order]


def flip_labels(labels: np.ndarray, fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Label poisoning: reassign `fraction` of labels to a different class.
    Returns (poisoned_labels, poisoned_mask)."""
    rng = np.random.default_rng(seed)
    poisoned = labels.copy()
    idx = rng.choice(len(labels), size=int(len(labels) * fraction), replace=False)
    num_classes = int(labels.max()) + 1
    poisoned[idx] = (labels[idx] + rng.integers(1, num_classes, size=len(idx))) % num_classes
    mask = np.zeros(len(labels), dtype=bool)
    mask[idx] = True
    return poisoned, mask


def poison_signal(signal: np.ndarray, sigma: float = 0.08, seed: int = 0) -> np.ndarray:
    """Signal poisoning: high-frequency additive perturbation."""
    rng = np.random.default_rng(seed)
    return (np.asarray(signal, dtype=np.float32) + rng.normal(0, sigma, np.shape(signal))).astype(np.float32)


def pgd_attack(backend, signal: np.ndarray, label: int, epsilon: float = 0.05, step_size: float = 0.01,
               num_steps: int = 10) -> np.ndarray:
    """Untargeted L-inf PGD against the classifier (same threat model as
    agents/signal_poisoning/noise/noise_datagen.pgd_adversarial_noise, but
    framework-agnostic via the backend's input gradient)."""
    x = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    x_adv = x.copy()
    for _ in range(num_steps):
        grad = backend.input_gradient(x_adv, np.array([label]))
        x_adv = np.clip(x_adv - step_size * np.sign(grad), x - epsilon, x + epsilon).astype(np.float32)
    return x_adv[0]
