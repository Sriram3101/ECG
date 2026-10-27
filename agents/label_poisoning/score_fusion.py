"""
agents/label_poisoning/score_fusion.py

"Score fusion — per-class calibration" box.

Each raw statistic is calibrated *per claimed class* (some classes are
inherently harder — e.g. A vs N beats overlap far more than V vs N — so a
single global threshold would flag every hard-class beat). The calibrated
check scores are then combined with a weighted mean into one label_score
in [0, 1], which is what the Decision agent consumes.
"""
from __future__ import annotations

from typing import Hashable

import numpy as np

from core.calibration import RobustTailCalibrator

DEFAULT_LABEL_CHECK_WEIGHTS: dict[str, float] = {
    "learning_dynamics": 0.35,
    "confident_learning": 0.35,
    "latent_attention": 0.30,
}

# Which raw statistics feed each check (all oriented higher = more suspicious).
CHECK_STATISTICS: dict[str, tuple[str, ...]] = {
    "learning_dynamics": ("negative_aum",),
    "confident_learning": ("oof_disagreement",),
    "latent_attention": ("knn_disagreement", "mahalanobis_margin", "label_cross_entropy"),
}


class PerClassScoreFusion:
    def __init__(self, weights: dict[str, float] | None = None, z_offset: float = 3.0):
        self.weights = weights or DEFAULT_LABEL_CHECK_WEIGHTS
        if set(self.weights) != set(CHECK_STATISTICS):
            raise ValueError(f"label check weights must cover exactly {sorted(CHECK_STATISTICS)}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("label check weights must sum to a positive number")
        self.z_offset = z_offset
        self._calibrators: dict[str, RobustTailCalibrator] = {}

    def fit(self, statistics: dict[str, np.ndarray], labels: np.ndarray) -> "PerClassScoreFusion":
        groups = [int(y) for y in labels]
        self._calibrators = {
            name: RobustTailCalibrator(z_offset=self.z_offset).fit(statistics[name], groups)
            for stats in CHECK_STATISTICS.values() for name in stats
        }
        return self

    def check_scores(self, statistics: dict[str, float], label: Hashable) -> dict[str, float]:
        """Calibrated [0,1] score per check for one record."""
        scores = {}
        for check, stat_names in CHECK_STATISTICS.items():
            scores[check] = float(np.mean([
                self._calibrators[name].score(statistics[name], int(label)) for name in stat_names
            ]))
        return scores

    def fuse(self, check_scores: dict[str, float]) -> float:
        total = sum(self.weights.values())
        fused = sum(self.weights[c] * check_scores[c] for c in self.weights) / total
        return float(min(1.0, max(0.0, fused)))
