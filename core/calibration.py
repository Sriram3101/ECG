"""
core/calibration.py

Turns a raw detector statistic (higher = more suspicious) into a score in
[0, 1] that the Decision agent can threshold.

Both detection agents use this so their scores live on the same scale:

    z     = (raw - median_g) / scale_g
    score = 1 / (1 + exp(-slope * (z - z_offset)))

`median_g` / `scale_g` are robust (median / MAD) statistics fitted per
group — per *claimed class* on the label side ("Score fusion: per-class
calibration" in the architecture), and globally on the signal side. Robust
statistics are used on purpose: the label side calibrates on the very
dataset it is auditing, so a poisoned minority must not drag the
reference distribution toward itself.

With the defaults a record sitting 3 robust standard deviations above its
group's median scores 0.5.
"""
from __future__ import annotations

from typing import Hashable, Iterable, Optional

import numpy as np

_GLOBAL = "__global__"
_MAD_TO_STD = 1.4826


class RobustTailCalibrator:
    def __init__(self, z_offset: float = 3.0, slope: float = 1.0, min_scale: float = 1e-6):
        self.z_offset = z_offset
        self.slope = slope
        self.min_scale = min_scale
        self._stats: dict[Hashable, tuple[float, float]] = {}

    def fit(self, values: Iterable[float], groups: Optional[Iterable[Hashable]] = None) -> "RobustTailCalibrator":
        values = np.asarray(list(values), dtype=float)
        if values.size == 0:
            raise ValueError("cannot calibrate on an empty set of values")
        self._stats = {_GLOBAL: self._robust_stats(values)}
        if groups is not None:
            groups = np.asarray(list(groups), dtype=object)
            for g in np.unique(groups):
                group_values = values[groups == g]
                # Too few members for stable statistics: fall back to global.
                if group_values.size >= 5:
                    self._stats[g] = self._robust_stats(group_values)
        return self

    def _robust_stats(self, values: np.ndarray) -> tuple[float, float]:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * _MAD_TO_STD
        # A MAD of 0 happens for near-discrete statistics (e.g. a k-NN
        # disagreement fraction that is 0 for most records). Fall back to a
        # fraction of the standard deviation so one tiny deviation doesn't
        # explode into an infinite z-score.
        scale = max(mad, 0.25 * float(np.std(values)), self.min_scale)
        return median, scale

    def score(self, raw: float, group: Optional[Hashable] = None) -> float:
        if not self._stats:
            raise RuntimeError("RobustTailCalibrator.score() called before fit()")
        median, scale = self._stats.get(group, self._stats[_GLOBAL])
        z = (float(raw) - median) / scale
        return float(1.0 / (1.0 + np.exp(-self.slope * (z - self.z_offset))))

    def score_many(self, raw: Iterable[float], groups: Optional[Iterable[Hashable]] = None) -> np.ndarray:
        raw = list(raw)
        groups = list(groups) if groups is not None else [None] * len(raw)
        return np.array([self.score(r, g) for r, g in zip(raw, groups)])
