"""
agents/decision/fusion.py

Combines the four *unfused* signal-poisoning evidence scores
(reconstruction error, feature-space distance, prediction fragility,
gradient sensitivity) into one signal_score in [0, 1].

The label-poisoning side does NOT need this module — per the pipeline
diagram, that pipeline already ends in its own "Score fusion" step and
hands the Decision agent a single calibrated label_score directly.

Two fusion strategies are provided:

- `fuse_signal_evidence`      weighted arithmetic mean (default, most
                              transparent / easiest to explain and audit)
- `fuse_signal_evidence_noisy_or`
                              probabilistic OR — use if you'd rather a
                              single strongly-suspicious check dominate
                              the fused score instead of being averaged
                              down by three calm ones. Swap in via config
                              once you've evaluated which behaves better
                              on your attack-simulation set.
"""
from __future__ import annotations

import math
from typing import Iterable

from shared.contracts import Evidence

# Default weights — override via agents/decision/config/thresholds.yaml
# ("signal_check_weights") rather than editing this dict in place.
DEFAULT_SIGNAL_CHECK_WEIGHTS: dict[str, float] = {
    "reconstruction_error": 0.30,
    "feature_space_distance": 0.25,
    "prediction_fragility": 0.20,
    "gradient_sensitivity": 0.25,
}


def _index_by_check_name(evidence: Iterable[Evidence]) -> dict[str, Evidence]:
    by_name: dict[str, Evidence] = {}
    for e in evidence:
        if e.channel != "signal":
            continue
        if e.check_name in by_name:
            raise ValueError(
                f"duplicate signal evidence for check '{e.check_name}' — "
                "the signal-poisoning agent should emit exactly one "
                "Evidence object per check per record"
            )
        by_name[e.check_name] = e
    return by_name


def _validate_coverage(by_name: dict[str, Evidence], weights: dict[str, float]) -> None:
    missing = set(weights) - set(by_name)
    if missing:
        raise ValueError(
            f"missing required signal-poisoning checks: {sorted(missing)}. "
            f"Got: {sorted(by_name)}. Every check listed in "
            "signal_check_weights must be present before fusion runs."
        )


def fuse_signal_evidence(
    evidence: list[Evidence],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted average of the signal-poisoning checks -> a single [0,1] score.

    Args:
        evidence: full evidence list for the record (label + signal checks
            mixed is fine — non-"signal" channel entries are ignored).
        weights: check_name -> weight, must sum to a positive number and
            cover every check named in it. Defaults to
            DEFAULT_SIGNAL_CHECK_WEIGHTS.

    Raises:
        ValueError: if a required check is missing, or a check appears
            more than once.
    """
    weights = weights or DEFAULT_SIGNAL_CHECK_WEIGHTS
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("signal_check_weights must sum to a positive number")

    by_name = _index_by_check_name(evidence)
    _validate_coverage(by_name, weights)

    weighted_sum = sum(weights[name] * by_name[name].score for name in weights)
    return _clamp(weighted_sum / total_weight)


def fuse_signal_evidence_noisy_or(
    evidence: list[Evidence],
    weights: dict[str, float] | None = None,
) -> float:
    """Probabilistic-OR fusion: P(at least one check is 'right' that this
    is poisoned), with each check's contribution scaled by its weight.

    fused = 1 - prod_i (1 - weight_i * score_i)

    Behaves like the weighted mean when scores are similar, but lets a
    single high-confidence check (e.g. reconstruction_error = 0.95) push
    the fused score high even if the other three are near zero — useful
    if false negatives on any single strong signal are more costly than
    false positives from noise in the calmer checks.
    """
    weights = weights or DEFAULT_SIGNAL_CHECK_WEIGHTS
    by_name = _index_by_check_name(evidence)
    _validate_coverage(by_name, weights)

    # Normalize weights to [0,1] so each term (1 - w*s) stays in [0,1].
    max_w = max(weights.values())
    product = 1.0
    for name, w in weights.items():
        normalized_w = w / max_w
        product *= 1.0 - normalized_w * by_name[name].score
    return _clamp(1.0 - product)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
