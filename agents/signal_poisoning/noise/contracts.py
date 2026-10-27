"""Noise-agent contracts.

The pipeline-wide ECGState lives in core/ecg_state.py (one contract for
every agent); this module only keeps the noise agent's own evidence shape
and the class-mapping guard.

Ground-truth fields are intentionally separate from detector inputs so they
cannot accidentally leak into a model or scoring rule.
"""
from __future__ import annotations

from typing import TypedDict

from core.ecg_state import CLASS_LABELS, EXPECTED_BEAT_LENGTH, ECGState

SCHEMA_VERSION = "1.1.0"
SAMPLE_COUNT = EXPECTED_BEAT_LENGTH
INDEX_TO_CLASS = dict(CLASS_LABELS)
CLASS_TO_INDEX = {value: key for key, value in INDEX_TO_CLASS.items()}

__all__ = ["ECGState", "NoiseEvidence", "CLASS_TO_INDEX", "INDEX_TO_CLASS", "SAMPLE_COUNT", "validate_label_mapping"]


class NoiseEvidence(TypedDict):
    score: float  # one calibrated scalar in [0, 1] per beat
    predicted_type: str
    severity: float
    features: dict[str, float]


def validate_label_mapping() -> None:
    """Fail early if a collaborator changes the class/model mapping."""
    if CLASS_TO_INDEX != {"A": 0, "B": 1, "N": 2, "V": 3}:
        raise ValueError("The configured class mapping no longer matches DenseNet training.")
