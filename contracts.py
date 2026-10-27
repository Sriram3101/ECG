"""Shared, versioned contracts for all ECG-defense contributors.

Ground-truth fields are intentionally separate from detector inputs so they
cannot accidentally leak into a model or scoring rule.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict
import numpy as np

SCHEMA_VERSION = "1.0.0"
SAMPLE_COUNT = 360
CLASS_TO_INDEX = {"A": 0, "B": 1, "N": 2, "V": 3}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}


class ValidationResult(TypedDict):
    status: Literal["PASS", "REPAIRED", "REJECT"]
    signal: np.ndarray
    flags: dict[str, Any]


class NoiseEvidence(TypedDict):
    score: float  # one calibrated scalar in [0, 1] per beat
    predicted_type: str
    severity: float
    features: dict[str, float]


class ECGState(TypedDict, total=False):
    """Only the orchestrator writes routing/decision keys.

    Owner map: validation_result=Neha S; poison_evidence=Nishitha/Sravya;
    noise_evidence=Sriram/Nihar; decision=Decision Agent.
    """
    schema_version: str
    record_id: str
    signal: np.ndarray              # detector input only, exactly (360, 1)
    label: int | None               # model index, never used as a noise feature
    validation_result: ValidationResult
    poison_evidence: dict[str, Any] # placeholder contract: {"score": float, ...}
    noise_evidence: NoiseEvidence
    decision: dict[str, Any]
    # Evaluation-only: NEVER consume these in a detector.
    ground_truth: dict[str, Any]


def validate_label_mapping() -> None:
    """Fail early if a collaborator changes the class/model mapping."""
    if CLASS_TO_INDEX != {"A": 0, "B": 1, "N": 2, "V": 3}:
        raise ValueError("The configured class mapping no longer matches DenseNet training.")
