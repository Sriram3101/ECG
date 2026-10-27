"""
core/ecg_state.py

Shared state schema for the ECG agentic pipeline.

This file is the single source of truth for the shape of data as it flows
through the pipeline:

    Raw ECG beat -> Validation Agent -> (Poison / Noise / Diagnosis) -> Decision

IMPORTANT: This is a shared contract. If you need to add or rename a field,
flag it in the team channel before merging a change here.
"""

from typing import TypedDict, Optional, Dict, Any, List
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_BEAT_SHAPE = (360, 1)   # canonical shape every raw ECG beat must match
EXPECTED_BEAT_LENGTH = 360       # flattened length, used by 1-D checks/masks

VALID_STATUSES = ("PASS", "REPAIR", "REJECT")


# ---------------------------------------------------------------------------
# Validation-specific types
# ---------------------------------------------------------------------------

class ValidationFlags(TypedDict):
    """One boolean per check. True means the check FAILED (problem found)."""
    wrong_shape: bool
    bad_dtype: bool
    nan: bool
    missing_samples: bool
    flatline: bool
    clipping: bool
    abnormal_amplitude: bool


class ValidationResult(TypedDict):
    """Output of validate_beat() — what the Validation Agent hands back."""
    status: str                       # one of VALID_STATUSES
    signal: Optional[np.ndarray]      # repaired signal if status == 'REPAIR',
                                       # original signal if 'PASS', None if 'REJECT'
    flags: ValidationFlags
    reason: str                       # short human-readable explanation
    repair_actions: List[str]         # e.g. ['nan_interpolated', 'clip_smoothed']


# ---------------------------------------------------------------------------
# Ground-truth mask (shared with Neha R's injection harness, and reused by
# the Poison Agent and Noise Agent teams for their own benchmarking)
# ---------------------------------------------------------------------------

class GroundTruth(TypedDict, total=False):
    """
    Ground truth attached to synthetically corrupted/poisoned/noised beats
    during evaluation. Not present on real production beats — only used in
    evaluation/generate_invalid_data.py and downstream benchmark harnesses.
    """
    corruption_type: str               # e.g. 'nan', 'clipping', 'flatline',
                                        # 'waveform_poison', 'nstdb_noise'
    params: Dict[str, Any]             # injector-specific parameters used
    mask: Optional[np.ndarray]         # bool array, shape (EXPECTED_BEAT_LENGTH,)
                                        # marks exactly which samples were altered
    expected_status: Optional[str]     # for validation-style corruptions only
    is_corrupted: Optional[bool]       # for poison/noise injections


# ---------------------------------------------------------------------------
# Full pipeline state
# ---------------------------------------------------------------------------

class ECGState(TypedDict, total=False):
    """
    The object that flows through every node in the LangGraph pipeline.
    Each agent reads what it needs and writes its own key — never overwrites
    another agent's key.
    """
    beat_id: str
    raw_signal: np.ndarray             # shape (360, 1), as it entered the pipeline
    metadata: Dict[str, Any]           # record id, lead, sampling rate, patient info

    # populated by the Validation Agent (Neha S)
    validation: Optional[ValidationResult]

    # populated only on synthetic/benchmark data (Neha R, and shared with
    # the Poison / Noise teams for their own harnesses)
    ground_truth: Optional[GroundTruth]

    # reserved for downstream agents — added here as placeholders so the
    # shape is documented in one place, even though Neha S does not populate
    # these herself
    poison: Optional[Dict[str, Any]]       # Agent 2 (Nishitha, Sravya)
    noise: Optional[Dict[str, Any]]        # Agent 3 (Sriram, Nihar)
    diagnosis: Optional[Dict[str, Any]]    # Diagnosis Model
    decision: Optional[Dict[str, Any]]     # Decision Agent


# ---------------------------------------------------------------------------
# Small factory helpers (optional convenience, keep call sites clean)
# ---------------------------------------------------------------------------

def empty_validation_flags() -> ValidationFlags:
    """All-clear flags — useful as a starting point before running checks."""
    return ValidationFlags(
        wrong_shape=False,
        bad_dtype=False,
        nan=False,
        missing_samples=False,
        flatline=False,
        clipping=False,
        abnormal_amplitude=False,
    )


def make_ecg_state(beat_id: str, raw_signal: np.ndarray,
                    metadata: Optional[Dict[str, Any]] = None) -> ECGState:
    """Construct a fresh ECGState for a beat entering the pipeline."""
    return ECGState(
        beat_id=beat_id,
        raw_signal=raw_signal,
        metadata=metadata or {},
        validation=None,
        ground_truth=None,
        poison=None,
        noise=None,
        diagnosis=None,
        decision=None,
    )
