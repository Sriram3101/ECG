"""
validation_agent.py

Top-level orchestrator for the ECG Validation Agent pipeline.

This module contains no validation logic of its own. It only wires
together the existing building blocks:

    format_checks.py  -> structural checks (shape, dtype, NaN, length, range)
    sanity_checks.py  -> signal-quality checks (flatline, clipping, amplitude)
    triage.py         -> turns flags + NaN ratio into a PASS/REPAIR/REJECT
                          decision
    repair.py          -> repairs a signal flagged for REPAIR

Workflow:

    signal
      -> format checks
      -> sanity checks
      -> build ValidationFlags
      -> check_invalid_entries()
      -> compute NaN ratio
      -> triage.decide()
          PASS   -> return ValidationResult(status="PASS")
          REJECT -> return ValidationResult(status="REJECT")
          REPAIR -> repair.repair_signal()
                    -> rerun ALL checks on the repaired signal
                    -> rerun triage.decide()
                        clean -> ValidationResult(status="REPAIR")
                        not clean -> ValidationResult(status="REJECT")
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agents.validation.format_checks import (
    check_shape,
    check_dtype,
    check_nan,
    check_missing_samples,
    check_invalid_entries,
)
from agents.validation.sanity_checks import (
    check_flatline,
    check_clipping,
    check_amplitude,
)
from agents.validation.triage import decide
from agents.validation.repair import repair_signal
from core.ecg_state import (
    ValidationFlags,
    ValidationResult,
    EXPECTED_BEAT_SHAPE,
    EXPECTED_BEAT_LENGTH,
    empty_validation_flags,
)

EXPECTED_SAMPLING_RATE = 360  # Hz, MIT-BIH


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds that depend on how beats are scaled.

    The defaults are the original ones, written for raw (mV / min-max)
    beats. The pipeline's beats are z-scored per beat (mean 0, std 1), where
    a normal R-peak sits at 5-8: with the defaults ~60% of genuine MIT-BIH
    beats fail the +/-5 range check, and R-peaks above 6 would be clamped
    as "abnormal amplitude". Use `ValidationConfig.zscored_beats()` there.
    """
    valid_range: Tuple[float, float] = (-5.0, 5.0)
    amplitude_z_thresh: float = 6.0
    flatline_std: float = 0.01
    clip_run_length: int = 3

    @classmethod
    def zscored_beats(cls) -> "ValidationConfig":
        # Largest |z| in the 69,802-beat training split is 8.32 (99.9th pct 7.93).
        return cls(valid_range=(-12.0, 12.0), amplitude_z_thresh=10.0)


def _run_checks(signal, config: ValidationConfig = ValidationConfig()) -> Tuple[ValidationFlags, float, bool]:
    """Run all format and sanity checks on ``signal``.

    Executes every check from format_checks.py and sanity_checks.py,
    folds the results into a single ValidationFlags dict, computes the
    NaN ratio, and separately reports whether check_invalid_entries()
    passed (this result is intentionally NOT stored on ValidationFlags,
    since it has no corresponding field in the core/ecg_state.py schema).

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.

    Returns:
        A tuple ``(flags, nan_ratio, invalid_entries_ok)`` where:
            - ``flags`` is a populated ValidationFlags dict.
            - ``nan_ratio`` is the fraction of samples that are NaN.
            - ``invalid_entries_ok`` is True if check_invalid_entries()
              passed (no infs / out-of-range values), False otherwise.
    """
    flags: ValidationFlags = empty_validation_flags()

    # --- format checks -----------------------------------------------
    shape_ok, _ = check_shape(signal, EXPECTED_BEAT_SHAPE)
    flags["wrong_shape"] = not shape_ok

    dtype_ok, _ = check_dtype(signal)
    flags["bad_dtype"] = not dtype_ok

    nan_ok, nan_mask = check_nan(signal)
    flags["nan"] = not nan_ok

    missing_ok, _ = check_missing_samples(signal, EXPECTED_BEAT_LENGTH)
    flags["missing_samples"] = not missing_ok

    # check_invalid_entries() is always executed, but its result is kept
    # separate from ValidationFlags (no such field exists on it).
    invalid_entries_ok, _ = check_invalid_entries(signal, config.valid_range)

    # --- sanity checks -------------------------------------------------
    flatline_ok, _ = check_flatline(signal, config.flatline_std)
    flags["flatline"] = not flatline_ok

    clipping_ok, _ = check_clipping(signal, config.clip_run_length)
    flags["clipping"] = not clipping_ok

    amplitude_ok, _ = check_amplitude(signal, config.amplitude_z_thresh)
    flags["abnormal_amplitude"] = not amplitude_ok

    # --- NaN ratio -------------------------------------------------------
    nan_mask_arr = np.asarray(nan_mask)
    total_samples = nan_mask_arr.size
    nan_ratio = float(nan_mask_arr.sum()) / float(total_samples) if total_samples > 0 else 0.0

    return flags, nan_ratio, invalid_entries_ok


def validate_beat(signal, metadata: Optional[Dict[str, Any]] = None,
                  config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Validate (and, if needed, repair) a single ECG beat.

    This is the single public entry point of the validation agent. It
    orchestrates format checks, sanity checks, triage, and repair, but
    performs no validation decisions itself -- all decisions come from
    ``triage.decide()``, and all repairs come from ``repair.repair_signal()``.

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.
        metadata: Optional beat metadata (record id, lead, sampling rate,
            etc.), accepted for interface compatibility with the rest of
            the ECGState pipeline. ValidationResult has no metadata field,
            except for a stated sampling rate ("fs" or "sampling_rate"),
            which must be 360 Hz.
        config: Scaling-dependent thresholds; see ValidationConfig.

    Returns:
        A ValidationResult dict describing the final outcome:
            - status="PASS": signal was clean, returned unmodified.
            - status="REJECT": signal failed validation (either
              initially, or the repaired signal still failed);
              ``signal`` is set to None.
            - status="REPAIR": signal was repaired and the repaired
              version passed re-validation cleanly.
    """
    config = config or ValidationConfig()
    flags, nan_ratio, invalid_entries_ok = _run_checks(signal, config)

    fs = (metadata or {}).get("fs", (metadata or {}).get("sampling_rate"))
    if fs is not None and float(fs) != EXPECTED_SAMPLING_RATE:
        result: ValidationResult = {
          "status": "REJECT",
          "signal": None,
          "flags": flags,
          "reason": f"sampling rate {fs} Hz, expected {EXPECTED_SAMPLING_RATE} Hz",
          "repair_actions": [],
        }
        return result

    # check_invalid_entries() failures are treated as a structural
    # validation failure and short-circuit straight to REJECT, without
    # ever being added to ValidationFlags or influencing triage.decide().
    if not invalid_entries_ok:
        result: ValidationResult = {
          "status": "REJECT",
          "signal": None,
          "flags": flags,
          "reason": "invalid entries: out-of-range or non-finite values",
          "repair_actions": [],
        }
        return result

    decision, reason = decide(flags, nan_ratio)

    if decision == "PASS":
        result: ValidationResult = {
                  "status": "PASS",
                  "signal": signal,
                  "flags": flags,
                  "reason": reason,
                  "repair_actions": [],
        }
        return result

    if decision == "REJECT":
        result: ValidationResult = {
                  "status": "REJECT",
                  "signal": None,
                  "flags": flags,
                  "reason": reason,
                  "repair_actions": [],
                }
        return result

    # decision == "REPAIR": repair the signal, then rerun the entire
    # validation pipeline from scratch on the repaired result.
    repaired_signal, repair_actions = repair_signal(signal, flags, config.amplitude_z_thresh)

    post_repair_flags, post_repair_nan_ratio, post_repair_invalid_ok = _run_checks(
        repaired_signal, config
    )

    if not post_repair_invalid_ok:
        result: ValidationResult = {
                  "status": "REJECT",
                  "signal": None,
                  "flags": post_repair_flags,
                  "reason": "invalid entries persisted after repair",
                  "repair_actions": repair_actions,
                }
        return result

    post_repair_decision, post_repair_reason = decide(
        post_repair_flags, post_repair_nan_ratio
    )

    if post_repair_decision == "PASS":
        result: ValidationResult = {
                  "status": "REPAIR",
                  "signal": repaired_signal,
                  "flags": post_repair_flags,
                  "reason": post_repair_reason,
                  "repair_actions": repair_actions,
                }
        return result

    # Repaired signal is still not clean (REPAIR again or REJECT) -- per
    # the required workflow, this is a final REJECT.
    result: ValidationResult = {
             "status": "REJECT",
             "signal": None,
             "flags": post_repair_flags,
             "reason": post_repair_reason,
             "repair_actions": repair_actions,
            }
    return result
