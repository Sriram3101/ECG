"""
repair.py

Signal-repair utilities for the ECG Validation Agent pipeline.

This module is invoked only when ``triage.py`` has already decided that a
signal's status is ``REPAIR``. It performs no validation and makes no
accept/reject decisions of its own -- it only transforms the signal so that
it can be re-validated afterwards by ``validation_agent.py``.

Supported repairs:
    1. NaN interpolation      -> repair_nan
    2. Clipped-run smoothing  -> repair_clipping
    3. Amplitude clamping     -> repair_amplitude

All functions are pure: they never mutate the array passed in by the
caller, and they always return a new array with the same shape as the
input (``(360,)`` in, ``(360,)`` out; ``(360, 1)`` in, ``(360, 1)`` out).
"""

from typing import Dict, List, Tuple

import numpy as np

# Tolerance used when comparing floating point samples to the signal's
# min/max value in order to decide whether a sample sits "at" a clipped
# rail. Mirrors the tolerance used in sanity_checks.check_clipping.
_CLIP_TOL = 1e-6

# Minimum run length (in consecutive samples) required for a run of
# max/min valued samples to be considered "clipping" rather than a signal
# that simply happens to pass through its own extremum once or twice.
# Mirrors sanity_checks.check_clipping's default.
_CLIP_RUN_LENGTH = 3

# Z-score threshold beyond which samples are clamped. Mirrors
# sanity_checks.check_amplitude's default.
_Z_THRESH = 6.0

# A standard deviation below this is treated as "effectively zero" for the
# purposes of z-scoring, to avoid division blow-ups on near-flat inputs.
_STD_EPS = 1e-9


def _flatten(signal) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Convert ``signal`` to a 1-D float64 working copy.

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.

    Returns:
        A tuple of (flat_copy, original_shape) where ``flat_copy`` is a
        brand-new 1-D ``float64`` array (safe to mutate freely) and
        ``original_shape`` is the shape of the input as a numpy array,
        used later to restore the caller's expected shape.
    """
    arr = np.asarray(signal, dtype=np.float64)
    original_shape = arr.shape
    # .flatten() always returns a copy (unlike .ravel()), so the caller's
    # original array is never touched by downstream in-place edits.
    flat = arr.flatten().copy()
    return flat, original_shape


def _restore_shape(flat: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a 1-D array back to the caller's original shape.

    Falls back to the flat array unchanged if the original shape's total
    element count doesn't match (defensive; should not happen in normal
    pipeline use).
    """
    if flat.size == int(np.prod(original_shape)) if original_shape else False:
        return flat.reshape(original_shape)
    return flat


def _find_run_mask(bool_mask: np.ndarray, run_length: int) -> np.ndarray:
    """Flag samples that belong to a run of ``True`` values of length >= run_length.

    Isolated True values (runs shorter than ``run_length``) are left
    unflagged, matching sanity_checks.check_clipping's definition of
    "clipping" as a sustained run at a rail rather than a single sample
    touching the extremum.

    Args:
        bool_mask: 1-D boolean array.
        run_length: Minimum consecutive True count to flag.

    Returns:
        Boolean array, same shape as ``bool_mask``, True only for samples
        that are part of a qualifying run.
    """
    n = bool_mask.shape[0]
    out = np.zeros(n, dtype=bool)
    run_start = None
    for i in range(n):
        if bool_mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start >= run_length:
                    out[run_start:i] = True
                run_start = None
    # Handle a run that extends to the very end of the array.
    if run_start is not None and n - run_start >= run_length:
        out[run_start:n] = True
    return out


def repair_nan(signal) -> np.ndarray:
    """Repair NaN samples using linear interpolation.

    Missing (NaN) samples are replaced by linearly interpolating between
    the nearest valid (non-NaN) neighbors on either side. NaNs at the very
    start or end of the signal (no neighbor on one side) are filled with
    the nearest valid sample's value, which is ``numpy.interp``'s default
    behavior for out-of-range x-values.

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.

    Returns:
        A new array, same shape as the input, with all NaNs replaced.
        If every sample is NaN, interpolation is impossible; in that case
        the signal is returned as all zeros (a safe, deterministic
        fallback) since there is no valid data to interpolate from.
    """
    flat, original_shape = _flatten(signal)

    nan_mask = np.isnan(flat)
    if not nan_mask.any():
        # Nothing to do; still return a fresh copy, not the input object.
        return _restore_shape(flat, original_shape)

    valid_mask = ~nan_mask
    if not valid_mask.any():
        # Entire signal is NaN -- no anchor points exist to interpolate
        # from. Fall back to zeros rather than raising, so the pipeline
        # can continue and let downstream re-validation reject/flag it.
        flat[:] = 0.0
        return _restore_shape(flat, original_shape)

    indices = np.arange(flat.shape[0])
    valid_indices = indices[valid_mask]
    valid_values = flat[valid_mask]

    # np.interp clamps to the first/last valid value for x outside the
    # known range, which naturally handles NaN runs touching the edges.
    flat[nan_mask] = np.interp(indices[nan_mask], valid_indices, valid_values)

    return _restore_shape(flat, original_shape)


def repair_clipping(signal) -> np.ndarray:
    """Repair clipped (saturated) runs via interpolation.

    Detects sustained runs of samples sitting at the signal's maximum or
    minimum value (within a small tolerance) and replaces them by
    linearly interpolating between the nearest non-clipped samples.

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.

    Returns:
        A new array, same shape as the input, with clipped runs smoothed
        out. If the signal has fewer than 2 samples, or every sample is
        part of a clipped run (no valid anchor points remain), the array
        is returned unchanged since there is nothing safe to interpolate
        from.
    """
    flat, original_shape = _flatten(signal)

    n = flat.shape[0]
    if n < 2:
        return _restore_shape(flat, original_shape)

    hi = flat.max()
    lo = flat.min()

    # Samples sitting at (or within tolerance of) the rails.
    at_hi = flat >= (hi - _CLIP_TOL)
    at_lo = flat <= (lo + _CLIP_TOL)

    # Only sustained runs count as "clipping" -- matches the detection
    # logic used upstream in sanity_checks.check_clipping.
    clipped_mask = _find_run_mask(at_hi, _CLIP_RUN_LENGTH) | _find_run_mask(
        at_lo, _CLIP_RUN_LENGTH
    )

    if not clipped_mask.any():
        return _restore_shape(flat, original_shape)

    valid_mask = ~clipped_mask
    if not valid_mask.any():
        # The entire signal is one giant clipped run -- no valid anchors
        # to interpolate from, so leave it as-is (deterministic no-op).
        return _restore_shape(flat, original_shape)

    indices = np.arange(n)
    valid_indices = indices[valid_mask]
    valid_values = flat[valid_mask]

    # Runs touching the start/end of the array are handled by np.interp's
    # boundary clamping (nearest valid sample is used for extrapolation),
    # which is deterministic and introduces no randomness.
    flat[clipped_mask] = np.interp(indices[clipped_mask], valid_indices, valid_values)

    return _restore_shape(flat, original_shape)


def repair_amplitude(signal, z_thresh: float = _Z_THRESH) -> np.ndarray:
    """Clamp amplitude outliers to +/- 6 standard deviations.

    Computes a z-score for every sample relative to the signal's own
    mean and standard deviation. Samples whose |z-score| exceeds
    ``_Z_THRESH`` (6.0) are clamped to exactly ``mean +/- 6 * std``.

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.

    Returns:
        A new array, same shape as the input, with outlier samples
        clamped. If the standard deviation is (near) zero -- i.e. the
        signal is effectively flat -- there is no meaningful z-score to
        compute, so the array is returned unchanged.
    """
    flat, original_shape = _flatten(signal)

    mu = flat.mean()
    sigma = flat.std()

    if sigma < _STD_EPS:
        # Effectively flat signal: z-scores would be undefined/unstable,
        # and there's nothing to clamp toward.
        return _restore_shape(flat, original_shape)

    z = (flat - mu) / sigma

    upper_bound = mu + z_thresh * sigma
    lower_bound = mu - z_thresh * sigma

    high_outliers = z > z_thresh
    low_outliers = z < -z_thresh

    flat[high_outliers] = upper_bound
    flat[low_outliers] = lower_bound

    return _restore_shape(flat, original_shape)


def repair_signal(signal, flags: Dict, amplitude_z_thresh: float = _Z_THRESH) -> Tuple[np.ndarray, List[str]]:
    """Apply the repairs indicated by ``flags`` to a copy of ``signal``.

    This is the single entry point ``validation_agent.py`` should call
    for signals whose triage status is ``REPAIR``. It never re-validates
    the result -- that responsibility belongs entirely to
    ``validation_agent.py`` after this function returns.

    Args:
        signal: Array-like ECG signal, shape ``(360,)`` or ``(360, 1)``.
        flags: The same flags dict produced upstream (e.g. by
            format_checks/sanity_checks and consumed by triage.decide),
            expected to use the keys ``'nan'``, ``'clipping'``, and
            ``'abnormal_amplitude'``. Missing keys are treated as False.

    Returns:
        A tuple ``(repaired_signal, actions)`` where:
            - ``repaired_signal`` is a new array (the original input is
              never mutated), same shape as the input.
            - ``actions`` is an ordered list of strings describing which
              repairs were applied, e.g.
              ``["nan_interpolated", "clip_smoothed", "amplitude_clamped"]``.
    """
    # Defensive copy up front: every helper below already returns a new
    # array, but we start from an explicit copy of the input so that
    # `working` never aliases the caller's original array, even before
    # the first repair step runs.
    arr = np.asarray(signal, dtype=np.float64).copy()

    actions: List[str] = []
    working = arr

    # Order matters: interpolate missing data first, then smooth clipped
    # rails, then clamp outliers -- each step operates on a cleaner
    # signal than the one before it.
    if flags.get('nan'):
        working = repair_nan(working)
        actions.append('nan_interpolated')

    if flags.get('clipping'):
        working = repair_clipping(working)
        actions.append('clip_smoothed')

    if flags.get('abnormal_amplitude'):
        working = repair_amplitude(working, amplitude_z_thresh)
        actions.append('amplitude_clamped')

    return working, actions
