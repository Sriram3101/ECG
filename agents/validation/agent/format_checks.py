# agents/validation/format_checks.py
import numpy as np

def check_shape(signal, expected_shape=(360, 1)):
    arr = np.asarray(signal)
    ok = arr.shape in (expected_shape, (expected_shape[0],))
    return ok, arr.shape

def check_dtype(signal):
    arr = np.asarray(signal)
    return np.issubdtype(arr.dtype, np.number), arr.dtype

def check_nan(signal):
    arr = np.asarray(signal, dtype=float)
    mask = np.isnan(arr)
    return not mask.any(), mask

def check_missing_samples(signal, expected_len=360):
    arr = np.asarray(signal)
    return arr.shape[0] == expected_len, arr.shape[0]

def check_invalid_entries(signal, valid_range=(-5.0, 5.0)):
    arr = np.asarray(signal, dtype=float)
    inf_mask = np.isinf(arr)
    range_mask = (arr < valid_range[0]) | (arr > valid_range[1])
    return not (inf_mask.any() or range_mask.any()), (inf_mask, range_mask)
