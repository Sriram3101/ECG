# agents/validation/sanity_checks.py
import numpy as np

def check_flatline(signal, std_threshold=0.01):
    arr = np.asarray(signal, dtype=float).flatten()
    return arr.std() > std_threshold, arr.std()

def check_clipping(signal, run_length=3):
    arr = np.asarray(signal, dtype=float).flatten()
    hi, lo = arr.max(), arr.min()

    def has_run(mask):
        run = 0
        for v in mask:
            run = run + 1 if v else 0
            if run >= run_length:
                return True
        return False

    clipped = has_run(arr >= hi - 1e-6) or has_run(arr <= lo + 1e-6)
    return not clipped, clipped

def check_amplitude(signal, z_thresh=6.0):
    arr = np.asarray(signal, dtype=float).flatten()
    mu, sigma = arr.mean(), arr.std() + 1e-9
    z = np.abs((arr - mu) / sigma)
    return not (z > z_thresh).any(), z.max()
