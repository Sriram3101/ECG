# evaluation/generate_invalid_data.py
import numpy as np

# Validation-style corruptions (for testing Agent 1)

def inject_nan(signal, n_samples=5, seed=42):
    """Randomly replace n_samples positions with NaN."""
    rng = np.random.default_rng(seed)
    s = signal.copy().astype(float)
    idx = rng.choice(360, n_samples, replace=False)
    s.flat[idx] = np.nan
    mask = np.zeros(360, dtype=bool)
    mask[idx] = True
    return s, mask, {"corruption_type": "nan", "params": {"n_samples": n_samples}}

def inject_flatline(signal, start=100, length=80):
    """Zero out a sustained window to simulate flatline."""
    s = signal.copy()
    s[start:start+length] = 0.0
    mask = np.zeros(360, dtype=bool)
    mask[start:start+length] = True
    return s, mask, {"corruption_type": "flatline", "params": {"start": start, "length": length}}

def inject_clipping(signal, clip_val=None):
    """Saturate signal at ±clip_val to simulate ADC clipping."""
    s = signal.copy()
    clip_val = clip_val or float(np.abs(s).max())
    clipped = np.clip(s, -clip_val, clip_val)
    # mark where the original exceeded the clip boundary
    mask = (np.abs(signal.flatten()) >= clip_val * 0.98)
    return clipped, mask, {"corruption_type": "clipping", "params": {"clip_val": clip_val}}

# Label noise injections (for Agent 2 benchmarking)

def inject_label_noise(y, noise_pct, mode="random", source_class=None,
                        target_class=None, seed=42):
    """
    Flip labels with ground-truth mask.
    mode='random'   — flip to any other class uniformly
    mode='targeted' — flip source_class → target_class only
    Returns: y_noisy, poison_mask (bool array, True = flipped)
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    classes = np.unique(y)

    if mode == "targeted":
        candidate_idx = np.where(y == source_class)[0]
    else:
        candidate_idx = np.arange(len(y))

    n_flip = int(len(candidate_idx) * noise_pct)
    flip_idx = rng.choice(candidate_idx, n_flip, replace=False)

    for i in flip_idx:
        choices = [c for c in classes if c != y_noisy[i]]
        if mode == "targeted" and target_class is not None:
            y_noisy[i] = target_class
        else:
            y_noisy[i] = rng.choice(choices)

    mask = np.zeros(len(y), dtype=bool)
    mask[flip_idx] = True
    return y_noisy, mask

# Signal noise injections (for Agent 3 benchmarking)

def inject_gaussian_noise(X, sigma, seed=42):
    """Add Gaussian noise to all signal samples."""
    rng = np.random.default_rng(seed)
    X_noisy = X + rng.normal(0, sigma, X.shape)
    mask = np.ones(len(X), dtype=bool)   # all samples are perturbed
    return X_noisy, mask
