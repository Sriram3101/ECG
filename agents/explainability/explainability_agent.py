# agents/explainability/explainability_agent.py
"""
Explainability Agent — attention maps and SHAP values for the diagnosis
prediction.

Input:  state["diagnosis"] + the signal that was diagnosed
        (state["active_signal"], falling back to state["validation"]["signal"])
Output: fills state["explainability"] with
          attention_weights     (360,) BiLSTM attention, resampled to input length
          gradient_sensitivity  (360,) |d p(pred) / d x|, normalized to [0, 1]
          shap_values           Shapley value of each 30-sample segment for
                                the predicted class (sums to p(pred | beat) -
                                p(pred | background beat))
          top_regions           the 3 segments that pushed hardest toward
                                the prediction
"""

import numpy as np

from core.ecg_state import ECGState
from models.backend import ClassifierBackend

FS = 360            # MIT-BIH sampling rate (Hz)
SEGMENT = 30        # samples per explained segment


def _segment_shapley(backend: ClassifierBackend, signal: np.ndarray, background: np.ndarray,
                     predicted_class: int, n_permutations: int = 64, seed: int = 0) -> np.ndarray:
    """Permutation-sampling estimate of segment-level Shapley values.

    Each 30-sample segment is a "player"; a coalition keeps its segments
    from the beat and replaces the rest with the background beat (mean of
    clean beats). Exact Shapley over 12 players needs 4096 forward passes;
    sampling permutations gives an unbiased estimate at a fraction of that.
    """
    n_segments = len(signal) // SEGMENT
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(n_segments) for _ in range(n_permutations)]
    batch = []
    for order in orders:
        current = background.copy()
        batch.append(current.copy())
        for seg in order:
            current[seg * SEGMENT:(seg + 1) * SEGMENT] = signal[seg * SEGMENT:(seg + 1) * SEGMENT]
            batch.append(current.copy())
    # One forward pass for every coalition of every permutation.
    probs = backend.predict_proba(np.stack(batch))[:, predicted_class].reshape(n_permutations, n_segments + 1)
    contributions = np.zeros(n_segments)
    for order, path in zip(orders, probs):
        contributions[order] += np.diff(path)
    return contributions / n_permutations


def _segment_label(i: int) -> str:
    start_ms = round(i * SEGMENT * 1000 / FS)
    end_ms = round((i + 1) * SEGMENT * 1000 / FS)
    return f"samples {i * SEGMENT}–{(i + 1) * SEGMENT} ({start_ms}–{end_ms} ms)"


def run_explainability(state: ECGState, backend: ClassifierBackend,
                       background: np.ndarray | None = None, n_permutations: int = 64) -> ECGState:
    """
    Generate attention map, gradient saliency and segment SHAP values for
    the diagnosis prediction.

    `background` is the reference beat SHAP measures contributions against;
    pass the mean of clean training beats. Defaults to a flat beat at the
    signal's own mean.
    """
    diagnosis = state.get("diagnosis") or {}
    signal = state.get("active_signal")
    if signal is None:
        signal = (state.get("validation") or {}).get("signal")

    # Guard: skip if diagnosis failed or beat was rejected
    if diagnosis.get("error") or signal is None or \
       diagnosis.get("predicted_class") is None:
        state["explainability"] = {
            "error": "Skipped — no valid diagnosis to explain",
            "attention_weights": None,
            "gradient_sensitivity": None,
            "shap_values": None,
            "top_regions": None,
        }
        return state

    arr = np.asarray(signal, dtype=np.float32).flatten()
    pred = int(diagnosis["predicted_class"])
    if background is None:
        background = np.full_like(arr, arr.mean())
    background = np.asarray(background, dtype=np.float32).flatten()

    attention = backend.attention(arr)
    attention = attention[0] if attention is not None else np.full(len(arr), 1.0 / len(arr))

    saliency = np.abs(backend.input_gradient(arr, np.array([pred]))[0])
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-9)

    shap_values = _segment_shapley(backend, arr, background, pred, n_permutations)
    top = np.argsort(-shap_values)[:3]

    state["explainability"] = {
        "attention_weights":    attention.round(6).tolist(),
        "gradient_sensitivity": saliency.round(6).tolist(),
        "shap_values":          shap_values.round(6).tolist(),
        "top_regions": [
            {"region": _segment_label(int(i)),
             "shap": round(float(shap_values[i]), 4),
             "attention": round(float(attention[i * SEGMENT:(i + 1) * SEGMENT].sum()), 4)}
            for i in top
        ],
        "predicted_label": diagnosis.get("predicted_label"),
        "confidence":      diagnosis.get("confidence"),
        "error":           None,
    }
    return state
