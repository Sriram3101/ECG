# agents/diagnosis/diagnosis_agent.py
"""
Diagnosis Agent — runs the trained DenseNet1D + BiLSTM + Attention
classifier on a validated (and, if needed, recovered) ECG beat.

The model is reached through models.backend (KerasBackend for the team's
`DenseNet1D_clean.keras` checkpoint, TorchBackend for the PyTorch port), so
this agent doesn't care which framework produced the weights.

Input:  state["active_signal"] — the signal the orchestrator decided is
        safe to diagnose (the Recovery agent's repaired signal if one was
        produced, otherwise the validated signal). Falls back to
        state["validation"]["signal"] when run outside the orchestrator.
Output: fills ECGState["diagnosis"] with prediction, confidence, per-class probs
"""

import numpy as np

from core.ecg_state import CLASS_LABELS, ECGState
from models.backend import ClassifierBackend


def _signal_for_diagnosis(state: ECGState):
    signal = state.get("active_signal")
    if signal is None:
        signal = (state.get("validation") or {}).get("signal")
    return signal


def run_diagnosis(state: ECGState, backend: ClassifierBackend) -> ECGState:
    """
    Run the classifier on the validated/recovered signal.

    Reads:  state["active_signal"] (or state["validation"]["signal"])
    Writes: state["diagnosis"]

    Returns the updated ECGState.
    """
    signal = _signal_for_diagnosis(state)
    if signal is None:
        state["diagnosis"] = {
            "error": "No signal available — beat was rejected or quarantined upstream",
            "predicted_class": None,
            "predicted_label": None,
            "confidence": None,
            "class_probabilities": None,
        }
        return state

    probs = backend.predict_proba(np.asarray(signal, dtype=np.float32).flatten())[0]
    pred = int(probs.argmax())

    state["diagnosis"] = {
        "predicted_class":      pred,
        "predicted_label":      CLASS_LABELS.get(pred, str(pred)),
        "confidence":           round(float(probs[pred]), 4),
        "class_probabilities":  {
            CLASS_LABELS.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probs)
        },
        "error": None,
    }
    return state
