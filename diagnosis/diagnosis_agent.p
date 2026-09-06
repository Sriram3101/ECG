# agents/diagnosis/diagnosis_agent.py
"""
Diagnosis Agent — wraps the trained DenseNet1D + BiLSTM + Attention
checkpoint and runs inference on a validated (and optionally recovered)
ECG beat.

Input:  validation["signal"] from ECGState  (shape 360,1)
Output: fills ECGState["diagnosis"] with prediction, confidence, per-class probs
"""

import numpy as np
import torch
from core.ecg_state import ECGState

# Class index → human-readable label (MIT-BIH 5-class)
CLASS_LABELS = {0: "A", 1: "B", 2: "N", 3: "V"}

_model = None   # loaded lazily on first call

def _load_model(checkpoint_path: str):
    """Load the trained model once and cache it."""
    global _model
    if _model is not None:
        return _model
    # Import here to avoid circular imports at module load time
    from models.densenet_bilstm import DenseNet1DBiLSTMAttention
    _model = DenseNet1DBiLSTMAttention(num_classes=4)
    state = torch.load(checkpoint_path, map_location="cpu")
    _model.load_state_dict(state)
    _model.eval()
    return _model


def run_diagnosis(state: ECGState,
                  checkpoint_path: str = "models/best_model.pth") -> ECGState:
    """
    Run the classifier on the validated/recovered signal.

    Reads:  state["validation"]["signal"]
    Writes: state["diagnosis"]

    Returns the updated ECGState.
    """
    # Get signal — prefer repaired signal if available
    signal = state.get("validation", {}).get("signal")
    if signal is None:
        state["diagnosis"] = {
            "error": "No signal available — validation rejected this beat",
            "predicted_class": None,
            "predicted_label": None,
            "confidence": None,
            "class_probabilities": None,
        }
        return state

    # Prepare tensor: (1, 1, 360) — batch=1, channel=1, length=360
    arr = np.asarray(signal, dtype=np.float32).flatten()
    x   = torch.tensor(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, 360)

    model = _load_model(checkpoint_path)

    with torch.no_grad():
        logits = model(x)                          # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)      # (1, num_classes)
        pred   = int(probs.argmax(dim=1).item())
        conf   = float(probs[0, pred].item())

    state["diagnosis"] = {
        "predicted_class":      pred,
        "predicted_label":      CLASS_LABELS.get(pred, str(pred)),
        "confidence":           round(conf, 4),
        "class_probabilities":  {
            CLASS_LABELS[i]: round(float(probs[0, i].item()), 4)
            for i in range(probs.shape[1])
        },
        "error": None,
    }
    return state
