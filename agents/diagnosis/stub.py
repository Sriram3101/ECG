# agents/diagnosis/stub.py
"""
Stub diagnosis agent — returns hardcoded output in the correct schema.
Used by Decision Agent and Explainability Agent teams for development
until the real checkpoint is plugged in.
"""

from core.ecg_state import ECGState

CLASS_LABELS = {0: "A", 1: "B", 2: "N", 3: "V"}

def run_diagnosis_stub(state: ECGState) -> ECGState:
    """Always predicts N (Normal) with 91% confidence — matches baseline accuracy."""
    state["diagnosis"] = {
        "predicted_class":     2,
        "predicted_label":     "N",
        "confidence":          0.91,
        "class_probabilities": {"A": 0.02, "B": 0.03, "N": 0.91, "V": 0.04},
        "error":               None,
    }
    return state
