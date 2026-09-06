# agents/explainability/stub.py
"""
Stub explainability agent — returns hardcoded output in the correct schema.
Used for parallel development until real checkpoint is available.
"""

import numpy as np
from core.ecg_state import ECGState


def run_explainability_stub(state: ECGState) -> ECGState:
    """Returns uniform attention weights and flat saliency — correct shape, fake values."""
    state["explainability"] = {
        "attention_weights":    [round(1/360, 6)] * 360,
        "gradient_sensitivity": [0.0] * 360,
        "top_regions": [
            {"region": "0–30ms",   "importance": 0.33},
            {"region": "150–180ms","importance": 0.33},
            {"region": "300–330ms","importance": 0.33},
        ],
        "predicted_label": state.get("diagnosis", {}).get("predicted_label", "N"),
        "confidence":      state.get("diagnosis", {}).get("confidence", 0.91),
        "error":           None,
    }
    return state
