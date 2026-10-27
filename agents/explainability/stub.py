# agents/explainability/stub.py
"""
Stub explainability agent — returns hardcoded output in the correct schema.
Used for parallel development until real checkpoint is available.
"""

from core.ecg_state import ECGState


def run_explainability_stub(state: ECGState) -> ECGState:
    """Returns uniform attention weights and flat saliency — correct shape, fake values."""
    state["explainability"] = {
        "attention_weights":    [round(1/360, 6)] * 360,
        "gradient_sensitivity": [0.0] * 360,
        "shap_values":          [0.0] * 12,
        "top_regions": [
            {"region": "samples 0–30 (0–83 ms)",       "shap": 0.0, "attention": 0.0833},
            {"region": "samples 150–180 (417–500 ms)", "shap": 0.0, "attention": 0.0833},
            {"region": "samples 300–330 (833–917 ms)", "shap": 0.0, "attention": 0.0833},
        ],
        "predicted_label": (state.get("diagnosis") or {}).get("predicted_label", "N"),
        "confidence":      (state.get("diagnosis") or {}).get("confidence", 0.91),
        "error":           None,
    }
    return state
