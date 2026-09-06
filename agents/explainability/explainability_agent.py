# agents/explainability/explainability_agent.py
"""
Explainability Agent — generates attention maps and SHAP values
for the diagnosis prediction.

Input:  state["diagnosis"] + state["validation"]["signal"]
Output: fills state["explainability"] with attention weights and
        feature importance scores
"""

import numpy as np
import torch
from core.ecg_state import ECGState

_model = None

def _load_model(checkpoint_path: str):
    global _model
    if _model is not None:
        return _model
    from models.densenet_bilstm import DenseNet1DBiLSTMAttention
    _model = DenseNet1DBiLSTMAttention(num_classes=4)
    state = torch.load(checkpoint_path, map_location="cpu")
    _model.load_state_dict(state)
    _model.eval()
    return _model


def _get_attention_weights(model, x: torch.Tensor) -> np.ndarray:
    """
    Extract BiLSTM attention weights for the input beat.
    Returns array of shape (360,) — one weight per timestep.
    Weights sum to 1.0 (softmax output of attention layer).
    """
    attention_weights = []

    def hook_fn(module, input, output):
        # Attention layer returns (context, weights) — grab weights
        if isinstance(output, tuple) and len(output) == 2:
            attention_weights.append(output[1].detach().cpu().numpy())

    # Register hook on the attention layer
    hooks = []
    for name, module in model.named_modules():
        if "attention" in name.lower():
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    if attention_weights:
        # Shape: (1, timesteps, 1) → flatten to (timesteps,)
        weights = attention_weights[0].squeeze()
        # Resize to 360 if needed (LSTM may compress sequence length)
        if weights.shape[0] != 360:
            weights = np.interp(
                np.linspace(0, 1, 360),
                np.linspace(0, 1, weights.shape[0]),
                weights
            )
        return weights
    else:
        # Fallback: uniform weights if hook didn't fire
        return np.ones(360) / 360


def _get_gradient_sensitivity(model, x: torch.Tensor,
                               predicted_class: int) -> np.ndarray:
    """
    Compute input gradient sensitivity (saliency map).
    High values = model is sensitive to that part of the signal.
    Returns array of shape (360,).
    """
    x_grad = x.clone().requires_grad_(True)
    logits = model(x_grad)
    score  = logits[0, predicted_class]
    score.backward()
    saliency = x_grad.grad.abs().squeeze().numpy()  # (360,)
    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-9)
    return saliency


def run_explainability(state: ECGState,
                       checkpoint_path: str = "models/best_model.pth") -> ECGState:
    """
    Generate attention map and gradient saliency for the diagnosis prediction.

    Reads:  state["validation"]["signal"], state["diagnosis"]
    Writes: state["explainability"]
    """
    diagnosis = state.get("diagnosis", {})
    signal    = state.get("validation", {}).get("signal")

    # Guard: skip if diagnosis failed or beat was rejected
    if diagnosis.get("error") or signal is None or \
       diagnosis.get("predicted_class") is None:
        state["explainability"] = {
            "error": "Skipped — no valid diagnosis to explain",
            "attention_weights": None,
            "gradient_sensitivity": None,
            "top_regions": None,
        }
        return state

    arr   = np.asarray(signal, dtype=np.float32).flatten()
    x     = torch.tensor(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, 360)
    model = _load_model(checkpoint_path)
    pred  = diagnosis["predicted_class"]

    attention   = _get_attention_weights(model, x)
    sensitivity = _get_gradient_sensitivity(model, x, pred)

    # Top 3 most important 30-sample regions (based on attention)
    region_scores = [
        attention[i:i+30].mean()
        for i in range(0, 360, 30)
    ]
    top_regions = sorted(
        enumerate(region_scores), key=lambda x: -x[1]
    )[:3]
    top_regions = [
        {"region": f"{i*30}–{i*30+30}ms", "importance":
