import numpy as np
from ecg_defense.datasets import add_gaussian_noise
from ecg_defense.noise_features import MorphologyTemplateBank, extract_noise_features
from ecg_defense.pipeline import build_ecg_defense_graph, initial_state


def test_features_and_graph_contract():
    clean = np.sin(np.linspace(0, 4 * np.pi, 360))[:, None]
    templates = MorphologyTemplateBank.fit([clean])
    features = extract_noise_features(add_gaussian_noise(clean), templates)
    assert set(features) == {"baseline_power_ratio", "powerline_50hz_ratio", "emg_power_ratio", "wavelet_mad_sigma", "wavelet_residual_kurtosis", "template_correlation"}
    output = build_ecg_defense_graph().invoke(initial_state("beat-1", clean))
    assert 0.0 <= output["noise_evidence"]["score"] <= 1.0


def test_invalid_shape_is_rejected():
    output = build_ecg_defense_graph().invoke(initial_state("bad", np.zeros((20, 1))))
    assert output["decision"]["action"] == "REJECT"
