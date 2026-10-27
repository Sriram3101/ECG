import numpy as np
from agents.signal_poisoning.noise.datasets import add_gaussian_noise
from agents.signal_poisoning.noise.noise_agent import NoiseFeatureAgent
from agents.signal_poisoning.noise.noise_features import MorphologyTemplateBank, extract_noise_features
from agents.validation.validation_agent import validate_beat


def test_features_and_agent_contract():
    clean = np.sin(np.linspace(0, 4 * np.pi, 360))[:, None]
    templates = MorphologyTemplateBank.fit([clean])
    features = extract_noise_features(add_gaussian_noise(clean), templates)
    assert set(features) == {"baseline_power_ratio", "powerline_50hz_ratio", "emg_power_ratio", "wavelet_mad_sigma", "wavelet_residual_kurtosis", "template_correlation"}
    output = NoiseFeatureAgent(templates)({"validation": validate_beat(clean)})
    assert 0.0 <= output["noise_evidence"]["score"] <= 1.0


def test_invalid_shape_is_rejected():
    assert validate_beat(np.zeros((20, 1)))["status"] == "REJECT"
