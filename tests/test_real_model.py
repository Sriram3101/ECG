"""Integration tests against the team's real checkpoint and data.

Skipped unless artifacts/DenseNet1D_clean.keras and artifacts/test.csv (or
its .npz cache) are present and TensorFlow is installed.
"""
import numpy as np
import pytest

from evaluation.data import ARTIFACTS_DIR

MODEL = ARTIFACTS_DIR / "DenseNet1D_clean.keras"
HAS_DATA = (ARTIFACTS_DIR / "test.csv").exists() or (ARTIFACTS_DIR / "test.npz").exists()

pytestmark = pytest.mark.skipif(not (MODEL.exists() and HAS_DATA), reason="real model/data not in artifacts/")


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("tensorflow")
    from models.backend import KerasBackend
    return KerasBackend.load(str(MODEL))


@pytest.fixture(scope="module")
def test_split():
    from evaluation.data import load_split
    return load_split("test")


def test_keras_backend_matches_model_contract(backend, test_split):
    x, y = test_split.signals[:500], test_split.labels[:500]
    probs = backend.predict_proba(x)
    assert probs.shape == (500, 4)
    assert (probs.argmax(axis=1) == y).mean() > 0.95
    assert backend.embed(x[:3]).shape == (3, 128)
    attention = backend.attention(x[:2])
    assert attention.shape == (2, 360) and np.allclose(attention.sum(axis=1), 1.0)
    assert backend.input_gradient(x[:2], probs[:2].argmax(axis=1)).shape == (2, 360)


def test_real_beats_pass_validation_with_zscore_preset(test_split):
    from agents.validation.validation_agent import ValidationConfig, validate_beat
    cfg = ValidationConfig.zscored_beats()
    statuses = [validate_beat(s, config=cfg)["status"] for s in test_split.signals[:300]]
    assert statuses.count("PASS") == len(statuses)


def test_full_pipeline_on_real_beat(backend, test_split):
    from evaluation.data import stratified_sample
    from orchestrator.pipeline import ECGDefensePipeline

    idx = stratified_sample(test_split, 600, seed=0)
    ref, audit = idx[:300], idx[300:]
    pipeline = ECGDefensePipeline.from_reference_data(
        backend, clean_signals=test_split.signals[ref], audit_signals=test_split.signals[audit],
        audit_labels=test_split.labels[audit], audit_record_ids=[test_split.beat_ids[i] for i in audit],
        shap_permutations=16)
    i = int(audit[0])
    state = pipeline.run(test_split.signals[i], test_split.beat_ids[i], label=int(test_split.labels[i]))
    assert state["report"]["alert_level"] in ("INFO", "WARNING", "CRITICAL")
    assert state["decision"] is not None
    assert "snr_db" in state["report"]["signal_quality"]
