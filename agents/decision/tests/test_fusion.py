import pytest

from agents.decision.fusion import (
    DEFAULT_SIGNAL_CHECK_WEIGHTS,
    fuse_signal_evidence,
    fuse_signal_evidence_noisy_or,
)
from agents.decision.tests.conftest import make_signal_evidence


def test_fuse_all_zero_is_zero():
    evidence = make_signal_evidence({
        "reconstruction_error": 0.0,
        "feature_space_distance": 0.0,
        "prediction_fragility": 0.0,
        "gradient_sensitivity": 0.0,
    })
    assert fuse_signal_evidence(evidence) == pytest.approx(0.0)


def test_fuse_all_one_is_one():
    evidence = make_signal_evidence({
        "reconstruction_error": 1.0,
        "feature_space_distance": 1.0,
        "prediction_fragility": 1.0,
        "gradient_sensitivity": 1.0,
    })
    assert fuse_signal_evidence(evidence) == pytest.approx(1.0)


def test_fuse_respects_weights():
    # reconstruction_error has the highest default weight (0.30) — a
    # spike there should move the fused score more than an equal-sized
    # spike in prediction_fragility (weight 0.20).
    evidence_recon = make_signal_evidence({"reconstruction_error": 0.9})
    evidence_fragility = make_signal_evidence({"prediction_fragility": 0.9})
    assert fuse_signal_evidence(evidence_recon) > fuse_signal_evidence(evidence_fragility)


def test_fuse_missing_check_raises():
    incomplete = [
        e for e in make_signal_evidence({}) if e.check_name != "gradient_sensitivity"
    ]
    with pytest.raises(ValueError, match="missing required signal-poisoning checks"):
        fuse_signal_evidence(incomplete)


def test_fuse_duplicate_check_raises():
    evidence = make_signal_evidence({}) + [make_signal_evidence({})[0]]
    with pytest.raises(ValueError, match="duplicate signal evidence"):
        fuse_signal_evidence(evidence)


def test_fuse_ignores_non_signal_evidence():
    from shared.contracts import Evidence

    evidence = make_signal_evidence({}) + [
        Evidence(check_name="aum_margin", channel="label", score=0.99)
    ]
    # Should not raise, and the label-channel entry should not affect
    # the fused signal score (all signal checks default to 0.05 here).
    assert fuse_signal_evidence(evidence) == pytest.approx(0.05)


def test_noisy_or_single_strong_check_dominates():
    evidence = make_signal_evidence({"reconstruction_error": 0.95})
    weighted_mean = fuse_signal_evidence(evidence)
    noisy_or = fuse_signal_evidence_noisy_or(evidence)
    # noisy-OR should score this at least as high as the weighted mean,
    # since one strong check shouldn't get averaged down by three calm ones.
    assert noisy_or >= weighted_mean


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_SIGNAL_CHECK_WEIGHTS.values()) == pytest.approx(1.0)
