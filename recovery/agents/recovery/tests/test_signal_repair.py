import numpy as np
import pytest

from agents.recovery.strategies.signal_repair import repair_signal
from agents.recovery.tests.conftest import make_record, make_signal
from shared.contracts import RecoveryAction


def test_repairs_when_confidence_high():
    original = make_signal(seed=1)
    record = make_record(signal=original)
    # A real repair candidate for a heavily-poisoned signal SHOULD look
    # very different from the original — that's the point. Confidence
    # is what gates trust, not similarity to the input.
    reconstruction = (np.asarray(original) + 5.0).tolist()

    result = repair_signal(record, reconstruction, reconstruction_confidence=0.85, min_confidence=0.6)

    assert result.action == RecoveryAction.RECONSTRUCT_SIGNAL
    assert result.repaired_record.signal == pytest.approx(reconstruction)
    assert result.repaired_record.source == "repaired"
    assert result.repair_confidence == pytest.approx(0.85)


def test_quarantines_when_confidence_too_low():
    original = make_signal(seed=1)
    record = make_record(signal=original)
    reconstruction = (np.asarray(original) + 0.01).tolist()  # even a "close" reconstruction...

    # ...should still be quarantined if the AE itself wasn't confident in it.
    result = repair_signal(record, reconstruction, reconstruction_confidence=0.3, min_confidence=0.6)

    assert result.action == RecoveryAction.QUARANTINE
    assert result.repaired_record is None
    assert "confidence" in result.notes


def test_large_difference_from_original_does_not_block_repair():
    """Regression test: a heavily-poisoned original signal SHOULD produce
    a reconstruction that looks very different from it. The repair must
    not be quarantined just because the reconstruction is far from the
    (poisoned) input — that would defeat the point of repairing it."""
    original = make_signal(seed=1)
    huge_perturbation = (np.asarray(original) + 50.0).tolist()
    record = make_record(signal=huge_perturbation)
    clean_reconstruction = original  # AE recovers something close to the true clean signal

    result = repair_signal(record, clean_reconstruction, reconstruction_confidence=0.9, min_confidence=0.6)

    assert result.action == RecoveryAction.RECONSTRUCT_SIGNAL
    assert result.repaired_record.signal == pytest.approx(clean_reconstruction)


def test_invalid_confidence_raises():
    record = make_record(signal=make_signal(seed=1))
    with pytest.raises(ValueError, match="reconstruction_confidence must be in"):
        repair_signal(record, make_signal(seed=1), reconstruction_confidence=1.5)


def test_shape_mismatch_raises():
    record = make_record(signal=make_signal(seed=1))
    wrong_shape_reconstruction = [0.0] * 100  # not 360

    with pytest.raises(ValueError, match="expected reconstruction shape"):
        repair_signal(record, wrong_shape_reconstruction, reconstruction_confidence=0.9)


def test_nan_in_reconstruction_raises():
    record = make_record(signal=make_signal(seed=1))
    bad_reconstruction = [float("nan")] * 360

    with pytest.raises(ValueError, match="contains NaN"):
        repair_signal(record, bad_reconstruction, reconstruction_confidence=0.9)


def test_original_record_is_not_mutated():
    original = make_signal(seed=1)
    record = make_record(signal=original)
    reconstruction = (np.asarray(original) + 0.01).tolist()

    repair_signal(record, reconstruction, reconstruction_confidence=0.9)

    assert record.signal == pytest.approx(original)
