import pytest

from agents.recovery.strategies.label_repair import repair_label
from agents.recovery.tests.conftest import make_record
from shared.contracts import RecoveryAction


def test_relabels_with_high_confidence_consensus():
    record = make_record(label=3)  # original (suspected poisoned) label
    oof = [0.02, 0.90, 0.05, 0.02, 0.01]  # consensus strongly favors class 1

    result = repair_label(record, oof, min_confidence=0.7)

    assert result.action == RecoveryAction.RELABEL
    assert result.repaired_record.label == 1
    assert result.repaired_record.source == "repaired"
    assert result.repair_confidence == pytest.approx(0.90)
    assert "relabeled 3 -> 1" in result.notes


def test_quarantines_when_consensus_confidence_too_low():
    record = make_record(label=3)
    oof = [0.25, 0.25, 0.25, 0.15, 0.10]  # no clear winner

    result = repair_label(record, oof, min_confidence=0.7)

    assert result.action == RecoveryAction.QUARANTINE
    assert result.repaired_record is None
    assert "below min_confidence" in result.notes


def test_notes_flag_when_consensus_matches_original_label():
    record = make_record(label=1)
    oof = [0.02, 0.90, 0.05, 0.02, 0.01]  # consensus agrees with original label

    result = repair_label(record, oof, min_confidence=0.7)

    assert result.action == RecoveryAction.RELABEL
    assert result.repaired_record.label == 1
    assert "matches original" in result.notes


def test_original_record_is_not_mutated():
    record = make_record(label=3)
    oof = [0.02, 0.90, 0.05, 0.02, 0.01]

    repair_label(record, oof, min_confidence=0.7)

    assert record.label == 3  # unchanged


@pytest.mark.parametrize(
    "bad_probs",
    [
        [0.5, 0.6],          # sums to > 1
        [0.1, 0.1],          # sums to < 1
        [-0.1, 1.1],         # out of [0,1] range
        [[0.1, 0.9]],        # wrong ndim
    ],
)
def test_invalid_probabilities_raise(bad_probs):
    record = make_record()
    with pytest.raises(ValueError):
        repair_label(record, bad_probs)
