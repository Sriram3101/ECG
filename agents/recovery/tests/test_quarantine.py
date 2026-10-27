from agents.recovery.strategies.quarantine import quarantine, reject
from shared.contracts import RecoveryAction


def test_quarantine_sets_correct_action_and_notes():
    result = quarantine("rec_1", "some reason")
    assert result.action == RecoveryAction.QUARANTINE
    assert result.notes == "some reason"
    assert result.repaired_record is None
    assert result.record_id == "rec_1"


def test_reject_sets_correct_action_and_notes():
    result = reject("rec_2", "unusable")
    assert result.action == RecoveryAction.REJECT
    assert result.notes == "unusable"
    assert result.repaired_record is None
