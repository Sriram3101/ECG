# agents/validation/validation_agent.py — temporary stub
from core.ecg_state import ValidationResult, empty_validation_flags

def validate_beat(signal, metadata=None) -> ValidationResult:
    """Placeholder — always returns PASS. Replace with real logic."""
    return ValidationResult(
        status="PASS",
        signal=signal,
        flags=empty_validation_flags(),
        reason="stub - not yet implemented",
        repair_actions=[],
    )
