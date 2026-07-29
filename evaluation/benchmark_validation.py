# evaluation/benchmark_validation.py
import numpy as np
import wfdb
from collections import Counter
from generate_invalid_data import inject_nan, inject_flatline, inject_clipping
from metrics import detection_metrics, validation_accuracy
from agents.validation.validation_agent import validate_beat

DATA_DIR = "./data/mitbih"
RECORDS  = ['100', '101', '103', '105']

# ── Load clean beats ─────────────────────────────────────────────────────────
def load_clean_beats(n=200):
    beats = []
    for rec in RECORDS:
        record = wfdb.rdrecord(f"{DATA_DIR}/{rec}")
        ann    = wfdb.rdann(f"{DATA_DIR}/{rec}", "atr")
        sig    = record.p_signal[:, 0]
        for r_peak in ann.sample:
            if r_peak - 180 < 0 or r_peak + 180 >= len(sig): continue
            beats.append(sig[r_peak-180:r_peak+180].reshape(360, 1))
            if len(beats) >= n: break
        if len(beats) >= n: break
    return beats

clean_beats = load_clean_beats()

# ── Triage rules (match Validation Agent exactly) ────────────────────────────
def expected_status_from_flags(flags: dict, nan_ratio: float = 0.0) -> str:
    """
    Mirror the triage logic in agents/validation/triage.py.
    Used to auto-generate expected_status per scenario.
    """
    if flags.get("wrong_shape") or flags.get("bad_dtype"):
        return "REJECT"
    if flags.get("nan"):
        return "REJECT" if nan_ratio >= 0.05 else "REPAIR"
    if flags.get("flatline"):
        return "REJECT"
    if flags.get("clipping") or flags.get("abnormal_amplitude"):
        return "REPAIR"
    return "PASS"

# ── Scenarios — uses ValidationFlags from core/ecg_state.py ─────────────────
# No invalid_entries flag — removed per schema (ecg_state.py)
scenarios = [
    {
        "name":     "clean",
        "injector": None,
        "flags_hint": {
            "wrong_shape": False, "bad_dtype": False, "nan": False,
            "missing_samples": False, "flatline": False,
            "clipping": False, "abnormal_amplitude": False
        },
        "nan_ratio": 0.0,
    },
    {
        "name":     "nan_sparse",       # < 5% NaN → REPAIR
        "injector": lambda b: inject_nan(b, n_samples=5),
        "flags_hint": {"nan": True},
        "nan_ratio": 5/360,             # ~1.4% → REPAIR
    },
    {
        "name":     "nan_heavy",        # ≥ 5% NaN → REJECT
        "injector": lambda b: inject_nan(b, n_samples=20),
        "flags_hint": {"nan": True},
        "nan_ratio": 20/360,            # ~5.6% → REJECT
    },
    {
        "name":     "flatline",         # → REJECT
        "injector": lambda b: inject_flatline(b, start=100, length=80),
        "flags_hint": {"flatline": True},
        "nan_ratio": 0.0,
    },
    {
        "name":     "clipping",         # → REPAIR
        "injector": lambda b: inject_clipping(b),
        "flags_hint": {"clipping": True},
        "nan_ratio": 0.0,
    },
]

# ── Run benchmark ─────────────────────────────────────────────────────────────
all_results = []

for scenario in scenarios:
    for beat in clean_beats[:50]:
        if scenario["injector"]:
            corrupted, mask, _ = scenario["injector"](beat)
        else:
            corrupted = beat

        result = validate_beat(corrupted)

        # Change 4: downstream agents must receive validation["signal"],
        # not raw_signal — verify this here
        signal_for_downstream = result.get("signal")
        assert signal_for_downstream is not None, \
            f"[{scenario['name']}] validation['signal'] is None — downstream agents will break"

        # Expected status derived from triage rules, not hardcoded
        expected = expected_status_from_flags(
            scenario["flags_hint"], scenario["nan_ratio"]
        )

        all_results.append({
            "corruption":  scenario["name"],
            "predicted":   result["status"],
            "expected":    expected,
            "flags":       result["flags"],
            "reason":      result["reason"],
            "signal_ok":   signal_for_downstream is not None,
        })

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nTriage Accuracy by Scenario:")
print(f"{'Scenario':<15} {'Expected':<10} {'Accuracy':>10}  Predictions")
for scenario in scenarios:
    name   = scenario["name"]
    subset = [r for r in all_results if r["corruption"] == name]
    acc    = validation_accuracy(subset)
    counts = Counter(r["predicted"] for r in subset)
    expected = expected_status_from_flags(scenario["flags_hint"], scenario["nan_ratio"])
    print(f"  {name:<13} {expected:<10} {acc:>9.1f}%  {dict(counts)}")

print("\nSignal handoff check (all should be True):")
for scenario in scenarios:
    name   = scenario["name"]
    subset = [r for r in all_results if r["corruption"] == name]
    ok     = all(r["signal_ok"] for r in subset)
    print(f"  {name:<15} signal_ok={ok}")
