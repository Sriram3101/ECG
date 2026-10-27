# evaluation/benchmark_validation.py
"""
Benchmark harness for the ECG Validation Agent.

Runs the full validation pipeline (format checks → sanity checks →
triage → repair) against synthetically corrupted beats and measures
triage accuracy, repair quality, and signal handoff correctness.
"""

import numpy as np
import wfdb
from collections import Counter
from generate_invalid_data import inject_nan, inject_flatline, inject_clipping
from metrics import detection_metrics, validation_accuracy
from agents.validation.validation_agent import validate_beat
from agents.validation.repair import repair_signal

DATA_DIR = "./data/mitbih"
RECORDS  = ['100', '101', '103', '105']

# ── Load clean beats ──────────────────────────────────────────────────────────
def load_clean_beats(n=200):
    beats = []
    for rec in RECORDS:
        try:
            record = wfdb.rdrecord(f"{DATA_DIR}/{rec}")
            ann    = wfdb.rdann(f"{DATA_DIR}/{rec}", "atr")
            sig    = record.p_signal[:, 0]
            for r_peak in ann.sample:
                if r_peak - 180 < 0 or r_peak + 180 >= len(sig):
                    continue
                beats.append(sig[r_peak-180:r_peak+180].reshape(360, 1))
                if len(beats) >= n:
                    break
        except Exception as e:
            print(f"  Warning: could not load record {rec} — {e}")
        if len(beats) >= n:
            break
    return beats

# ── Triage rules (mirrors triage.py exactly) ──────────────────────────────────
def expected_status(flags: dict, nan_ratio: float = 0.0) -> str:
    """
    Must stay in sync with agents/validation/triage.py decide().
    NaN < 5%  → REPAIR
    NaN ≥ 5%  → REJECT
    Flatline  → REJECT
    Clipping  → REPAIR
    Abnormal amplitude → REPAIR
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

# ── Repair quality check ───────────────────────────────────────────────────────
def check_repair_quality(original, repaired) -> dict:
    """
    Verifies that repair_signal() produced a valid, usable signal.
    Checks:
      - No NaNs remain
      - Shape is preserved
      - Signal is not all zeros (degenerate fallback)
      - Signal actually changed (repair did something)
    """
    orig = np.asarray(original, dtype=float).flatten()
    rep  = np.asarray(repaired, dtype=float).flatten()
    return {
        "no_nan":          not np.isnan(rep).any(),
        "shape_preserved": rep.shape == orig.shape,
        "not_all_zeros":   not np.allclose(rep, 0.0),
        "signal_changed":  not np.allclose(orig, rep),
    }

# ── Scenarios ─────────────────────────────────────────────────────────────────
# ValidationFlags keys from core/ecg_state.py:
#   wrong_shape, bad_dtype, nan, missing_samples,
#   flatline, clipping, abnormal_amplitude
# Note: invalid_entries is NOT in the schema and is not used here.

scenarios = [
    {
        "name":      "clean",
        "injector":  None,
        "flags_hint": {
            "wrong_shape": False, "bad_dtype": False, "nan": False,
            "missing_samples": False, "flatline": False,
            "clipping": False, "abnormal_amplitude": False,
        },
        "nan_ratio": 0.0,
        "repair_flags": {},
    },
    {
        "name":      "nan_sparse",       # ~1.4% NaN → REPAIR
        "injector":  lambda b: inject_nan(b, n_samples=5),
        "flags_hint": {"nan": True},
        "nan_ratio": 5 / 360,
        "repair_flags": {"nan": True},
    },
    {
        "name":      "nan_heavy",        # ~5.6% NaN → REJECT
        "injector":  lambda b: inject_nan(b, n_samples=20),
        "flags_hint": {"nan": True},
        "nan_ratio": 20 / 360,
        "repair_flags": {"nan": True},
    },
    {
        "name":      "flatline",         # → REJECT
        "injector":  lambda b: inject_flatline(b, start=100, length=80),
        "flags_hint": {"flatline": True},
        "nan_ratio": 0.0,
        "repair_flags": {},
    },
    {
        "name":      "clipping",         # → REPAIR
        "injector":  lambda b: inject_clipping(b),
        "flags_hint": {"clipping": True},
        "nan_ratio": 0.0,
        "repair_flags": {"clipping": True},
    },
]

# ── Run benchmark ─────────────────────────────────────────────────────────────
print("Loading clean beats...")
clean_beats = load_clean_beats()
print(f"Loaded {len(clean_beats)} beats.\n")

all_results = []

for scenario in scenarios:
    exp = expected_status(scenario["flags_hint"], scenario["nan_ratio"])

    for beat in clean_beats[:50]:
        # 1. Inject corruption
        if scenario["injector"]:
            corrupted, mask, _ = scenario["injector"](beat)
        else:
            corrupted = beat

        # 2. Run validation agent
        result = validate_beat(corrupted)

        # 3. Check 4: downstream agents must receive validation["signal"],
        #    not raw_signal
        signal_for_downstream = result.get("signal")
        assert signal_for_downstream is not None, (
            f"[{scenario['name']}] validation['signal'] is None — "
            "downstream agents (poison/noise) will break"
        )

        # 4. For REPAIR beats: run repair_signal and verify quality
        repair_quality = None
        if result["status"] == "REPAIR" and scenario["repair_flags"]:
            repaired, actions = repair_signal(
                signal_for_downstream, scenario["repair_flags"]
            )
            repair_quality = check_repair_quality(corrupted, repaired)
            # downstream receives the repaired signal, not the raw corrupted one
            signal_for_downstream = repaired

        all_results.append({
            "corruption":     scenario["name"],
            "predicted":      result["status"],
            "expected":       exp,
            "flags":          result["flags"],
            "reason":         result["reason"],
            "repair_actions": result.get("repair_actions", []),
            "repair_quality": repair_quality,
            "signal_ok":      signal_for_downstream is not None,
        })

# ── Print results ─────────────────────────────────────────────────────────────
print("=" * 60)
print("TRIAGE ACCURACY BY SCENARIO")
print("=" * 60)
print(f"{'Scenario':<15} {'Expected':<10} {'Accuracy':>10}  Predictions")
for scenario in scenarios:
    name   = scenario["name"]
    exp    = expected_status(scenario["flags_hint"], scenario["nan_ratio"])
    subset = [r for r in all_results if r["corruption"] == name]
    acc    = validation_accuracy(subset)
    counts = Counter(r["predicted"] for r in subset)
    print(f"  {name:<13} {exp:<10} {acc:>9.1f}%  {dict(counts)}")

print("\n" + "=" * 60)
print("SIGNAL HANDOFF CHECK (all should be True)")
print("=" * 60)
for scenario in scenarios:
    name   = scenario["name"]
    subset = [r for r in all_results if r["corruption"] == name]
    ok     = all(r["signal_ok"] for r in subset)
    print(f"  {name:<15} signal_ok={ok}")

print("\n" + "=" * 60)
print("REPAIR QUALITY CHECK")
print("=" * 60)
repair_scenarios = [s["name"] for s in scenarios if s["repair_flags"]]
for name in repair_scenarios:
    subset = [r for r in all_results
              if r["corruption"] == name and r["repair_quality"] is not None]
    if not subset:
        print(f"  {name:<15} no repair results (triage may not have returned REPAIR)")
        continue
    for check in ["no_nan", "shape_preserved", "not_all_zeros", "signal_changed"]:
        passed = sum(r["repair_quality"][check] for r in subset)
        total  = len(subset)
        status = "✓" if passed == total else "✗"
        print(f"  {name:<15} {check:<20} {status} {passed}/{total}")

print("\n" + "=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
total   = len(all_results)
correct = sum(r["predicted"] == r["expected"] for r in all_results)
print(f"  Total beats tested : {total}")
print(f"  Correct triage     : {correct}/{total} ({correct/total*100:.1f}%)")
print(f"  Signal handoff OK  : {sum(r['signal_ok'] for r in all_results)}/{total}")
