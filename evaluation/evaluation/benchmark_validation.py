# evaluation/benchmark_validation.py
import numpy as np
import wfdb
from generate_invalid_data import inject_nan, inject_flatline, inject_clipping
from metrics import detection_metrics, validation_accuracy
from agents.validation.validation_agent import validate_beat

DATA_DIR = "./data/mitbih"
RECORDS  = ['100', '101', '103', '105']   # small golden subset

# Load ~200 clean beats
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

# Run each corruption type
scenarios = [
    ("nan",      inject_nan,      {"expected_status": "REPAIR"}),
    ("flatline", inject_flatline, {"expected_status": "REPAIR"}),
    ("clipping", inject_clipping, {"expected_status": "REPAIR"}),
    ("clean",    None,            {"expected_status": "PASS"}),
]

all_results = []
for name, injector, meta in scenarios:
    for beat in clean_beats[:50]:
        if injector:
            corrupted, mask, _ = injector(beat)
        else:
            corrupted = beat

        result = validate_beat(corrupted)   # calls Neha S's agent

        all_results.append({
            "corruption":  name,
            "predicted":   result["status"],
            "expected":    meta["expected_status"],
            "flags":       result["flags"],
            "reason":      result["reason"],
        })

# Print summary
from collections import Counter
print("\nTriage Accuracy by Corruption Type:")
for name, _, meta in scenarios:
    subset = [r for r in all_results if r["corruption"] == name]
    acc = validation_accuracy(subset)
    counts = Counter(r["predicted"] for r in subset)
    print(f"  {name:12s}  acc={acc}%  {dict(counts)}")
