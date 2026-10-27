"""
Run the integrated pipeline on the real model and data.

    python -m scripts.run_real_pipeline                 # defaults below
    python -m scripts.run_real_pipeline --n 200 --reference 6000 --audit 10000

Needs artifacts/DenseNet1D_clean.keras and artifacts/{train,val,test}.csv
(the ECG_Splits files). Steps:

1. Fit the detection agents on the training split: signal poisoning on
   `--reference` clean beats, label poisoning by auditing `--audit` beats
   of which `--flip-fraction` have their labels deliberately flipped (so the
   audit's precision/recall can be measured; pass 0 for a real audit).
   Scores for new beats are calibrated on `--calibration` beats of the
   validation split (patients in neither train nor test).
2. Push `--n` unseen test beats per scenario through every agent and print
   the verdict / recovery / alert counts, plus both signal-fusion
   strategies' flag rates so the Decision thresholds can be calibrated.
3. Print one full clinician report per scenario.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from agents.decision.fusion import fuse_signal_evidence, fuse_signal_evidence_noisy_or
from evaluation.data import ARTIFACTS_DIR, load_split, stratified_sample
from evaluation.synthetic_beats import flip_labels, pgd_attack, poison_signal
from models.backend import KerasBackend
from orchestrator.pipeline import ECGDefensePipeline


def _scenarios(backend, test, idx, rng_seed):
    rng = np.random.default_rng(rng_seed)
    x, y = test.signals[idx], test.labels[idx]
    flipped = (y + rng.integers(1, backend.num_classes, size=len(y))) % backend.num_classes
    return {
        "clean": [(x[k], int(y[k])) for k in range(len(idx))],
        "label flipped": [(x[k], int(flipped[k])) for k in range(len(idx))],
        "gaussian s=0.01": [(poison_signal(x[k], 0.01, seed=k), int(y[k])) for k in range(len(idx))],
        "gaussian s=0.05": [(poison_signal(x[k], 0.05, seed=k), int(y[k])) for k in range(len(idx))],
        "PGD eps=0.05": [(pgd_attack(backend, x[k], int(y[k]), 0.05, 0.0125), int(y[k])) for k in range(len(idx))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ARTIFACTS_DIR / "DenseNet1D_clean.keras"))
    parser.add_argument("--reference", type=int, default=4000, help="clean training beats for signal detection")
    parser.add_argument("--audit", type=int, default=8000, help="training beats audited for label poisoning")
    parser.add_argument("--calibration", type=int, default=600,
                        help="validation-split beats (unseen patients) used to calibrate scores for new beats")
    parser.add_argument("--flip-fraction", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=60, help="test beats per scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ARTIFACTS_DIR / "results" / "real_results.json"),
                        help="where to save the results (read by scripts/build_report.py)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    t0 = time.time()
    backend = KerasBackend.load(args.model)
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    print(f"loaded model + data in {time.time() - t0:.0f}s "
          f"(train {len(train.labels)}, test {len(test.labels)} beats)")

    ref_idx = stratified_sample(train, args.reference, seed=args.seed)
    audit_idx = stratified_sample(train, args.audit, seed=args.seed + 1)
    audit_labels, flipped_mask = flip_labels(train.labels[audit_idx], args.flip_fraction, seed=args.seed)
    audit_ids = [train.beat_ids[i] for i in audit_idx]
    cal_idx = stratified_sample(val, args.calibration, seed=args.seed + 3)

    t0 = time.time()
    pipeline = ECGDefensePipeline.from_reference_data(
        backend, clean_signals=train.signals[ref_idx],
        audit_signals=train.signals[audit_idx], audit_labels=audit_labels, audit_record_ids=audit_ids,
        calibration_signals=val.signals[cal_idx], calibration_labels=val.labels[cal_idx])
    fit_seconds = time.time() - t0
    print(f"fitted detection agents in {fit_seconds:.0f}s")
    results: dict = {"config": vars(args), "fit_seconds": fit_seconds,
                     "thresholds": {"label": pipeline.decision_agent.config.label_threshold,
                                    "signal": pipeline.decision_agent.config.signal_threshold},
                     "scenarios": {}, "examples": {}}

    if args.flip_fraction > 0:
        flagged = {r["record_id"] for r in pipeline.audit_dataset()}
        is_flagged = np.array([rid in flagged for rid in audit_ids])
        tp = int((is_flagged & flipped_mask).sum())
        print(f"\n=== training-set label audit ({len(audit_ids)} beats, {int(flipped_mask.sum())} flipped) ===")
        print(f"  flagged {int(is_flagged.sum())}: recall {tp / max(flipped_mask.sum(), 1):.2f}, "
              f"precision {tp / max(is_flagged.sum(), 1):.2f}")
        results["audit"] = {"n": len(audit_ids), "flipped": int(flipped_mask.sum()),
                            "flagged": int(is_flagged.sum()), "true_positives": tp}

    test_idx = stratified_sample(test, args.n, seed=args.seed + 2)[: args.n]
    scenarios = _scenarios(backend, test, test_idx, args.seed)

    print(f"\n=== {len(test_idx)} unseen test beats per scenario ===")
    examples = {}
    t0, n_runs = time.time(), 0
    for name, beats in scenarios.items():
        verdicts, actions, alerts, fused_wm, fused_no = Counter(), Counter(), Counter(), [], []
        label_scores, attack_types, severities = [], Counter(), Counter()
        correct_dx = 0
        for k, (signal, label) in enumerate(beats):
            state = pipeline.run(signal, f"{name}_{k}", label=label)
            n_runs += 1
            alerts[state["report"]["alert_level"]] += 1
            if state["decision"] is None:
                verdicts["rejected by validation"] += 1
                continue
            verdicts[state["decision"].verdict.value] += 1
            label_scores.append(state["decision"].label_score)
            attack_types[state["decision"].attack_type or "none"] += 1
            severities[state["decision"].severity] += 1
            actions[state["recovery"].action.value if state["recovery"] else "none"] += 1
            evidence = state["signal_poisoning"].evidence
            fused_wm.append(fuse_signal_evidence(evidence))
            fused_no.append(fuse_signal_evidence_noisy_or(evidence))
            dx = state["diagnosis"]
            if dx and dx["predicted_class"] == test.labels[test_idx[k]]:
                correct_dx += 1
            examples.setdefault(name, state)
        thr = pipeline.decision_agent.config.signal_threshold
        print(f"\n  {name}")
        print(f"    verdicts: {dict(verdicts)}")
        print(f"    recovery: {dict(actions)}   alerts: {dict(alerts)}")
        print(f"    diagnosed with the true class: {correct_dx}/{len(beats)}")
        print(f"    signal flag rate @ {thr}: weighted_mean {np.mean(np.array(fused_wm) >= thr):.2f}, "
              f"noisy_or {np.mean(np.array(fused_no) >= thr):.2f}")
        results["scenarios"][name] = {
            "n": len(beats), "verdicts": dict(verdicts), "recovery": dict(actions), "alerts": dict(alerts),
            "attack_types": dict(attack_types), "severity": dict(severities), "correct_diagnosis": correct_dx,
            "signal_scores_weighted_mean": [float(v) for v in fused_wm],
            "signal_scores_noisy_or": [float(v) for v in fused_no],
            "label_scores": [float(v) for v in label_scores],
        }
    results["seconds_per_beat"] = (time.time() - t0) / max(n_runs, 1)
    print(f"\n  {results['seconds_per_beat']:.2f}s per beat through all agents")

    for name, state in examples.items():
        print(f"\n=== example report: {name} ===")
        print(state["report"]["text"])
        results["examples"][name] = state["report"]["text"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved results to {out}")


if __name__ == "__main__":
    main()
