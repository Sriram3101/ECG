"""
agents/recovery/evaluation/evaluate_recovery.py

Offline evaluation of the Recovery agent against a labeled
attack-simulation dataset where you know the *true* clean label and/or
true clean signal underneath a synthetic attack.

This measures the thing that actually matters: does repair get you
closer to ground truth than doing nothing, and how often does the agent
correctly recognize when a repair isn't trustworthy and quarantines
instead of guessing.

Expected input: a JSONL file, one record per line:

    {
      "record_id": "rec_0001",
      "verdict": "label_poisoned",
      "label_score": 0.91,
      "signal_score": 0.10,
      "record": {
        "record_id": "rec_0001",
        "signal": [0.01, 0.02, ...],   // length 360, the POISONED signal
        "label": 3                       // the POISONED label
      },
      "true_label": 1,                   // ground truth, if verdict touches label
      "true_signal": [0.00, 0.01, ...],  // ground truth, if verdict touches signal
      "oof_probabilities": [0.05, 0.85, 0.03, 0.05, 0.02],   // if label-side
      "autoencoder_reconstruction": [0.00, 0.01, ...],        // if signal-side
      "reconstruction_confidence": 0.88                       // if signal-side — NOT a
                                                                // distance from the original;
                                                                // see strategies/signal_repair.py
    }

`signal_evidence` is synthesized internally as a single dummy
"reconstruction_error"/etc. entry since the Recovery agent's dispatch
only depends on `verdict`, not the raw evidence — this dataset format
mirrors what you'd get from a full pipeline run without requiring you to
also fake the Decision agent's fusion inputs.

Usage:
    python -m agents.recovery.evaluation.evaluate_recovery --data attack_sim.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents.recovery.recovery_agent import RecoveryAgent, RecoveryAgentConfig
from shared.contracts import DecisionRecord, ECGRecord, Evidence, RecoveryAction, Verdict

_DUMMY_SIGNAL_EVIDENCE = [
    Evidence(check_name=name, channel="signal", score=0.5)
    for name in [
        "reconstruction_error",
        "feature_space_distance",
        "prediction_fragility",
        "gradient_sensitivity",
    ]
]


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def _build_decision(row: dict) -> DecisionRecord:
    return DecisionRecord(
        record_id=row["record_id"],
        agent_name="decision",
        agent_version="0.0.0-eval",
        verdict=Verdict(row["verdict"]),
        confidence=max(row.get("label_score", 0.0), row.get("signal_score", 0.0)),
        label_score=row.get("label_score", 0.0),
        signal_score=row.get("signal_score", 0.0),
        evidence=_DUMMY_SIGNAL_EVIDENCE,
        recommended_action="repair",
    )


def evaluate(agent: RecoveryAgent, rows: list[dict]) -> dict:
    action_counts: Counter = Counter()
    relabel_correct = relabel_total = 0
    signal_mae_before, signal_mae_after = [], []

    for row in rows:
        decision = _build_decision(row)
        record = ECGRecord(**row["record"])

        result = agent.run(
            decision,
            record,
            oof_probabilities=row.get("oof_probabilities"),
            autoencoder_reconstruction=row.get("autoencoder_reconstruction"),
            reconstruction_confidence=row.get("reconstruction_confidence"),
        )
        action_counts[result.action.value] += 1

        if result.action == RecoveryAction.RELABEL and "true_label" in row:
            relabel_total += 1
            if result.repaired_record.label == row["true_label"]:
                relabel_correct += 1

        if result.action == RecoveryAction.RECONSTRUCT_SIGNAL and "true_signal" in row:
            true_signal = np.asarray(row["true_signal"])
            original_signal = np.asarray(row["record"]["signal"])
            repaired_signal = np.asarray(result.repaired_record.signal)
            signal_mae_before.append(float(np.mean(np.abs(original_signal - true_signal))))
            signal_mae_after.append(float(np.mean(np.abs(repaired_signal - true_signal))))

    result = {
        "total_records": len(rows),
        "action_counts": dict(action_counts),
        "action_rate": {k: v / len(rows) for k, v in action_counts.items()},
    }
    if relabel_total:
        result["relabel_accuracy"] = relabel_correct / relabel_total
        result["relabel_total"] = relabel_total
    if signal_mae_before:
        result["signal_mae_before_repair"] = float(np.mean(signal_mae_before))
        result["signal_mae_after_repair"] = float(np.mean(signal_mae_after))
        result["signal_repair_improvement"] = float(
            np.mean(signal_mae_before) - np.mean(signal_mae_after)
        )
    return result


def print_report(result: dict) -> None:
    print(f"\nTotal records evaluated: {result['total_records']}")

    print("\n=== Action distribution ===")
    for action, count in sorted(result["action_counts"].items()):
        rate = result["action_rate"][action]
        print(f"{action:>20}: {count:>5}  ({rate:.1%})")

    if "relabel_accuracy" in result:
        print(f"\n=== Label repair ===")
        print(f"relabel accuracy (vs. ground truth): {result['relabel_accuracy']:.1%} "
              f"over {result['relabel_total']} relabeled records")

    if "signal_mae_before_repair" in result:
        print(f"\n=== Signal repair ===")
        print(f"mean abs error before repair: {result['signal_mae_before_repair']:.4f}")
        print(f"mean abs error after repair:  {result['signal_mae_after_repair']:.4f}")
        improvement = result["signal_repair_improvement"]
        verdict = "improved" if improvement > 0 else "WORSE — investigate before trusting this repair path"
        print(f"net change: {improvement:+.4f} ({verdict})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to JSONL attack-simulation dataset")
    parser.add_argument("--config", default=None, help="path to repair_thresholds.yaml")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    config = RecoveryAgentConfig.from_yaml(args.config) if args.config else RecoveryAgentConfig.from_yaml()
    agent = RecoveryAgent(config)

    print(f"Loaded {len(rows)} records from {args.data}")
    result = evaluate(agent, rows)
    print_report(result)


if __name__ == "__main__":
    main()
