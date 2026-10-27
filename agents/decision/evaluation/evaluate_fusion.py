"""
agents/decision/evaluation/evaluate_fusion.py

Offline evaluation of the Decision agent against a labeled attack-
simulation dataset, and an optional threshold sweep to calibrate
thresholds.yaml.

Expected input: a JSONL file, one record per line:

    {
      "record_id": "rec_0001",
      "label_score": 0.12,
      "signal_evidence": [
        {"check_name": "reconstruction_error",  "channel": "signal", "score": 0.08},
        {"check_name": "feature_space_distance", "channel": "signal", "score": 0.15},
        {"check_name": "prediction_fragility",   "channel": "signal", "score": 0.05},
        {"check_name": "gradient_sensitivity",   "channel": "signal", "score": 0.10}
      ],
      "true_verdict": "clean"
    }

`true_verdict` must be one of: clean, label_poisoned, signal_poisoned,
both_poisoned (see shared.contracts.Verdict). Build this file from a
dataset where you've synthetically poisoned a known fraction of labels
and/or signals, so ground truth is known.

Usage:
    python -m agents.decision.evaluation.evaluate_fusion \\
        --data attack_sim.jsonl \\
        --config agents/decision/config/thresholds.yaml

    # Grid-search label_threshold / signal_threshold for best macro F1:
    python -m agents.decision.evaluation.evaluate_fusion \\
        --data attack_sim.jsonl --sweep-thresholds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from agents.decision.decision_agent import DecisionAgent, DecisionAgentConfig
from agents.decision.fusion import fuse_signal_evidence
from shared.contracts import Evidence, Verdict

VERDICT_ORDER = [v.value for v in Verdict]


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


def _parse_signal_evidence(raw: list[dict]) -> list[Evidence]:
    return [Evidence(**e) for e in raw]


def evaluate(agent: DecisionAgent, rows: list[dict]) -> dict:
    y_true, y_pred = [], []
    label_scores, signal_scores, poisoned_true = [], [], []

    for row in rows:
        evidence = _parse_signal_evidence(row["signal_evidence"])
        decision = agent.run(row["record_id"], row["label_score"], evidence)

        y_true.append(row["true_verdict"])
        y_pred.append(decision.verdict.value)
        label_scores.append(decision.label_score)
        signal_scores.append(decision.signal_score)
        poisoned_true.append(0 if row["true_verdict"] == "clean" else 1)

    report = classification_report(
        y_true, y_pred, labels=VERDICT_ORDER, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=VERDICT_ORDER)

    result = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": VERDICT_ORDER,
    }

    # AUC of each component score against binary poisoned/clean, so you
    # can see whether the label or signal channel is doing the heavy
    # lifting (or whether one is barely better than noise).
    if len(set(poisoned_true)) > 1:
        result["label_score_auc"] = roc_auc_score(poisoned_true, label_scores)
        result["signal_score_auc"] = roc_auc_score(poisoned_true, signal_scores)

    return result


def print_report(result: dict) -> None:
    print("\n=== Confusion matrix (rows=true, cols=predicted) ===")
    labels = result["confusion_matrix_labels"]
    print(" " * 18 + " ".join(f"{l[:14]:>14}" for l in labels))
    for label, row in zip(labels, result["confusion_matrix"]):
        print(f"{label[:16]:>16}  " + " ".join(f"{v:>14d}" for v in row))

    print("\n=== Per-verdict precision / recall / F1 ===")
    for verdict, metrics in result["classification_report"].items():
        if verdict in ("accuracy", "macro avg", "weighted avg"):
            continue
        print(
            f"{verdict:>16}: precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f} "
            f"support={int(metrics['support'])}"
        )
    print(f"\naccuracy: {result['classification_report']['accuracy']:.3f}")
    print(f"macro F1: {result['classification_report']['macro avg']['f1-score']:.3f}")

    if "label_score_auc" in result:
        print(f"\nlabel_score  AUC (vs. clean/poisoned): {result['label_score_auc']:.3f}")
        print(f"signal_score AUC (vs. clean/poisoned): {result['signal_score_auc']:.3f}")


def sweep_thresholds(
    rows: list[dict],
    base_config: DecisionAgentConfig,
    grid: np.ndarray = np.round(np.arange(0.30, 0.81, 0.05), 2),
) -> tuple[float, float, float]:
    """Grid search label_threshold x signal_threshold for best macro F1.
    Returns (best_label_threshold, best_signal_threshold, best_macro_f1).
    """
    best = (base_config.label_threshold, base_config.signal_threshold, -1.0)

    for lt in grid:
        for st in grid:
            cfg = DecisionAgentConfig(
                label_threshold=float(lt),
                signal_threshold=float(st),
                uncertain_margin=base_config.uncertain_margin,
                fusion_strategy=base_config.fusion_strategy,
                signal_check_weights=base_config.signal_check_weights,
            )
            agent = DecisionAgent(cfg)
            result = evaluate(agent, rows)
            macro_f1 = result["classification_report"]["macro avg"]["f1-score"]
            if macro_f1 > best[2]:
                best = (float(lt), float(st), macro_f1)

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to JSONL attack-simulation dataset")
    parser.add_argument("--config", default=None, help="path to thresholds.yaml (default: agent's built-in config)")
    parser.add_argument("--sweep-thresholds", action="store_true", help="grid-search thresholds for best macro F1")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    config = DecisionAgentConfig.from_yaml(args.config) if args.config else DecisionAgentConfig.from_yaml()
    agent = DecisionAgent(config)

    print(f"Loaded {len(rows)} records from {args.data}")
    result = evaluate(agent, rows)
    print_report(result)

    if args.sweep_thresholds:
        print("\n=== Threshold sweep ===")
        best_lt, best_st, best_f1 = sweep_thresholds(rows, config)
        print(f"best label_threshold={best_lt:.2f}  signal_threshold={best_st:.2f}  macro F1={best_f1:.3f}")
        print("Update agents/decision/config/thresholds.yaml with these values if they hold up on a separate held-out split.")


if __name__ == "__main__":
    main()
