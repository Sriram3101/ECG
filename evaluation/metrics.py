# evaluation/metrics.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def detection_metrics(true_mask, pred_mask, label=""):
    """
    Evaluate how well an agent detected corrupted/poisoned/noisy samples.
    true_mask: ground-truth bool array from generate_invalid_data
    pred_mask: predicted bool array from agent output
    """
    p = precision_score(true_mask, pred_mask, zero_division=0)
    r = recall_score(true_mask, pred_mask, zero_division=0)
    f = f1_score(true_mask, pred_mask, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_mask, pred_mask, labels=[False, True]).ravel()
    return {
        "label": label,
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "detected_pct": round(pred_mask.mean() * 100, 2)
    }

def validation_accuracy(results):
    """
    Given a list of ValidationResult dicts and expected statuses,
    return % correct triage decisions.
    results: list of {"predicted": "PASS"|"REPAIR"|"REJECT",
                       "expected":  "PASS"|"REPAIR"|"REJECT"}
    """
    correct = sum(r["predicted"] == r["expected"] for r in results)
    return round(correct / len(results) * 100, 2)
