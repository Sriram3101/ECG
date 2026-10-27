"""
End-to-end demo of the integrated ECG defense pipeline on synthetic beats.

    python -m scripts.run_pipeline_demo            # PyTorch model trained on the fly
    python -m scripts.run_pipeline_demo --keras artifacts/DenseNet1D_clean.keras \
        --data artifacts/clean_train.npz           # the team's real checkpoint + data

Runs one clean beat, one label-poisoned beat, one signal-poisoned beat and
one invalid beat through every agent and prints each clinician report.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter

import numpy as np

from core.ecg_state import make_ecg_state
from evaluation.synthetic_beats import flip_labels, make_beats, pgd_attack, poison_signal
from orchestrator.pipeline import ECGDefensePipeline


def _torch_backend(signals, labels):
    from models.backend import TorchBackend
    from models.densenet_bilstm import build_and_train_classifier

    model = build_and_train_classifier(signals, labels, epochs=8)
    return TorchBackend(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", help="path to DenseNet1D_clean.keras (default: train a small PyTorch model)")
    parser.add_argument("--data", help=".npz with 'signals' (N,360[,1]) and 'labels' (default: synthetic beats)")
    parser.add_argument("--graph", action="store_true", help="run through the LangGraph graph instead of the plain runner")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="also print verdict counts over N beats per scenario")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.data:
        with np.load(args.data) as npz:
            signals, labels = npz["signals"].reshape(len(npz["signals"]), -1), npz["labels"].astype(int)
    else:
        signals, labels = make_beats(n_per_class=120, seed=0)

    if args.keras:
        from models.backend import KerasBackend
        backend = KerasBackend.load(args.keras)
    else:
        backend = _torch_backend(signals, labels)

    # The dataset under audit has 8% of its labels flipped.
    audit_labels, poisoned_mask = flip_labels(labels, 0.08, seed=1)
    record_ids = [f"beat_{i:05d}" for i in range(len(labels))]
    pipeline = ECGDefensePipeline.from_reference_data(
        backend, clean_signals=signals, audit_signals=signals, audit_labels=audit_labels,
        audit_record_ids=record_ids)

    clean_idx = int(np.flatnonzero(~poisoned_mask)[0])
    flipped_idx = int(np.flatnonzero(poisoned_mask)[0])
    cases = [
        ("clean beat", signals[clean_idx], record_ids[clean_idx], int(audit_labels[clean_idx])),
        ("label-poisoned beat", signals[flipped_idx], record_ids[flipped_idx], int(audit_labels[flipped_idx])),
        ("signal-poisoned beat (PGD, eps=0.05)", pgd_attack(backend, signals[clean_idx], int(labels[clean_idx])),
         "incoming_0001", int(labels[clean_idx])),
        ("signal-poisoned beat (Gaussian, sigma=0.08)", poison_signal(signals[clean_idx], seed=2),
         "incoming_0002", int(labels[clean_idx])),
        ("invalid beat (NaN run)", np.where(np.arange(360) < 60, np.nan, signals[clean_idx]), "incoming_0003", None),
    ]
    graph = pipeline.build_graph() if args.graph else None
    for title, signal, beat_id, label in cases:
        if graph is not None:
            state = graph.invoke(make_ecg_state(beat_id, signal, label=label))
        else:
            state = pipeline.run(signal, beat_id, label=label)
        print(f"\n=== {title} ===")
        print(state["report"]["text"])

    if args.summary:
        _print_verdict_summary(pipeline, backend, signals, labels, audit_labels, poisoned_mask, record_ids, args.summary)


def _print_verdict_summary(pipeline, backend, signals, labels, audit_labels, poisoned_mask, record_ids, n):
    """Verdict counts per scenario — a quick sanity check, not a calibrated
    evaluation (use agents/decision/evaluation/evaluate_fusion.py for that)."""
    clean = np.flatnonzero(~poisoned_mask)[:n]
    flipped = np.flatnonzero(poisoned_mask)[:n]
    scenarios = {
        "clean": [(signals[i], record_ids[i], audit_labels[i]) for i in clean],
        "label flipped": [(signals[i], record_ids[i], audit_labels[i]) for i in flipped],
        "PGD eps=0.05": [(pgd_attack(backend, signals[i], int(labels[i])), f"pgd_{i}", labels[i]) for i in clean],
    }
    print("\n=== verdict counts ===")
    for name, beats in scenarios.items():
        verdicts = Counter(pipeline.run(s, b, label=int(y))["decision"].verdict.value for s, b, y in beats)
        print(f"  {name:<14} n={len(beats):<3} {dict(verdicts)}")


if __name__ == "__main__":
    main()
