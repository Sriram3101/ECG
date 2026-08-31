"""Fits Nihar's noise classifier and saves it for `NoiseClassifierAgent`.

Two-stage pipeline:
  1. `noise_datagen.generate_and_save` builds a corrupted training split
     (NSTDB bw/ma/em + optionally PGD against the DenseNet1D checkpoint).
  2. This script extracts Sriram's features from that split and fits
     `TrainedNoiseClassifier` on them.

Usage:
    python -m ecg_defense.scripts.train_noise_classifier \\
        --clean-signals artifacts/clean_train.npz \\
        --nstdb-dir data/nstdb \\
        --checkpoint artifacts/DenseNet1D_clean.keras \\
        --out artifacts/noise_classifier.joblib

Pass --no-pgd to skip the checkpoint-dependent PGD split (NSTDB-only agent).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

from ecg_defense.noise_classifier import features_to_vector, train_noise_classifier, evaluate_distractor_specificity
from ecg_defense.noise_datagen import generate_and_save, sample_role_from_noise_type
from ecg_defense.noise_features import MorphologyTemplateBank, extract_noise_features


def load_model(checkpoint_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-signals", required=True, help=".npz with 'signals','labels','record_ids'")
    parser.add_argument("--nstdb-dir", required=True, help="Directory containing NSTDB bw/ma/em records")
    parser.add_argument("--checkpoint", help="Path to the trained DenseNet1D .keras checkpoint (for PGD)")
    parser.add_argument("--no-pgd", action="store_true", help="Skip PGD; train on NSTDB noise only")
    parser.add_argument("--pgd-fraction-of-corrupted", type=float, default=0.6,
                         help="Share of corrupted beats that are PGD rather than NSTDB (default 0.6). "
                              "Distractor (NSTDB) beats are excluded from training, so this needs to be "
                              "high enough to leave a reasonable number of attack positives.")
    parser.add_argument("--out", required=True, help="Output path for the trained classifier (.joblib)")
    parser.add_argument("--work-dir", default="artifacts/noise_train", help="Where to write the generated corrupted split")
    args = parser.parse_args()

    with np.load(args.clean_signals) as npz:
        clean_signals, labels, record_ids = npz["signals"], npz["labels"], npz["record_ids"]

    model = None if args.no_pgd else load_model(args.checkpoint)
    if not args.no_pgd and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no-pgd is set")

    corrupted_path = generate_and_save("train", clean_signals, labels, record_ids,
                                        nstdb_dir=args.nstdb_dir, out_dir=args.work_dir, model=model,
                                        pgd_fraction_of_corrupted=None if args.no_pgd else args.pgd_fraction_of_corrupted)
    print(f"Generated corrupted training set at {corrupted_path}")

    with np.load(corrupted_path) as npz:
        signals = npz["signals"]
    truth = json.loads(corrupted_path.with_suffix(".truth.json").read_text())

    # Fit morphology templates on the clean beats in this split, save alongside the classifier
    # so NoiseClassifierAgent and this script always agree on what "clean" looks like.
    clean_idx = [i for i, t in enumerate(truth) if not t["is_corrupted"]]
    templates = MorphologyTemplateBank.fit([signals[i] for i in clean_idx[:64]])

    feature_matrix = np.array([features_to_vector(extract_noise_features(s, templates)) for s in signals])
    sample_role = np.array([sample_role_from_noise_type(t["noise_type"]) for t in truth])
    role_counts = {role: int((sample_role == role).sum()) for role in ("clean", "distractor", "attack")}
    print(f"Sample roles in training split: {role_counts}")

    classifier = train_noise_classifier(feature_matrix, sample_role)
    evaluate_distractor_specificity(classifier, feature_matrix, sample_role)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(out_path)
    templates.save(str(out_path.with_suffix(".templates.npz")))
    print(f"Saved classifier to {out_path} (templates alongside it)")


if __name__ == "__main__":
    main()
