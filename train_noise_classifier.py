"""Fits Nihar's signal-poisoning classifier and saves it for `NoiseClassifierAgent`.

Two-stage pipeline:
  1. `noise_datagen.generate_and_save` builds a corrupted training split
     (NSTDB bw/ma/em distractors + an "attack" split -- Gaussian poisoning
     by default, matching the sigma=0.01/0.05 experiment already run; PGD
     is available via --attack-mode pgd but isn't the current target).
  2. This script extracts Sriram's features from that split and fits
     `TrainedNoiseClassifier` on them.

Usage (current default -- Gaussian poisoning, no checkpoint needed):
    python -m ecg_defense.train_noise_classifier \\
        --clean-signals artifacts/clean_train.npz \\
        --nstdb-dir data/nstdb \\
        --out artifacts/noise_classifier.joblib

Usage (PGD, kept available for later):
    python -m ecg_defense.train_noise_classifier \\
        --clean-signals artifacts/clean_train.npz \\
        --nstdb-dir data/nstdb \\
        --attack-mode pgd --checkpoint artifacts/DenseNet1D_clean.keras \\
        --out artifacts/noise_classifier.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

from ecg_defense.noise_classifier import (features_to_vector, train_noise_classifier,
                                           evaluate_distractor_specificity, split_distractor_for_training)
from ecg_defense.noise_datagen import generate_and_save, sample_role_from_noise_type
from ecg_defense.noise_features import MorphologyTemplateBank, extract_noise_features


def load_model(checkpoint_path: str):
    import tensorflow as tf
    from ecg_defense.model_layers import AttentionLayer
    return tf.keras.models.load_model(checkpoint_path, custom_objects={"AttentionLayer": AttentionLayer})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-signals", required=True, help=".npz with 'signals','labels','record_ids'")
    parser.add_argument("--nstdb-dir", required=True, help="Directory containing NSTDB bw/ma/em records")
    parser.add_argument("--attack-mode", choices=["gaussian", "pgd"], default="gaussian",
                         help="What the 'attack' class is (default: gaussian, matching the sigma=0.01/0.05 "
                              "experiment already run -- needs no checkpoint). 'pgd' needs --checkpoint.")
    parser.add_argument("--gaussian-sigmas", type=float, nargs="+", default=[0.01, 0.05],
                         help="Sigma values (z-score space) to draw from for gaussian attack beats.")
    parser.add_argument("--checkpoint", help="Path to the trained DenseNet1D .keras checkpoint (only for --attack-mode pgd)")
    parser.add_argument("--attack-fraction-of-corrupted", type=float, default=0.6,
                         help="Share of corrupted beats that are the attack type rather than NSTDB (default 0.6). "
                              "Distractor (NSTDB) beats are excluded from training, so this needs to be "
                              "high enough to leave a reasonable number of attack positives.")
    parser.add_argument("--distractor-train-fraction", type=float, default=0.7,
                         help="Share of NSTDB distractor beats folded into training as explicit negative "
                              "('not attack') examples, rather than held out entirely (default 0.7). "
                              "The rest stay held out for the specificity check. Pass 0 to disable (old "
                              "behavior: all distractor beats held out).")
    parser.add_argument("--out", required=True, help="Output path for the trained classifier (.joblib)")
    parser.add_argument("--work-dir", default="artifacts/noise_train", help="Where to write the generated corrupted split")
    args = parser.parse_args()

    if args.attack_mode == "pgd" and args.checkpoint is None:
        parser.error("--checkpoint is required when --attack-mode pgd is set")

    with np.load(args.clean_signals) as npz:
        clean_signals, labels, record_ids = npz["signals"], npz["labels"], npz["record_ids"]

    model = load_model(args.checkpoint) if args.attack_mode == "pgd" else None

    corrupted_path = generate_and_save("train", clean_signals, labels, record_ids,
                                        nstdb_dir=args.nstdb_dir, out_dir=args.work_dir, model=model,
                                        attack_mode=args.attack_mode, gaussian_sigmas=tuple(args.gaussian_sigmas),
                                        attack_fraction_of_corrupted=args.attack_fraction_of_corrupted)
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

    if args.distractor_train_fraction > 0:
        role_for_training, distractor_holdout_mask = split_distractor_for_training(
            sample_role, train_fraction=args.distractor_train_fraction)
        print(f"Folded {int((sample_role == 'distractor').sum() - distractor_holdout_mask.sum())} distractor "
              f"beats into training as negatives; {int(distractor_holdout_mask.sum())} held out for specificity.")
    else:
        role_for_training = sample_role
        distractor_holdout_mask = sample_role == "distractor"  # old behavior: all distractor held out

    classifier = train_noise_classifier(feature_matrix, role_for_training)
    evaluate_distractor_specificity(classifier, feature_matrix, sample_role, restrict_to=distractor_holdout_mask)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(out_path)
    templates.save(str(out_path.with_suffix(".templates.npz")))
    print(f"Saved classifier to {out_path} (templates alongside it)")


if __name__ == "__main__":
    main()
