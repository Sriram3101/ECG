"""Nihar's noise-corrupted training/eval data generation.

Combines two corruption sources into per-beat records that match the
"corrupted signal set" contract from `agents_pipeline.txt` / `datasets.py`:

  - NSTDB real-world noise (baseline wander, muscle artifact, electrode
    motion) added at controlled SNR levels.
  - PGD adversarial perturbations computed against the trained DenseNet1D
    classifier.

Only `signal` is ever touched here -- `label`/`given_label` stay untouched,
and `is_corrupted` / `noise_type` / `severity` are written to a separate
`*.truth.json` file, never fed to a detector (see contracts.py docstring).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal
import json
import numpy as np
import wfdb

NoiseType = Literal["baseline_wander", "ma", "em", "pgd_adversarial"]

# NSTDB record names on PhysioNet for each noise type we use.
_NSTDB_RECORD_FOR_TYPE = {"baseline_wander": "bw", "ma": "ma", "em": "em"}


def sample_role_from_noise_type(noise_type: str | None) -> str:
    """Maps a per-beat `noise_type` to its training role.

    "attack" (PGD) is the only positive class the detector should learn to
    catch. "distractor" (real NSTDB sensor noise: baseline_wander/ma/em)
    must NEVER be used as a positive training label -- it exists purely to
    check afterward that the detector doesn't confuse ordinary sensor
    artifacts with adversarial perturbation (see
    `noise_classifier.evaluate_distractor_specificity`). This is the single
    source of truth for that mapping; both `train_noise_classifier.py` and
    tests should derive `sample_role` from this function rather than
    re-deriving it inline.
    """
    if noise_type is None:
        return "clean"
    if noise_type == "pgd_adversarial":
        return "attack"
    return "distractor"


@dataclass
class NSTDBNoiseBank:
    """Loads MIT-BIH Noise Stress Test Database noise segments once and
    slices random windows out of them for reuse across many beats."""
    noise_by_type: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def load(cls, nstdb_dir: str | Path, types: Iterable[str] = ("baseline_wander", "ma", "em")) -> "NSTDBNoiseBank":
        nstdb_dir = Path(nstdb_dir)
        noise_by_type = {}
        for noise_type in types:
            record_name = _NSTDB_RECORD_FOR_TYPE[noise_type]
            record = wfdb.rdrecord(str(nstdb_dir / record_name))
            # NSTDB noise records have two channels; average to one trace.
            noise_by_type[noise_type] = record.p_signal.astype(np.float64).mean(axis=1)
        return cls(noise_by_type=noise_by_type)

    def sample_segment(self, noise_type: str, length: int, rng: np.random.Generator) -> np.ndarray:
        trace = self.noise_by_type[noise_type]
        if len(trace) < length:
            raise ValueError(f"NSTDB '{noise_type}' trace shorter than requested segment ({length}).")
        start = int(rng.integers(0, len(trace) - length + 1))
        segment = trace[start:start + length]
        return segment - segment.mean()  # drop DC offset so target SNR is meaningful


def _scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scales `noise` so clean + scaled_noise lands at the requested SNR (dB)."""
    signal_power = float(np.mean(clean ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    return noise * np.sqrt(target_noise_power / noise_power)


def add_nstdb_noise(beat: np.ndarray, noise_type: str, snr_db: float, bank: NSTDBNoiseBank,
                     rng: np.random.Generator) -> np.ndarray:
    """Adds a real NSTDB noise segment to one (360, 1) beat at a target SNR."""
    x = np.asarray(beat, dtype=np.float64).reshape(-1)
    noise = bank.sample_segment(noise_type, len(x), rng)
    scaled = _scale_noise_to_snr(x, noise, snr_db)
    return (x + scaled).astype(np.float32).reshape(-1, 1)


@dataclass
class PGDConfig:
    epsilon: float = 0.05    # L-infinity budget, post-z-score units (matches add_gaussian_noise's convention)
    step_size: float = 0.01
    num_steps: int = 10
    random_start: bool = True


def pgd_adversarial_noise(model, beats: np.ndarray, labels: np.ndarray, config: PGDConfig, seed: int = 42):
    """Untargeted PGD (L-inf) against the trained Keras DenseNet1D classifier.

    `beats` is (N, 360, 1) float32 in the same normalized space the model
    trains on. Returns (adversarial_beats, epsilon_used_per_beat) -- the
    latter is what gets stored as `severity` for pgd_adversarial beats.
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(beats, dtype=tf.float32)
    y = tf.convert_to_tensor(labels, dtype=tf.int64)
    rng = np.random.default_rng(seed)

    if config.random_start:
        start_noise = rng.uniform(-config.epsilon, config.epsilon, size=beats.shape).astype(np.float32)
        x_adv = tf.clip_by_value(x + start_noise, x - config.epsilon, x + config.epsilon)
    else:
        x_adv = tf.identity(x)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    for _ in range(config.num_steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv, training=False)
            loss = loss_fn(y, logits)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + config.step_size * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x - config.epsilon, x + config.epsilon)  # project to epsilon ball
        x_adv = tf.stop_gradient(x_adv)

    return x_adv.numpy(), np.full(len(beats), config.epsilon, dtype=np.float32)


def build_noise_training_set(clean_signals: np.ndarray, labels: np.ndarray, nstdb_bank: NSTDBNoiseBank,
                              model=None, snr_levels_db: tuple[float, ...] = (0.0, 6.0, 12.0),
                              pgd_config: PGDConfig | None = None, corrupt_fraction: float = 0.5,
                              pgd_fraction_of_corrupted: float | None = None, seed: int = 42):
    """Builds one corrupted signal set mixing NSTDB noise types and PGD.

    `corrupt_fraction` of beats are corrupted. By default the corrupted pool
    is split uniformly across whichever types are enabled (unchanged from
    before -- fine for generating an eval-only distractor set). Pass
    `pgd_fraction_of_corrupted` (e.g. 0.6) to bias that split toward PGD --
    do this for the split you'll actually train the classifier on, since
    `train_noise_classifier` excludes NSTDB-corrupted ("distractor") beats
    from training entirely, so a uniform split leaves too few real "attack"
    positives once distractors are dropped.

    Returns (signals, is_corrupted, noise_type, severity) -- four parallel
    arrays/lists matching the per-beat contract in `agents_pipeline.txt`.
    """
    rng = np.random.default_rng(seed)
    n = len(clean_signals)
    is_corrupted = rng.random(n) < corrupt_fraction
    noise_types = ["baseline_wander", "ma", "em"] + (["pgd_adversarial"] if model is not None else [])

    if model is not None and pgd_fraction_of_corrupted is not None:
        distractor_share = (1.0 - pgd_fraction_of_corrupted) / 3
        weights = [distractor_share, distractor_share, distractor_share, pgd_fraction_of_corrupted]
        assigned_type = rng.choice(noise_types, size=n, p=weights)
    else:
        assigned_type = rng.choice(noise_types, size=n)

    out_signals = np.array(clean_signals, dtype=np.float32, copy=True)
    out_severity: list[float | None] = [None] * n
    out_type: list[str | None] = [None] * n

    for i in range(n):
        if not is_corrupted[i] or assigned_type[i] == "pgd_adversarial":
            continue
        snr_db = float(rng.choice(snr_levels_db))
        out_signals[i] = add_nstdb_noise(clean_signals[i], assigned_type[i], snr_db, nstdb_bank, rng)
        out_severity[i] = snr_db
        out_type[i] = assigned_type[i]

    if model is not None:
        pgd_mask = is_corrupted & (assigned_type == "pgd_adversarial")
        if pgd_mask.any():
            cfg = pgd_config or PGDConfig()
            adv, eps_used = pgd_adversarial_noise(model, clean_signals[pgd_mask], labels[pgd_mask], cfg, seed=seed)
            out_signals[pgd_mask] = adv
            for idx, eps in zip(np.flatnonzero(pgd_mask), eps_used):
                out_severity[idx] = float(eps)
                out_type[idx] = "pgd_adversarial"

    return out_signals, is_corrupted, out_type, out_severity


def save_per_beat_noise_set(path: str | Path, signals: np.ndarray, labels: np.ndarray, record_ids: np.ndarray,
                             is_corrupted: np.ndarray, noise_type: list[str | None],
                             severity: list[float | None]) -> None:
    """Saves a heterogeneous (mixed noise-type) set. Unlike
    `datasets.save_corrupted_signal_set` (one noise_type/severity per file),
    this writes per-beat truth since one call to `build_noise_training_set`
    mixes bw/ma/em/pgd in the same split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signals=np.asarray(signals, dtype=np.float32),
                         labels=np.asarray(labels), record_ids=np.asarray(record_ids))
    truth = [{"is_corrupted": bool(flag), "noise_type": nt, "severity": sv}
              for flag, nt, sv in zip(is_corrupted, noise_type, severity)]
    path.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")


def generate_and_save(split_name: str, clean_signals: np.ndarray, labels: np.ndarray, record_ids: np.ndarray,
                       nstdb_dir: str | Path, out_dir: str | Path, model=None, **kwargs) -> Path:
    """End-to-end: load NSTDB, corrupt one split, save. Only call with
    `model` set for the train split you'll train the noise classifier on --
    val/test for the *classifier itself* should include held-out corruption
    (see agents_pipeline.txt point 4) generated the same way on unseen beats."""
    bank = NSTDBNoiseBank.load(nstdb_dir)
    signals, is_corrupted, noise_type, severity = build_noise_training_set(
        clean_signals, labels, bank, model=model, **kwargs)
    out_path = Path(out_dir) / f"{split_name}_noise.npz"
    save_per_beat_noise_set(out_path, signals, labels, record_ids, is_corrupted, noise_type, severity)
    return out_path
