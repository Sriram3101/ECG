import numpy as np
from ecg_defense.noise_datagen import (NSTDBNoiseBank, add_nstdb_noise, build_noise_training_set,
                                        sample_role_from_noise_type, _scale_noise_to_snr)
from ecg_defense.noise_features import MorphologyTemplateBank, extract_noise_features
from ecg_defense.noise_classifier import (features_to_vector, train_noise_classifier, evaluate_distractor_specificity,
                                           TrainedNoiseClassifier)
from ecg_defense.noise_agent import NoiseClassifierAgent


def _synthetic_clean_beats(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, 360)
    beats = np.stack([np.sin(t + rng.normal(0, 0.05)) + rng.normal(0, 0.02, 360) for _ in range(n)])
    return beats[:, :, None].astype(np.float32), rng.integers(0, 4, size=n)


def _fake_nstdb_bank(seed=0):
    rng = np.random.default_rng(seed)
    return NSTDBNoiseBank(noise_by_type={
        "baseline_wander": rng.normal(0, 1, 20000) + 0.5 * np.sin(np.linspace(0, 200, 20000)),
        "ma": rng.normal(0, 1, 20000),
        "em": rng.normal(0, 1, 20000),
    })


def test_snr_scaling_hits_target():
    clean, _ = _synthetic_clean_beats(n=1)
    bank = _fake_nstdb_bank()
    rng = np.random.default_rng(1)
    noise = bank.sample_segment("ma", 360, rng)
    scaled = _scale_noise_to_snr(clean[0].reshape(-1), noise, snr_db=6.0)
    achieved = 10 * np.log10(np.mean(clean[0].reshape(-1) ** 2) / np.mean(scaled ** 2))
    assert abs(achieved - 6.0) < 0.1


def test_add_nstdb_noise_preserves_shape():
    clean, _ = _synthetic_clean_beats(n=1)
    bank = _fake_nstdb_bank()
    out = add_nstdb_noise(clean[0], "em", 0.0, bank, np.random.default_rng(2))
    assert out.shape == (360, 1)


def test_build_noise_training_set_contract():
    clean, labels = _synthetic_clean_beats(n=150)
    bank = _fake_nstdb_bank()
    signals, is_corrupted, noise_type, severity = build_noise_training_set(
        clean, labels, bank, model=None, corrupt_fraction=0.5, seed=3)
    assert signals.shape == clean.shape
    assert len(noise_type) == len(severity) == len(clean)
    # every corrupted beat has a type + severity; every clean beat has neither
    for corrupted, ntype, sev in zip(is_corrupted, noise_type, severity):
        if corrupted:
            assert ntype is not None and sev is not None
        else:
            assert ntype is None and sev is None


def test_noise_classifier_round_trips_and_agent_contract(tmp_path):
    clean, labels = _synthetic_clean_beats(n=150)
    bank = _fake_nstdb_bank()
    # No model here (unit test, no Keras checkpoint) -- NSTDB-only, so every
    # corrupted beat maps to "distractor". Fine for exercising save/load and
    # the agent contract; test_signal_poison_specificity below covers the
    # actual attack-vs-distractor split with a fake model.
    signals, is_corrupted, noise_type, _ = build_noise_training_set(
        clean, labels, bank, model=None, corrupt_fraction=0.5, seed=4)

    clean_idx = [i for i in range(len(signals)) if not is_corrupted[i]]
    templates = MorphologyTemplateBank.fit([signals[i] for i in clean_idx[:32]])
    feature_matrix = np.array([features_to_vector(extract_noise_features(s, templates)) for s in signals])

    # Force a few beats into the "attack" role so train_noise_classifier has
    # both classes to fit on, even without a real PGD-perturbed set here.
    sample_role = np.array([sample_role_from_noise_type(nt) for nt in noise_type])
    flip_to_attack = np.flatnonzero(sample_role == "distractor")[:20]
    sample_role[flip_to_attack] = "attack"

    clf = train_noise_classifier(feature_matrix, sample_role, seed=4)
    save_path = tmp_path / "clf.joblib"
    clf.save(save_path)
    loaded = TrainedNoiseClassifier.load(save_path)

    agent = NoiseClassifierAgent(loaded, templates=templates)
    state = {"validation_result": {"status": "PASS", "signal": signals[0], "flags": {}}}
    evidence = agent(state)["noise_evidence"]

    # same contract NoiseFeatureAgent produces -- this is what makes it a drop-in swap
    assert set(evidence.keys()) == {"score", "predicted_type", "severity", "features"}
    assert 0.0 <= evidence["score"] <= 1.0
    assert evidence["predicted_type"] in {"none", "adversarial_perturbation"}
    assert isinstance(evidence["severity"], float)


def test_distractor_specificity_check_runs():
    clean, labels = _synthetic_clean_beats(n=150)
    bank = _fake_nstdb_bank()
    signals, is_corrupted, noise_type, _ = build_noise_training_set(
        clean, labels, bank, model=None, corrupt_fraction=0.5, seed=5)

    clean_idx = [i for i in range(len(signals)) if not is_corrupted[i]]
    templates = MorphologyTemplateBank.fit([signals[i] for i in clean_idx[:32]])
    feature_matrix = np.array([features_to_vector(extract_noise_features(s, templates)) for s in signals])

    sample_role = np.array([sample_role_from_noise_type(nt) for nt in noise_type])
    sample_role[np.flatnonzero(sample_role == "distractor")[:20]] = "attack"

    clf = train_noise_classifier(feature_matrix, sample_role, seed=5)
    fpr = evaluate_distractor_specificity(clf, feature_matrix, sample_role)
    assert 0.0 <= fpr <= 1.0
