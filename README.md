# Data-Poisoning-in-Healthcare

## 1) Project overview
This repository is used for experiments on data poisoning in healthcare datasets, primarily through a Google Colab notebook. It also implements an agentic AI pipeline, where specialized agents collaboratively perform ECG validation, noise and poison detection, recovery, and reporting to improve the robustness of healthcare AI systems.

## Getting Started (quick steps)
1. Clone this repository
2. Check out the `dev` branch
3. Open the notebook in Google Colab
4. Create a feature branch before making changes

## Integrated agent pipeline

Every agent in the architecture diagram is wired together by `orchestrator/pipeline.py`:

```
Validation ──REJECT──────────────────────────────────────────────┐
   │ PASS / REPAIR                                               │
   ├──> Label poisoning detection ──┐                            │
   └──> Signal poisoning detection ─┴──> Decision                │
                                          │ proceed   │ repair / quarantine
                                          │           v          │
                                          │        Recovery ─quarantined/rejected─┤
                                          v           │ repaired (re-validated)   │
                                       Diagnosis <────┘                           │
                                          v                                       │
                                    Explainability ──> Alert & reporting <────────┘
```

| Diagram box | Code |
|---|---|
| Validation agent (format check, signal sanity, triage) | `agents/validation/` |
| Label poisoning: learning dynamics (AUM) | `agents/label_poisoning/learning_dynamics.py` |
| Label poisoning: confident learning (k-fold OOF) | `agents/label_poisoning/confident_learning.py` |
| Label poisoning: latent + attention (k-NN, Mahalanobis, entropy) | `agents/label_poisoning/latent_attention.py` |
| Label poisoning: score fusion (per-class calibration) | `agents/label_poisoning/score_fusion.py` |
| Signal poisoning: reconstruction check (autoencoder) | `agents/signal_poisoning/reconstruction.py` |
| Signal poisoning: feature-space distance (Mahalanobis) | `agents/signal_poisoning/feature_distance.py` |
| Signal poisoning: prediction fragility, gradient sensitivity | `agents/signal_poisoning/fragility.py` |
| Spectral noise features + trained noise classifier (optional 5th signal check) | `agents/signal_poisoning/noise/` |
| Decision agent | `agents/decision/` |
| Recovery agent | `agents/recovery/` |
| Diagnosis agent (DenseNet+BiLSTM+Attn) | `agents/diagnosis/`, `models/` |
| Explainability agent (attention maps, SHAP) | `agents/explainability/` |
| Alert & reporting | `agents/reporting/` |

Shared contracts: `core/ecg_state.py` (the `ECGState` every node reads and writes) and
`shared/contracts.py` (`Evidence`, `DetectionReport`, `DecisionRecord`, `RecoveryResult`).
`models/backend.py` gives every agent one interface to the classifier — `KerasBackend` for
`DenseNet1D_clean.keras`, `TorchBackend` for the PyTorch port.

### What each agent does (per `Agentic_AI_Pipeline.pdf`)

| PDF step | Implementation |
|---|---|
| 1. Validation — integrity, format, sampling | shape/NaN/range/flatline/clipping/amplitude checks, sampling-rate check (`metadata["fs"]` must be 360 Hz), PASS / REPAIR / REJECT. REJECT → report says "request a new recording". `ValidationConfig.zscored_beats()` is the preset for the ECG_Splits data (used by default in the pipeline). |
| 2. Poison detection — label attacks | prediction-label disagreement, confidence, AUM, out-of-fold probabilities, k-NN / Mahalanobis / entropy → one calibrated attack probability (`label_score`) |
| 3. Noise detection — signal quality & signal attacks | 4 attack checks (reconstruction, Mahalanobis, fragility, gradient) + signal quality: SNR, spectral entropy, baseline wander / powerline / EMG ratios → noise type and severity |
| 4. Decision — reasoning, attack type, severity | verdict (clean / label / signal / both / uncertain), `attack_type` (label_flip, waveform_corruption, adversarial_perturbation, …), `severity` (low / moderate / high), whether recovery is needed |
| 5. Recovery — remove / denoise / relabel / review | signal: autoencoder reconstruction, else Butterworth 0.5–40 Hz + wavelet denoising; the repaired beat is re-validated and re-checked and referred for expert review if still flagged. Label: relabel from out-of-fold consensus, or review. `pipeline.audit_dataset()` lists training samples to remove. |
| 6. Diagnosis — DenseNet + BiLSTM + Attention | your `DenseNet1D_clean.keras` through `KerasBackend` |
| 7. Explainability | attention maps, gradient saliency, segment-level SHAP values |
| 8. Alert & reporting | alert level, validation result, attack type / severity, recovery action, diagnosis + confidence, top regions, signal quality, recommendations for clinical review |

### Run it

Put `DenseNet1D_clean.keras` and the ECG_Splits `train.csv`, `val.csv`, `test.csv` in `artifacts/`
(gitignored; the CSVs are cached as `.npz` on first load).

```bash
python -m venv venv && venv/Scripts/activate      # Windows; source venv/bin/activate elsewhere
pip install -r requirements.txt tensorflow
python -m pytest -q                               # unit, end-to-end and real-model tests
python -m scripts.run_real_pipeline --n 60        # real model + data, every attack scenario (~7 min)
python -m scripts.build_report                    # PDF: documentation + results (needs reportlab)
python -m scripts.run_pipeline_demo --graph       # synthetic beats through the LangGraph graph
```

From code:

```python
from evaluation.data import load_split
from models.backend import KerasBackend
from orchestrator.pipeline import ECGDefensePipeline

backend = KerasBackend.load("artifacts/DenseNet1D_clean.keras")
train, val = load_split("train"), load_split("val")
pipeline = ECGDefensePipeline.from_reference_data(
    backend,
    clean_signals=train.signals[:4000],                                   # reference for signal checks + SHAP
    audit_signals=train.signals, audit_labels=train.labels, audit_record_ids=train.beat_ids,  # label audit
    calibration_signals=val.signals[:600], calibration_labels=val.labels[:600])  # unseen patients
state = pipeline.run(beat, beat_id="rec_0001", label=claimed_label)
print(state["report"]["text"])
suspicious_training_samples = pipeline.audit_dataset()
```

Always pass `calibration_*` from patients that are in neither the reference nor the audit set:
calibrating on the training patients flagged 62% of clean test beats; calibrating on the
validation split brings that to 20%.

### Results on the real model and data

`python -m scripts.run_real_pipeline --n 60`: detectors fitted on 4,000 training beats, label audit
on 8,000 training beats with 400 labels deliberately flipped, calibrated on 600 validation beats,
then 60 unseen test beats per scenario. 1.06 s per beat through all eight agents on CPU.

| scenario | flagged | notes |
|---|---|---|
| training-set label audit | 400/400 flips found | precision 0.82 (488 flagged) |
| clean test beats | 12/60 (20%) | false positives |
| label flipped (unseen beats) | 38/60 | |
| Gaussian noise σ = 0.01 / 0.05 | same as clean | imperceptible in z-scored beats; does not change diagnoses either |
| PGD, ε = 0.05 | 40/60 | but 34 of them blamed on the label (see below) |

### Known limitations (calibrate before trusting verdicts)

- **False-positive rate is still 20% on unseen patients.** The thresholds in
  `agents/decision/config/thresholds.yaml` are uncalibrated starting points. Sweep them with
  `agents/decision/evaluation/evaluate_fusion.py` on a larger validation run.
- **Signal fusion policy.** With the default `weighted_mean` fusion one strongly-firing signal
  check is averaged down by three calm ones: on real PGD beats only 10% cross the signal
  threshold with `weighted_mean` vs 67% with `noisy_or` (43% vs 10% on clean beats). Choose the
  strategy together with the thresholds.
- **Signal attacks get blamed on the label.** For beats outside the audited dataset the label
  checks see the classifier's view of the tampered signal, so a successful adversarial example
  looks like a mislabeled beat: 34 of 60 PGD beats came out `LABEL_POISONED`. Recovery then
  relabels or quarantines instead of repairing the signal. A fusion policy where the signal side
  also fires turns these into `BOTH_POISONED` → quarantine.
- **Class mapping.** The model and pipeline use 4 classes (A, B, N, V; B = merged LBBB/RBBB).
  `notebooks/mitbih_preprocessing.ipynb` still encodes 5 (A, L, N, R, V) — merge L/R before
  retraining.

## 2) Repository structure
- `main` branch
  - Contains only `README.md` and `.gitignore`
  - Stable branch for reviewed, finalized content
  - Do not merge direct changes without review
- `dev` branch
  - Contains the Google Colab notebooks (`notebooks/`) and the integrated agent pipeline
  - All development and experimentation happen here

## 3) Branch workflow
1. Create new branches **from `dev`**
2. Make changes in your feature branch
3. Open a PR and merge back into **`dev`**
4. Merge into **`main`** only after review and stabilization

## 4) Open and use the notebook in Google Colab
1. Go to the repository on GitHub and switch to the `dev` branch
2. Open the notebook file
3. Click **Open in Colab** (or copy the GitHub URL into Colab)
4. Run cells in order and save updates

## 5) GitHub + Colab workflow
Use Colab for notebook edits and GitHub for branch/PR workflow:
- Pull the latest changes from `dev`
- Create/update your feature branch
- Commit and push from Colab (or use **File → Save a copy in GitHub** for notebook commits)
- Open PR to `dev`

## 6) Colab Git commands (clone, commit, push)
Run these in a Colab cell (prefix with `!`):

```bash
!git clone https://github.com/Nihar654/Data-Poisoning-in-Healthcare.git
%cd Data-Poisoning-in-Healthcare
!git checkout dev
!git checkout -b feature/<short-name>

# after notebook changes
!git add .
!git commit -m "feat: short description of change"
!git push -u origin feature/<short-name>
```

> **Note:** GitHub authentication from Google Colab may require a Personal Access Token (PAT) instead of your GitHub password when pushing changes.

## 7) Team collaboration guidelines
- Keep active work in `dev`-based branches only
- Keep PRs small and focused
- Review before merging
- Do not push experimental work to `main`
- Avoid editing the same notebook at the same time to reduce Jupyter/Colab merge conflicts
- Resolve review comments before merge

## 8) Commit message guidelines
Use short, clear messages:
- `feat: add baseline poisoning experiment`
- `fix: correct data split in notebook`
- `docs: update Colab workflow steps`

Format:
- `<type>: <what changed>`
- Common types: `feat`, `fix`, `docs`, `refactor`, `chore`
