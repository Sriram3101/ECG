# Agentic ECG defense

LangGraph-based ECG safety pipeline for 360-sample, single-lead beats. It includes Sriram's noise-feature agent (spectral bands, wavelet residual, morphology templates), a validation node, and stable integration placeholders for the poison and decision agents.

## Shared contracts

- `ECGState` is defined in `ecg_defense/contracts.py`; it is the single cross-team state contract.
- Validation always returns `{"status": "PASS"|"REPAIRED"|"REJECT", "signal": array, "flags": {...}}`.
- Noise evidence always contains one `score: float` in `[0, 1]` per beat.
- Corruption truth is evaluation-only and deliberately saved separately as `*.truth.json` by `datasets.py`.
- **Mapping warning:** the supplied training script uses scikit-learn's sorted labels: `A=0, B=1, N=2, V=3`, where `B` is merged LBBB/RBBB. Confirm with the team before changing it.

## Setup and run

```powershell
python -m pip install -r requirements.txt
python scripts/run_noise_demo.py
python -m pytest -q
```

## Using the trained model

The supplied Keras checkpoint can remain outside the repository. Place a copy at `artifacts/DenseNet1D_clean.keras` when the classifier integration is ready. The current noise feature agent needs no TensorFlow/model inference; Nihar's classifier can replace `NoiseFeatureAgent` while preserving its `noise_evidence` contract.

## Dataset contract

Signal artifacts contain `(N, 360, 1)` `signals`, unchanged `labels`, and `record_ids`. Corruption truth (`is_corrupted`, `noise_type`, `severity`) is stored separately. `add_gaussian_noise(..., sigma=0.01)` exactly matches the current attack experiment: zero-mean noise in the post-z-score feature space, no clipping.
