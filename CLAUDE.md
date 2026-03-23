# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABR (Auditory Brainstem Response) Transformer V2 — a diffusion-based deep learning system for ABR signal generation, reconstruction, and classification. Uses V-prediction parameterization with a flat Transformer architecture (no U-Net/S4), FiLM conditioning from clinical parameters, and multi-task learning (signal reconstruction + peak detection + hearing loss classification).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Training
python train.py --config configs/train.yaml
python train.py --config configs/train_hpo_optimized.yaml   # HPO-optimized settings

# Evaluation
python eval.py --config configs/eval.yaml

# Hyperparameter optimization
python scripts/train_with_hpo.py --config configs/hpo_search_space.yaml

# Tests
pytest tests/                    # all tests
pytest tests/test_abr_transformer.py -v   # single test file
pytest --cov=. tests/            # with coverage

# Formatting
black .
isort .
flake8 .
```

## Architecture

### Core Pipeline
- **`train.py`** (main training loop) → loads config, builds model/dataset, runs training with EMA, AMP, gradient clipping, TensorBoard logging, checkpointing
- **`eval.py`** (evaluation pipeline) → reconstruction/generation modes, 52+ metrics, ROC/PR analysis, clinical validation
- **`configs/train.yaml`** — primary training config; all features controlled via YAML flags

### Model (`models/abr_transformer.py`)
`ABRTransformerGenerator`: ~6.56M params. Input/output shape: `[B, 1, 200]`.
- **MultiScaleStem**: Conv1D branches (kernels 3,7,15) → d_model(256)
- **FiLM conditioning**: static_dim=4 (Age, Intensity, StimulusRate, FMP) modulates features
- **6 Transformer layers**: 8-head attention + FFN
- Ablation flags control optional features: `use_cross_attention`, `joint_static_generation`, `use_learned_pos_emb`, `use_multi_scale_fusion`, `use_advanced_blocks`

### Model Blocks (`models/blocks/`)
- `transformer_block.py` — MultiHeadAttention, ConvModule
- `film.py` — FiLM layers, ConditionalEmbedding, TokenFiLM
- `positional.py` — learned and sinusoidal positional embeddings
- `heads.py` — output heads, attention pooling

### Diffusion (`utils/schedules.py`, `inference/sampler.py`)
- V-prediction: `x₀ = √(ᾱₜ)·xₜ - √(1-ᾱₜ)·v_θ(xₜ, c, t)`
- Cosine beta schedule, DDIM sampling (50-1000 steps)
- Classifier-free guidance support

### Data Pipeline (`data/`)
- **`dataset.py`**: `ABRDataset` loads from `.pkl` files; returns signal[200], static_params[4], target, v_peak[2], patient_id
- **`preprocessing.py`**: Excel→pickle pipeline with sweep rejection, SNR validation, clinical threshold computation
- **`augmentations.py`**: MixUp and CutMix preserving ABR morphology
- **`curriculum.py`**: SNR-based difficulty metrics, progressive sampling

### Training Utilities (`training/`)
- `monitoring.py` — real-time metrics, gradient flow analysis, dead neuron detection, HTML dashboards
- `early_stopping.py` — multi-strategy early stopping with warmup
- `ensemble.py` — snapshot ensembles, cross-validation ensembles, uncertainty quantification
- `distillation.py` — knowledge distillation (feature-level and output-level)
- `cross_validation.py` — stratified group K-fold (patient-level, no data leakage)

### Evaluation (`evaluation/`)
- `metrics.py` — 52+ metrics (MSE, SNR, PSNR, SSIM, spectral similarity, DTW, peak F1, ROC-AUC)
- `analysis.py` — ROC analysis, bootstrap CIs, statistical significance, clinical validation, threshold optimization
- `visualization.py` — publication-ready figures (300 DPI, serif fonts, colorblind-friendly)

### Losses (`utils/losses.py`)
`CombinedLoss` weights: signal reconstruction (MSE/MAE), STFT loss, peak detection (BCE/Focal), classification (CE), static parameter reconstruction. Weights configured in YAML.

## Configuration System

YAML-based via OmegaConf. Configs support environment variable substitution, reference resolution, and dot-notation CLI overrides. Key configs:
- `configs/train.yaml` — full training config with ablation flags
- `configs/eval.yaml` — evaluation settings
- `configs/hpo_search_space.yaml` — Optuna HPO search space
- `configs/ablation_configs/` — systematic feature ablation configs

## Data

Primary dataset: `data/processed/ultimate_dataset_with_clinical_thresholds.pkl` (~152MB). Stratified splitting by patient_id prevents data leakage. 5 hearing loss classes.

## Key Conventions

- Signal tensors are always `[B, 1, 200]` (batch, channel, sequence_length=200)
- Static parameters are `[B, 4]`: Age, Intensity, StimulusRate, FMP
- Model forward returns a dict with keys like `v_pred`, `peak_logits`, `static_pred` (not a bare tensor)
- V-prediction parameterization throughout (not epsilon-prediction)
- All training features are toggled via config flags, not code changes
