# Hyper-DM — Hyper-Diffusion Models for Uncertainty-Aware Medical Image Reconstruction

A clean, config-driven PyTorch implementation of **Hyper-Diffusion Models (Hyper-DM)** for medical image reconstruction with principled uncertainty quantification.

A HyperNetwork generates diverse weight sets for a frozen U-Net backbone acting as the noise predictor in a DDPM/DDIM diffusion framework. Uncertainty is decomposed into:

- **Epistemic (EU)** — variance across *M* HyperNetwork weight samples (model uncertainty)
- **Aleatoric (AU)** — variance across *N* DDIM reverse-process draws per weight set (data uncertainty)


## Quick Start

### Installation

```bash
cd hyper_dm
pip install -e ".[dev,mri]"
```

### Training

All experiments are configured via YAML files. Override any parameter from the CLI:

```bash
# Train Hyper-DM on LUNA16 CT (standard mode)
python train.py --config configs/ct_luna16.yaml

# Train with epistemic uncertainty on fastMRI knee
python train.py --config configs/knee_singlecoil.yaml training.mode=eu

# Override hyperparameters from CLI
python train.py --config configs/ct_luna16.yaml \
    training.lr=1e-4 \
    training.epochs=50 \
    model.z_dim=16
```

### Inference

```bash
# Reconstruct with uncertainty maps (M=10, N=10)
python infer.py --config configs/ct_luna16.yaml \
    --ckpt checkpoints/hnet_ct.pt \
    --M 10 --N 10 --outdir results/ct/
```

### Export for Downstream

```bash
# Generate reconstructed slices for nodule segmentation
python export.py --config configs/ct_luna16.yaml \
    --ckpt checkpoints/hnet_ct.pt \
    --outdir data/recon_ct/
```

### Tests

```bash
pytest tests/ -v
```

## Config System

Each YAML config has four sections:

| Section     | Controls                                    |
|-------------|---------------------------------------------|
| `data`      | Dataset type, paths, acceleration, image size |
| `model`     | Backbone architecture, base channels, z_dim  |
| `diffusion` | Timesteps T, beta range, DDIM steps          |
| `training`  | Mode (standard/eu/au), lr, epochs, M, λ_EU   |

Example — override any `key.subkey=value` from the command line.

## Training Modes

| Mode       | Loss                                       | Purpose                 |
|------------|--------------------------------------------|-------------------------|
| `standard` | MSE noise prediction                       | Baseline                |
| `eu`       | MSE − λ_EU · pairwise_L1(M samples)       | Epistemic uncertainty   |
| `au`       | MSE with noise-augmented conditioning      | Aleatoric uncertainty   |
