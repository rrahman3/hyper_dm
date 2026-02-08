#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Submit all Hyper-DM training jobs to Slurm.
#
# Usage:
#   bash slurm/submit_all.sh              # submit everything
#   bash slurm/submit_all.sh recon        # only reconstruction
#   bash slurm/submit_all.sh seg          # only LUNA16 segmentation
#   bash slurm/submit_all.sh cls          # only MRI classification
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure logs/ directory exists
mkdir -p logs

GROUP="${1:-all}"

submit() { echo "Submitting $1 ..."; sbatch "$1"; }

# ── Reconstruction training (LUNA16 / Knee / Brain × standard/EU/AU) ──
if [[ "$GROUP" == "all" || "$GROUP" == "recon" ]]; then
    echo "=== Reconstruction jobs ==="
    submit slurm/train_luna16_standard.slurm
    submit slurm/train_luna16_eu.slurm
    submit slurm/train_luna16_au.slurm
    submit slurm/train_knee_standard.slurm
    submit slurm/train_knee_eu.slurm
    submit slurm/train_knee_au.slurm
    submit slurm/train_brain_standard.slurm
    submit slurm/train_brain_eu.slurm
    submit slurm/train_brain_au.slurm
fi

# ── LUNA16 segmentation (standard / EU / AU) ──
if [[ "$GROUP" == "all" || "$GROUP" == "seg" ]]; then
    echo "=== Segmentation jobs ==="
    submit slurm/seg_luna16_standard.slurm
    submit slurm/seg_luna16_eu.slurm
    submit slurm/seg_luna16_au.slurm
fi

# ── MRI classification (Knee & Brain × standard/EU/uncertainty) ──
if [[ "$GROUP" == "all" || "$GROUP" == "cls" ]]; then
    echo "=== Classification jobs ==="
    submit slurm/cls_knee_standard.slurm
    submit slurm/cls_knee_eu.slurm
    submit slurm/cls_knee_uncertainty.slurm
    submit slurm/cls_brain_standard.slurm
    submit slurm/cls_brain_eu.slurm
    submit slurm/cls_brain_uncertainty.slurm
fi

echo "Done. Check queue with: squeue -u \$USER"
