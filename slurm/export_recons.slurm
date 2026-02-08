#!/bin/bash
#SBATCH --job-name=export-recons
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

# Export reconstructions for downstream classification / segmentation.
# Edit the config if you need a different checkpoint or output directory.
python export.py --config configs/downstream.yaml
