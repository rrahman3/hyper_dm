#!/bin/bash
#SBATCH --job-name=cls-knee-unc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

python train.py --config configs/cls_knee_uncertainty.yaml
