#!/bin/bash
#SBATCH --job-name=brain-eu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

python train.py --config configs/brain_multicoil.yaml \
    training.mode=eu
