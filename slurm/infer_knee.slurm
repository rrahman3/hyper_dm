#!/bin/bash
#SBATCH --job-name=infer-knee
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

CKPT=${CKPT:-"checkpoints/hnet_knee_final.ckpt"}
MODE=${MODE:-"standard"}

python infer.py --config configs/knee_singlecoil.yaml \
    --ckpt "${CKPT}" \
    --split test \
    training.mode="${MODE}"
