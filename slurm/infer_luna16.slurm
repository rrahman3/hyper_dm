#!/bin/bash
#SBATCH --job-name=infer-luna16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

# ── Set the checkpoint path for the mode you trained ──
CKPT=${CKPT:-"checkpoints/hnet_final.ckpt"}
MODE=${MODE:-"standard"}                 # standard | eu | au

python infer.py --config configs/ct_luna16.yaml \
    --ckpt "${CKPT}" \
    --split test \
    training.mode="${MODE}"
