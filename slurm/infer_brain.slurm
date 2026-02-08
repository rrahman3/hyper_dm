#!/bin/bash
#SBATCH --job-name=infer-brain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

CKPT=${CKPT:-"checkpoints/hnet_brain_final.ckpt"}
MODE=${MODE:-"eu"}

python infer.py --config configs/brain_multicoil.yaml \
    --ckpt "${CKPT}" \
    --split test \
    training.mode="${MODE}"
