#!/bin/bash
#SBATCH --job-name=luna16-au
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

python train.py --config configs/ct_luna16.yaml \
    training.mode=au
