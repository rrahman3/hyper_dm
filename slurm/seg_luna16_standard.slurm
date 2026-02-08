#!/bin/bash
#SBATCH --job-name=seg-luna16-std
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

python train.py --config configs/seg_luna16_standard.yaml
