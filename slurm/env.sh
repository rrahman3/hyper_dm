#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# UNCC HPC — Hyper-DM: Common environment setup
# Source this from other slurm scripts:  source slurm/env.sh
# ──────────────────────────────────────────────────────────────────

# Adjust these to match your HPC environment
module purge
module load anaconda3
module load cuda/12.1

# Activate your conda environment (create once with: conda create -n hyperdm python=3.10)
conda activate hyperdm

# Ensure the hyper_dm package is importable
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Logging helpers
echo "──────────────────────────────────────────"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job Name     : ${SLURM_JOB_NAME}"
echo "Node         : $(hostname)"
echo "GPUs         : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start Time   : $(date)"
echo "Working Dir  : ${SLURM_SUBMIT_DIR}"
echo "──────────────────────────────────────────"
nvidia-smi || true
echo "──────────────────────────────────────────"
