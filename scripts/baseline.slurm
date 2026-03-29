#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=autoresearch-baseline
#SBATCH --output=/p/scratch/westai0047/autoresearch/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/autoresearch/logs/%x_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
# Compute nodes have no internet — use local HF cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_DIR="${PROJECT}/autoresearch"

mkdir -p "${SCRATCH}/autoresearch/logs"

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Autoresearch baseline run"
echo "==> Start: $(date)"

cd "$REPO_DIR"

# Use uv run directly (venv has torch+cu128 and all deps)
uv run train.py

echo "==> Done: $(date)"
