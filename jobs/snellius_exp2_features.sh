#!/bin/bash
#SBATCH --job-name=kernelltl_exp2_features
#SBATCH --output=logs/kernelltl_exp2_features_%j.out
#SBATCH --error=logs/kernelltl_exp2_features_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

# ============================================================================
# Experiment 2 feature extraction (geometry + faithfulness), CPU-only.
#
# One streaming pass over the validation satisfactions tensor; caches per-target
# features (exp2_features.csv) plus the landmark pair matrices and the anchor
# covariance Gram for post-hoc faithfulness analyses. Includes a hard gate that
# recomputes a sample of embeddings from F_c @ satvec_c / N and aborts on
# mismatch -- i.e. it verifies that KERNEL_DIR is the kernel that built the
# dataset before any feature is trusted.
#
# Depends only on the dataset + kernel artifacts (no model, no GPU).
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
VALIDATION_DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/exp2/features"

N_LANDMARKS=256
SEED=0

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Experiment 2 feature extraction"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs" "$OUTPUT_DIR"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "$HOME_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

# ---------------------------- RUN ------------------------------------------
CMD=(
    python -u scripts/analysis_exp2/compute_features.py
    --validation-dataset-dir "$VALIDATION_DATASET_DIR"
    --kernel-dir "$KERNEL_DIR"
    --output-dir "$OUTPUT_DIR"
    --n-landmarks "$N_LANDMARKS"
    --seed "$SEED"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Feature extraction complete!  End: $(date)"
ls -la "$OUTPUT_DIR"
echo "=============================================="
