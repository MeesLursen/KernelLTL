#!/bin/bash
#SBATCH --job-name=kernelltl_dataset_prep_finetune
#SBATCH --output=logs/kernelltl_dataset_prep_finetune_%j.out
#SBATCH --error=logs/kernelltl_dataset_prep_finetune_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120GB

set -euo pipefail

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Source assets
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
STAGE_TRAIN_DIRS=(
    "$PROJECT_DIR/artifacts/datasets/stage1/train"
    "$PROJECT_DIR/artifacts/datasets/stage2/train"
    "$PROJECT_DIR/artifacts/datasets/stage3/train"
    "$PROJECT_DIR/artifacts/datasets/stage4/train"
)
EXCLUDE_DATASET_DIRS=(
    "$PROJECT_DIR/artifacts/datasets/stage0/train"
    "$PROJECT_DIR/artifacts/datasets/stage4/eval"
)

# Optional exclusion files with one formula string per line (for example depth0/depth1 catalogs).
EXCLUDE_FORMULA_FILES=(
)

# Output target on project storage.
FINAL_OUTPUT_DIR="$PROJECT_DIR/artifacts/datasets/finetune/train"
mkdir -p "$FINAL_OUTPUT_DIR"

# Finetuning dataset generation parameters.
SAMPLE_COUNT=20000
EQUIVALENT_MUTATIONS_PER_FORMULA=2
NEAR_MISS_MUTATIONS_PER_FORMULA=1
SATISFACTION_BATCH_SIZE=614400
SATISFACTION_TIME_INDEX=0
SEED=69


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=============================================="
echo "KernelLTL Finetune Dataset Preparation"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"

# Load modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"

# Setup virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs visible: $NUM_GPUS"
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Expected at least 1 GPU, found $NUM_GPUS"
    exit 1
fi

# ============================================================================
# RUN DATASET CREATION
# ============================================================================
CMD=(
    python -u scripts/prepare_finetune_dataset.py
    --kernel-dir "$KERNEL_DIR"
    --stage-train-dirs "${STAGE_TRAIN_DIRS[@]}"
    --output-dir "$FINAL_OUTPUT_DIR"
    --sample-count "$SAMPLE_COUNT"
    --equivalent-mutations-per-formula "$EQUIVALENT_MUTATIONS_PER_FORMULA"
    --near-miss-mutations-per-formula "$NEAR_MISS_MUTATIONS_PER_FORMULA"
    --satisfaction-batch-size "$SATISFACTION_BATCH_SIZE"
    --satisfaction-time-index "$SATISFACTION_TIME_INDEX"
    --seed "$SEED"
)

if [ ${#EXCLUDE_DATASET_DIRS[@]} -gt 0 ]; then
    CMD+=(--exclude-dataset-dirs "${EXCLUDE_DATASET_DIRS[@]}")
fi

if [ ${#EXCLUDE_FORMULA_FILES[@]} -gt 0 ]; then
    CMD+=(--exclude-formula-files "${EXCLUDE_FORMULA_FILES[@]}")
fi

echo "Running finetune dataset generation..."
"${CMD[@]}"


echo "Finetune dataset preparation finished successfully."
echo "Output: $FINAL_OUTPUT_DIR"
echo "End time: $(date)"
