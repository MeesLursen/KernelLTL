#!/bin/bash
#SBATCH --job-name=kernelltl_dataset_prep_validation
#SBATCH --output=logs/kernelltl_dataset_prep_validation_%j.out
#SBATCH --error=logs/kernelltl_dataset_prep_validation_%j.err
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

KERNEL_DIR="$HOME_DIR/artifacts/kernel"

EXCLUDE_DATASET_DIRS=(
    "$PROJECT_DIR/artifacts/datasets/stage4/train"
    "$PROJECT_DIR/artifacts/datasets/stage4/eval"
    "$PROJECT_DIR/artifacts/datasets/finetune/train"
)

# Optional files with one formula string per line
EXCLUDE_FORMULA_FILES=(
)

FINAL_OUTPUT_DIR="$PROJECT_DIR/artifacts/datasets/validation"
mkdir -p "$FINAL_OUTPUT_DIR"

# Validation target: 5k formulas at each depth 2,3,4,5
DEPTH_TARGETS=(
    "2:5000"
    "3:5000"
    "4:5000"
    "5:5000"
)

P_LEAF_RANGE=(0.2 0.6)
MIN_DEPTH=2
MAX_DEPTH=5
MAX_SAMPLING_ATTEMPTS=1000
SATISFACTION_BATCH_SIZE=256000
SATISFACTION_TIME_INDEX=0

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=============================================="
echo "KernelLTL Validation Dataset Preparation"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"

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
# RUN VALIDATION DATASET CREATION
# ============================================================================
CMD=(
    python -u scripts/prepare_validation_dataset.py
    --kernel-dir "$KERNEL_DIR"
    --exclude-dataset-dirs "${EXCLUDE_DATASET_DIRS[@]}"
    --output-dir "$FINAL_OUTPUT_DIR"
    --p-leaf-range "${P_LEAF_RANGE[@]}"
    --min-depth "$MIN_DEPTH"
    --max-depth "$MAX_DEPTH"
    --depth-targets "${DEPTH_TARGETS[@]}"
    --batch-size "$SATISFACTION_BATCH_SIZE"
    --time-index "$SATISFACTION_TIME_INDEX"
    --max-sampling-attempts "$MAX_SAMPLING_ATTEMPTS"
)

if [ ${#EXCLUDE_FORMULA_FILES[@]} -gt 0 ]; then
    CMD+=(--exclude-formula-files "${EXCLUDE_FORMULA_FILES[@]}")
fi

echo "Running validation dataset generation..."
"${CMD[@]}"

echo "Validation dataset preparation finished successfully."
echo "Output: $FINAL_OUTPUT_DIR"
echo "End time: $(date)"
