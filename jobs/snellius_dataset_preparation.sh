#!/bin/bash
#SBATCH --job-name=kernelltl_dataset_prep
#SBATCH --output=logs/kernelltl_dataset_prep_%j.out
#SBATCH --error=logs/kernelltl_dataset_prep_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

# ============================================================================
# Snellius Job Script for KernelLTL Dataset Preparation
# ============================================================================
#
# This script generates train/eval datasets for curriculum learning stages.
# Each stage uses different complexity parameters (depth, number of formulas).
#
# Usage:
#   sbatch snellius_dataset_preparation.sh
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION
# ============================================================================

PROJECT_DIR="$HOME/KernelLTL"
VENV_DIR="$PROJECT_DIR/venv"

# Path to the saved kernel
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"

# Base output directory for datasets
DATASETS_BASE="$PROJECT_DIR/artifacts/datasets"

# ============================================================================
# CURRICULUM STAGE DEFINITIONS
# ============================================================================
# Format: "stage_name:k_samples:max_depth:p_leaf_range:eval_ratio"
# - k_samples: total number of formulas to sample
# - max_depth: maximum formula tree depth
# - p_leaf_range: probability of leaf node (higher = simpler formulas)
# - eval_ratio: fraction for evaluation set (using disjoint split)

STAGES=(
    "stage0:50000:1:0.3 0.6:0.05"
)
#    "stage1:100000:2:0.2 0.5:0.025"
#    "stage2:200000:3:0.1 0.5:0.025"
#    "stage3:400000:4:0.01 0.5:0.025"
#    "stage4:800000:5:0.01 0.4:0.025"

# Common options
TRAIN_DEDUPE=""
TRAIN_STORE_FORMULA_STR=""
TRAIN_STORE_SATISFACTION=""  # Add --train-store-satisfaction if needed
TRAIN_SATISFACTION_BATCH_SIZE=81920
TRAIN_SATISFACTION_TIME_INDEX=0
EVAL_DEDUPE="--eval-dedupe"
EVAL_STORE_FORMULA_STR="--eval-store-formula-str"
EVAL_STORE_SATISFACTION="--eval-store-satisfaction"  # Add --eval-store-satisfaction if needed
EVAL_SATISFACTION_BATCH_SIZE=81920
EVAL_SATISFACTION_TIME_INDEX=0


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Load modules
module purge
module load 2025
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

echo "Loaded modules:"
module list

# ============================================================================
# VIRTUAL ENVIRONMENT SETUP
# ============================================================================

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Using existing virtual environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

echo "Python version: $(python --version)"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ============================================================================
# GENERATE DATASETS FOR EACH STAGE
# ============================================================================

echo "=============================================="
echo "Generating curriculum datasets..."
echo "=============================================="

for stage_def in "${STAGES[@]}"; do
    # Parse stage definition
    IFS=':' read -r STAGE_NAME K_SAMPLES MAX_DEPTH P_LEAF_RANGE EVAL_RATIO <<< "$stage_def"
    
    TRAIN_OUT="$DATASETS_BASE/$STAGE_NAME/train"
    EVAL_OUT="$DATASETS_BASE/$STAGE_NAME/eval"
    
    echo ""
    echo "=============================================="
    echo "Generating dataset for $STAGE_NAME"
    echo "  - Total samples: $K_SAMPLES"
    echo "  - Max depth: $MAX_DEPTH"
    echo "  - P_leaf range: $P_LEAF_RANGE"
    echo "  - Eval ratio: $EVAL_RATIO"
    echo "  - Train output: $TRAIN_OUT"
    echo "  - Eval output: $EVAL_OUT"
    echo "=============================================="
    
    # Create output directories
    mkdir -p "$TRAIN_OUT"
    mkdir -p "$EVAL_OUT"
    
    # Build command
    CMD=(
        python -u scripts/prepare_datasets.py
        --kernel-dir "$KERNEL_DIR"
        --disjoint-split
        --eval-ratio "$EVAL_RATIO"
        --train-out "$TRAIN_OUT"
        --train-k "$K_SAMPLES"
        --train-p-leaf-range $P_LEAF_RANGE
        --train-max-depth "$MAX_DEPTH"
        $TRAIN_STORE_FORMULA_STR
        $TRAIN_STORE_SATISFACTION
        --train-satisfaction-batch-size "$TRAIN_SATISFACTION_BATCH_SIZE"
        --train-satisfaction-time-index "$TRAIN_SATISFACTION_TIME_INDEX"
        --eval-out "$EVAL_OUT"
        $EVAL_DEDUPE

    )
    
    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
    
    echo "$STAGE_NAME dataset generation complete!"
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "All datasets generated successfully!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Dataset locations:"
for stage_def in "${STAGES[@]}"; do
    IFS=':' read -r STAGE_NAME _ _ _ _ <<< "$stage_def"
    echo "  $STAGE_NAME:"
    echo "    Train: $DATASETS_BASE/$STAGE_NAME/train"
    echo "    Eval:  $DATASETS_BASE/$STAGE_NAME/eval"
done
echo ""