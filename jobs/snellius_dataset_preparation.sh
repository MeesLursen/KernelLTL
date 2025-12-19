#!/bin/bash
#SBATCH --job-name=kernelltl_prep
#SBATCH --output=logs/kernelltl_prep_%j.out
#SBATCH --error=logs/kernelltl_prep_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G

# ============================================================================
# Snellius Job Script for KernelLTL Dataset Preparation
# ============================================================================
#
# This script generates train/eval datasets for curriculum learning stages.
# Each stage uses different complexity parameters (depth, number of formulas).
#
# Usage:
#   sbatch snellius_prepare_datasets.sh
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
DATASETS_BASE="$PROJECT_DIR/datasets"

# ============================================================================
# CURRICULUM STAGE DEFINITIONS
# ============================================================================
# Format: "stage_name:k_samples:max_depth:p_leaf:eval_ratio"
# - k_samples: total number of formulas to sample
# - max_depth: maximum formula tree depth
# - p_leaf: probability of leaf node (higher = simpler formulas)
# - eval_ratio: fraction for evaluation set (using disjoint split)

STAGES=(
    "stage1:50000:2:0.5:0.05"
    "stage2:100000:3:0.45:0.05"
    "stage3:200000:4:0.4:0.05"
)

# Common options
DEDUPE="--train-dedupe"
STORE_FORMULA_STR="--train-store-formula-str"
STORE_SATISFACTION=""  # Add --train-store-satisfaction if needed
SATISFACTION_BATCH_SIZE=1024
SATISFACTION_TIME_INDEX=0

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
module load 2023
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
    IFS=':' read -r STAGE_NAME K_SAMPLES MAX_DEPTH P_LEAF EVAL_RATIO <<< "$stage_def"
    
    TRAIN_OUT="$DATASETS_BASE/$STAGE_NAME/train"
    EVAL_OUT="$DATASETS_BASE/$STAGE_NAME/eval"
    
    echo ""
    echo "=============================================="
    echo "Generating dataset for $STAGE_NAME"
    echo "  - Total samples: $K_SAMPLES"
    echo "  - Max depth: $MAX_DEPTH"
    echo "  - P(leaf): $P_LEAF"
    echo "  - Eval ratio: $EVAL_RATIO"
    echo "  - Train output: $TRAIN_OUT"
    echo "  - Eval output: $EVAL_OUT"
    echo "=============================================="
    
    # Create output directories
    mkdir -p "$TRAIN_OUT"
    mkdir -p "$EVAL_OUT"
    
    # Build command
    CMD=(
        python scripts/prepare_datasets.py
        --kernel-dir "$KERNEL_DIR"
        --disjoint-split
        --eval-ratio "$EVAL_RATIO"
        --train-out "$TRAIN_OUT"
        --train-k "$K_SAMPLES"
        --train-p-leaf "$P_LEAF"
        --train-max-depth "$MAX_DEPTH"
        --eval-out "$EVAL_OUT"
        $DEDUPE
        $STORE_FORMULA_STR
    )
    
    # Add satisfaction options if enabled
    if [ -n "$STORE_SATISFACTION" ]; then
        CMD+=(
            "$STORE_SATISFACTION"
            --train-satisfaction-batch-size "$SATISFACTION_BATCH_SIZE"
            --train-satisfaction-time-index "$SATISFACTION_TIME_INDEX"
        )
    fi
    
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