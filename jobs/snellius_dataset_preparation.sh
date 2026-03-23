#!/bin/bash
#SBATCH --job-name=kernelltl_dataset_prep
#SBATCH --output=logs/kernelltl_dataset_prep_%j.out
#SBATCH --error=logs/kernelltl_dataset_prep_%j.err
#SBATCH --time=5:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G

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
SCRATCH_BASE="/scratch-local/$USER/KernelLTL/datasets"

# Job mode:
# - "build": sample formulas and create/update stage datasets
# - "add_satisfactions": compute satisfactions for existing stage datasets only
JOB_MODE="add_satisfactions"
# In add_satisfactions mode, control which datasets to update:
# Set to "1" to add satisfactions, "0" to skip
ADD_TRAIN_SATISFACTIONS=1
ADD_EVAL_SATISFACTIONS=0

# ============================================================================
# CURRICULUM STAGE DEFINITIONS
# ============================================================================
# Format: "stage_name:k_samples:max_depth:p_leaf_range:eval_ratio"
# - k_samples: total number of formulas to sample
# - max_depth: maximum formula tree depth
# - p_leaf_range: probability of leaf node (higher = simpler formulas)
# - eval_ratio: fraction for evaluation set (using disjoint split)

STAGES=(
    "stage2:200000:3:3:0.1 0.5:0.025"
    "stage3:400000:4:4:0.01 0.5:0.025"
    "stage4:800000:5:5:0.01 0.4:0.025"
)
#    "stage1:100000:2:2:0.2 0.5:0.025"

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
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

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
if [ "$JOB_MODE" = "add_satisfactions" ]; then
    echo "Adding satisfactions to existing curriculum datasets..."
else
    echo "Generating curriculum datasets..."
fi
echo "=============================================="

# Track previous stage output for incremental satisfaction computation
# These are updated at the end of each stage
# Start with empty for the first stage
PREV_STAGE_NAME="stage1"

for stage_def in "${STAGES[@]}"; do
    # Parse stage definition
    IFS=':' read -r STAGE_NAME K_SAMPLES MAX_DEPTH MIN_DEPTH P_LEAF_RANGE EVAL_RATIO <<< "$stage_def"
    
    TRAIN_OUT="$DATASETS_BASE/$STAGE_NAME/train"
    EVAL_OUT="$DATASETS_BASE/$STAGE_NAME/eval"
    SCRATCH_STAGE_BASE="$SCRATCH_BASE/$STAGE_NAME"
    SCRATCH_TRAIN_OUT="$SCRATCH_STAGE_BASE/train"
    SCRATCH_EVAL_OUT="$SCRATCH_STAGE_BASE/eval"
    if [ -n "$PREV_STAGE_NAME" ]; then
        PREV_TRAIN_OUT="$DATASETS_BASE/$PREV_STAGE_NAME/train"
        PREV_EVAL_OUT="$DATASETS_BASE/$PREV_STAGE_NAME/eval"
        SCRATCH_PREV_STAGE_BASE="$SCRATCH_BASE/$PREV_STAGE_NAME"
        SCRATCH_PREV_TRAIN_OUT="$SCRATCH_PREV_STAGE_BASE/train"
        SCRATCH_PREV_EVAL_OUT="$SCRATCH_PREV_STAGE_BASE/eval"
    fi

    echo ""
    echo "=============================================="
    if [ "$JOB_MODE" = "add_satisfactions" ]; then
        echo "Adding satisfactions for $STAGE_NAME"
    else
        echo "Generating dataset for $STAGE_NAME"
    fi
    echo "  - Total samples: $K_SAMPLES"
    echo "  - Max depth: $MAX_DEPTH"
    echo "  - Min depth: $MAX_DEPTH" 
    echo "  - P_leaf range: $P_LEAF_RANGE"
    echo "  - Eval ratio: $EVAL_RATIO"
    echo "  - Train output: $TRAIN_OUT"
    echo "  - Eval output: $EVAL_OUT"
    if [ -n "$PREV_STAGE_NAME" ]; then
        echo "  - Train base: $PREV_TRAIN_OUT"
        echo "  - Eval base: $PREV_EVAL_OUT"
    fi
    echo "=============================================="
    
    if [ "$JOB_MODE" = "add_satisfactions" ]; then
        # --add-satisfactions requires --kernel-dir and at least one of --train-out/--eval-out.
        # User can now control which datasets to update.
        if [ "$ADD_TRAIN_SATISFACTIONS" = "1" ] && [ ! -d "$TRAIN_OUT" ]; then
            echo "Error: Expected existing train dataset at $TRAIN_OUT for add_satisfactions mode."
            exit 1
        fi
        if [ "$ADD_EVAL_SATISFACTIONS" = "1" ] && [ ! -d "$EVAL_OUT" ]; then
            echo "Error: Expected existing eval dataset at $EVAL_OUT for add_satisfactions mode."
            exit 1
        fi

        # Prepare scratch-local directories
        mkdir -p "$SCRATCH_STAGE_BASE"
        if [ "$ADD_TRAIN_SATISFACTIONS" = "1" ]; then
            echo "Copying train dataset to scratch-local: $TRAIN_OUT -> $SCRATCH_TRAIN_OUT"
            mkdir -p "$SCRATCH_TRAIN_OUT"
            rsync -a --delete "$TRAIN_OUT/" "$SCRATCH_TRAIN_OUT/"
        fi
        if [ "$ADD_EVAL_SATISFACTIONS" = "1" ]; then
            echo "Copying eval dataset to scratch-local: $EVAL_OUT -> $SCRATCH_EVAL_OUT"
            mkdir -p "$SCRATCH_EVAL_OUT"
            rsync -a --delete "$EVAL_OUT/" "$SCRATCH_EVAL_OUT/"
        fi

        if [ -n "$PREV_STAGE_NAME" ]; then
            if [ "$ADD_TRAIN_SATISFACTIONS" = "1" ] && [ -d "$PREV_TRAIN_OUT" ] && [! -d "$SCRATCH_PREV_TRAIN_OUT" ]; then
                echo "Copying previous train dataset to scratch-local: $PREV_TRAIN_OUT -> $SCRATCH_PREV_TRAIN_OUT"
                mkdir -p "$SCRATCH_PREV_TRAIN_OUT"
                rsync -a --delete "$PREV_TRAIN_OUT/" "$SCRATCH_PREV_TRAIN_OUT/"
            fi
            if [ "$ADD_EVAL_SATISFACTIONS" = "1" ] && [ -d "$PREV_EVAL_OUT" ] && [! -d "$SCRATCH_PREV_EVAL_OUT" ]; then
                echo "Copying previous eval dataset to scratch-local: $PREV_EVAL_OUT -> $SCRATCH_PREV_EVAL_OUT"
                mkdir -p "$SCRATCH_PREV_EVAL_OUT"
                rsync -a --delete "$PREV_EVAL_OUT/" "$SCRATCH_PREV_EVAL_OUT/"
            fi
        fi

        # Always use torchrun with 4 processes (GPUs)

        TORCHRUN_ARGS=(--nproc_per_node=4)
        CMD=(torchrun "${TORCHRUN_ARGS[@]}" scripts/prepare_datasets.py --kernel-dir "$KERNEL_DIR" --add-satisfactions)
        if [ "$ADD_TRAIN_SATISFACTIONS" = "1" ]; then
            CMD+=(--train-out "$SCRATCH_TRAIN_OUT" --train-satisfaction-batch-size "$TRAIN_SATISFACTION_BATCH_SIZE" --train-satisfaction-time-index "$TRAIN_SATISFACTION_TIME_INDEX")
            if [ -d "$SCRATCH_PREV_TRAIN_OUT" ]; then
                CMD+=(--base-train-dir "$SCRATCH_PREV_TRAIN_OUT")
            fi
        fi
        if [ "$ADD_EVAL_SATISFACTIONS" = "1" ]; then
            CMD+=(--eval-out "$SCRATCH_EVAL_OUT" --eval-satisfaction-batch-size "$EVAL_SATISFACTION_BATCH_SIZE" --eval-satisfaction-time-index "$EVAL_SATISFACTION_TIME_INDEX")
            if [ -d "$SCRATCH_PREV_EVAL_OUT" ]; then
                CMD+=(--base-eval-dir "$SCRATCH_PREV_EVAL_OUT")
            fi
        fi
        if [ "$ADD_TRAIN_SATISFACTIONS" != "1" ] && [ "$ADD_EVAL_SATISFACTIONS" != "1" ]; then
            echo "Warning: Both ADD_TRAIN_SATISFACTIONS and ADD_EVAL_SATISFACTIONS are 0; nothing to do."
            continue
        fi

        echo "Running: ${CMD[*]}"
        "${CMD[@]}"

        PREV_STAGE_NAME="$STAGE_NAME"

        # Copy results back to main storage
        if [ "$ADD_TRAIN_SATISFACTIONS" = "1" ]; then
            echo "Copying updated train dataset back to main storage: $SCRATCH_TRAIN_OUT -> $TRAIN_OUT"
            rsync -a --delete "$SCRATCH_TRAIN_OUT/" "$TRAIN_OUT/"
        fi
        if [ "$ADD_EVAL_SATISFACTIONS" = "1" ]; then
            echo "Copying updated eval dataset back to main storage: $SCRATCH_EVAL_OUT -> $EVAL_OUT"
            rsync -a --delete "$SCRATCH_EVAL_OUT/" "$EVAL_OUT/"
        fi
        
        echo "$STAGE_NAME dataset satisfactions update complete!"
        continue
    else
        # Create output directories
        mkdir -p "$TRAIN_OUT"
        mkdir -p "$EVAL_OUT"

        # Build command
        CMD=(
            python -u scripts/prepare_datasets.py
            --kernel-dir "$KERNEL_DIR"
            --base-train-dir "$PREV_TRAIN_OUT"
            --base-eval-dir "$PREV_EVAL_OUT"
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
        
        PREV_STAGE_NAME="$STAGE_NAME"

        echo "$STAGE_NAME dataset generation complete!"
    fi
done

# Clean up scratch-local 
rm -rf "$SCRATCH_BASE"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
if [ "$JOB_MODE" = "add_satisfactions" ]; then
    echo "Satisfactions added successfully for all configured stages!"
else
    echo "All datasets generated successfully!"
fi
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