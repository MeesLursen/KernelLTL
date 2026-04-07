#!/bin/bash
#SBATCH --job-name=kernelltl-curriculum
#SBATCH --output=logs/kernelltl_curriculum_%j.out
#SBATCH --error=logs/kernelltl_curriculum_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G

# ============================================================================
# Snellius Multi-Stage Curriculum Training Script for KernelLTL
# ============================================================================
#
# This script runs multiple curriculum stages sequentially, automatically
# loading the model checkpoint from the previous stage.
#
# Usage:
#   1. Configure the STAGE_CONFIGS array below with your stage parameters
#   2. Submit with: sbatch snellius_curriculum.sh
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION
# ============================================================================

HOME_DIR="$HOME/KernelLTL"
PROJECT_DIR="/projects/prjs2029/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Shared artifacts
KERNEL_DIR="$HOME_DIR/artifacts/kernel"
TOKENIZER_DIR="$HOME_DIR/artifacts/tokenizer"

# Home output directory (for persisted copies)
PROJECT_OUTPUT_DIR="$PROJECT_DIR/artifacts/models/CE"

# Scratch (fast) storage
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_BASE="$SCRATCH_BASE/models/CE"

# Training defaults (can be overridden per stage)
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_BATCH_SIZE=256
DEFAULT_WARMUP_RATIO=0.05

# Mixed precision
MIXED_PRECISION="--bf16"

# Evaluation Batch Size
EVAL_BATCH_SIZE="81920"

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_THRESHOLD=0.0

# ============================================================================
# STAGE CONFIGURATION
# ============================================================================
# Define your curriculum stages here
# Format: "STAGE_NAME|TRAIN_DIR|EVAL_DIR|EPOCHS|LEARNING_RATE"

# ============================================================================

STAGE_CONFIGS=(
    "stage0:$PROJECT_DIR/artifacts/datasets/stage0/train:$PROJECT_DIR/artifacts/datasets/stage0/eval:10:1e-4"
)   
    # "stage1:$PROJECT_DIR/artifacts/datasets/stage1/train:$PROJECT_DIR/artifacts/datasets/stage1/eval:100:1e-4"
    # "stage2:$PROJECT_DIR/artifacts/datasets/stage2/train:$PROJECT_DIR/artifacts/datasets/stage2/eval:100:5e-5"
    # "stage3:$PROJECT_DIR/artifacts/datasets/stage3/train:$PROJECT_DIR/artifacts/datasets/stage3/eval:100:1e-5"
    # "stage4:$PROJECT_DIR/artifacts/datasets/stage4/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:100:5e-6" 


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL Multi-Stage Curriculum Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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

export PYTHONPATH="$HOME_DIR:$PYTHONPATH"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# ============================================================================
# RUN CURRICULUM STAGES
# ============================================================================
PREV_MODEL_PROJECT_DIR=""
PREV_MODEL_DIR=""
PREV_TRAINING_ARGS_DIR=""

if [ -n "$PREV_MODEL_PROJECT_DIR" ] && [ -d "$PREV_MODEL_PROJECT_DIR" ]; then
    # Copy previous model dir from home to scratch-local
    echo "Copying previous model from $PREV_MODEL_PROJECT_DIR to $PREV_MODEL_DIR..."
    mkdir -p "$PREV_MODEL_DIR"
    if [ -d "$PREV_MODEL_DIR" ]; then
        rsync -a --delete "$PREV_MODEL_PROJECT_DIR/" "$PREV_MODEL_DIR/"
    fi
fi

DEBUG_OPTION="underflow_overflow"

for i in "${!STAGE_CONFIGS[@]}"; do
    # Parse stage configuration
    IFS=':' read -r STAGE_NAME TRAIN_DIR EVAL_DIR EPOCHS LR BATCH_SIZE <<< "${STAGE_CONFIGS[$i]}"
    
    # Use defaults if not specified
    LR=${LR:-$DEFAULT_LEARNING_RATE}
    BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
    STEP_INTERVAL=$(echo "scale=6; 1/$EPOCHS" | bc -l)
    
    STAGE_OUTPUT_DIR="$SCRATCH_OUTPUT_BASE/$STAGE_NAME"
    STAGE_MODEL_SAVE_DIR="$STAGE_OUTPUT_DIR/final_model"

    STAGE_PROJECT_OUTPUT_DIR="$PROJECT_OUTPUT_DIR/$STAGE_NAME"
    STAGE_PROJECT_MODEL_SAVE_DIR="$STAGE_PROJECT_OUTPUT_DIR/final_model"
    
    echo ""
    echo "=============================================="
    echo "Starting $STAGE_NAME (Run $((i+1)) of ${#STAGE_CONFIGS[@]})"
    echo "  Train dataset: $TRAIN_DIR"
    echo "  Eval dataset: $EVAL_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Learning rate: $LR"
    echo "  Batch size: $BATCH_SIZE"
    echo "=============================================="
    
    mkdir -p "$STAGE_OUTPUT_DIR"
    mkdir -p "$STAGE_MODEL_SAVE_DIR"
    mkdir -p "$STAGE_PROJECT_OUTPUT_DIR"

    SCRATCH_TRAIN_DIR="/scratch-local/$USER/KernelLTL/datasets/$STAGE_NAME/train"
    SCRATCH_EVAL_DIR="/scratch-local/$USER/KernelLTL/datasets/$STAGE_NAME/eval"

    echo ""
    echo "=============================================="
    echo "Copying train+eval datasets from home to scratch..."
    echo "  From: $TRAIN_DIR" and "$EVAL_DIR"
    echo "  To:   $SCRATCH_TRAIN_DIR" and "$SCRATCH_EVAL_DIR"
    echo "=============================================="

    mkdir -p $SCRATCH_TRAIN_DIR
    mkdir -p $SCRATCH_EVAL_DIR

    cp -r "$TRAIN_DIR/." "$SCRATCH_TRAIN_DIR/"
    cp -r "$EVAL_DIR/." "$SCRATCH_EVAL_DIR/"

    # Build command arguments
    CMD_ARGS=(
        "--kernel-dir" "$KERNEL_DIR"
        "--tokenizer-dir" "$TOKENIZER_DIR"
        "--train-dataset-dir" "$SCRATCH_TRAIN_DIR"
        "--eval-dataset-dir" "$SCRATCH_EVAL_DIR"
        "--output-dir" "$STAGE_OUTPUT_DIR"
        "--model-save-dir" "$STAGE_MODEL_SAVE_DIR"
        "--num-train-epochs" "$EPOCHS"
        "--learning-rate" "$LR"
        "--per-device-train-batch-size" "$DEFAULT_BATCH_SIZE"
        "--per-device-eval-batch-size" "$DEFAULT_BATCH_SIZE"
        "--warmup-ratio" "$DEFAULT_WARMUP_RATIO"
        "--logging-steps" "$STEP_INTERVAL"
        "--eval-steps" "$STEP_INTERVAL"
        "--save-steps" "$STEP_INTERVAL"
        "--dataloader-num-workers" "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
        "--dataloader-pin-memory"
        $MIXED_PRECISION
        "--semantic-eval-batch-size" "$EVAL_BATCH_SIZE"
        "--metric-for-best-model"        "eval_semantic_distance"
        "--greater-is-better"            "false"
        "--early-stopping-patience" "$EARLY_STOPPING_PATIENCE"
        "--early-stopping-threshold" "$EARLY_STOPPING_THRESHOLD"
    )
    
    # Set debugging options
    if [ -n "$DEBUG_OPTION" ]; then
        echo "  Running with debug option: $DEBUG_OPTION"
        CMD_ARGS+=("--debug" "$DEBUG_OPTION")
    fi

    # Load previous stage model (if not first stage)
    if [ -n "$PREV_MODEL_DIR" ] && [ -d "$PREV_MODEL_DIR" ]; then
        echo "  Loading model from previous stage: $PREV_MODEL_DIR"
        CMD_ARGS+=("--model-load-dir" "$PREV_MODEL_DIR")
    fi
    
    # Load previous stage training args (if not first stage)
    if [ -n "$PREV_TRAINING_ARGS_DIR" ] && [ -d "$PREV_TRAINING_ARGS_DIR" ]; then
        CMD_ARGS+=("--training-args-load-dir" "$PREV_TRAINING_ARGS_DIR")
    fi
    
    # Run training
    STAGE_START=$(date +%s)
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
            scripts/curriculum_train.py \
            "${CMD_ARGS[@]}"
    else
        python scripts/curriculum_train.py \
            "${CMD_ARGS[@]}"
    fi
    
    STAGE_END=$(date +%s)
    STAGE_DURATION=$((STAGE_END - STAGE_START))
    
    echo "$STAGE_NAME completed in $((STAGE_DURATION / 3600))h $(((STAGE_DURATION % 3600) / 60))m $((STAGE_DURATION % 60))s"
    
    # Copy logs and final model back to home for persistence
    echo "Copying logs and final model to home storage..."
    mkdir -p "$STAGE_PROJECT_OUTPUT_DIR/logs"
    if [ -d "$STAGE_OUTPUT_DIR/logs" ]; then
        rsync -a --delete "$STAGE_OUTPUT_DIR/logs/" "$STAGE_PROJECT_OUTPUT_DIR/logs/"
    fi
    mkdir -p "$STAGE_PROJECT_MODEL_SAVE_DIR"
    if [ -d "$STAGE_MODEL_SAVE_DIR" ]; then
        rsync -a --delete "$STAGE_MODEL_SAVE_DIR/" "$STAGE_PROJECT_MODEL_SAVE_DIR/"
    fi

    # Set paths for next stage (use scratch paths for speed)
    PREV_MODEL_DIR="$STAGE_MODEL_SAVE_DIR"
    PREV_TRAINING_ARGS_DIR="$STAGE_MODEL_SAVE_DIR"
done

echo "Cleaning scratch-local"
rm -rf "$SCRATCH_BASE"

echo ""
echo "=============================================="
echo "All curriculum stages completed!"
echo "End time: $(date)"
echo "Final model: $PREV_MODEL_DIR"
echo "=============================================="
