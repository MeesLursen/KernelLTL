#!/bin/bash
#SBATCH --job-name=kernelltl-curriculum
#SBATCH --output=logs/kernelltl_curriculum_%j.out
#SBATCH --error=logs/kernelltl_curriculum_%j.err
#SBATCH --time=12:00:00
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

PROJECT_DIR="$HOME/KernelLTL"
VENV_DIR="$PROJECT_DIR/venv"

# Shared artifacts
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"

# Base output directory (stages will be saved as models/stage1, models/stage2, etc.)
BASE_OUTPUT_DIR="$PROJECT_DIR/artifacts/models"

# Training defaults (can be overridden per stage)
DEFAULT_LEARNING_RATE=5e-4
DEFAULT_BATCH_SIZE=64
DEFAULT_WARMUP_STEPS=500

# Mixed precision
MIXED_PRECISION="--bf16"

# Evaluation Batch Size
EVAL_BATCH_SIZE="81920"

# ============================================================================
# STAGE CONFIGURATION
# ============================================================================
# Define your curriculum stages here
# Format: "STAGE_NAME|TRAIN_DIR|EVAL_DIR|EPOCHS|LEARNING_RATE|BATCH_SIZE"

# ============================================================================

STAGE_CONFIGS=(
    "stage0:$PROJECT_DIR/artifacts/datasets/stage0/train:$PROJECT_DIR/datasets/stage0/eval:25:5e-4:64"

)
    # "stage1:$PROJECT_DIR/artifacts/datasets/stage1/train:$PROJECT_DIR/datasets/stage1/eval:50:5e-4:64"
    # "stage2:$PROJECT_DIR/artifacts/datasets/stage2/train:$PROJECT_DIR/datasets/stage2/eval:100:5e-4:64"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL Multi-Stage Curriculum Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p "$PROJECT_DIR/logs"

# Load modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$PROJECT_DIR"

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

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# ============================================================================
# RUN CURRICULUM STAGES
# ============================================================================

PREV_MODEL_DIR=""
PREV_TRAINING_ARGS_DIR=""

for i in "${!STAGE_CONFIGS[@]}"; do
    # Parse stage configuration
    IFS=':' read -r STAGE_NAME TRAIN_DIR EVAL_DIR EPOCHS LR BATCH_SIZE <<< "${STAGE_CONFIGS[$i]}"
    
    # Use defaults if not specified
    LR=${LR:-$DEFAULT_LEARNING_RATE}
    BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
    
    STAGE_OUTPUT_DIR="$BASE_OUTPUT_DIR/$STAGE_NAME"
    STAGE_MODEL_SAVE_DIR="$STAGE_OUTPUT_DIR/final_model"
    
    echo ""
    echo "=============================================="
    echo "Starting $STAGE_NAME (Stage $((i)) of ${#STAGE_CONFIGS[@]})"
    echo "  Train dataset: $TRAIN_DIR"
    echo "  Eval dataset: $EVAL_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Learning rate: $LR"
    echo "  Batch size: $BATCH_SIZE"
    echo "=============================================="
    
    mkdir -p "$STAGE_OUTPUT_DIR"
    mkdir -p "$STAGE_MODEL_SAVE_DIR"
    
    # Build command arguments
    CMD_ARGS=(
        "--kernel-dir" "$KERNEL_DIR"
        "--tokenizer-dir" "$TOKENIZER_DIR"
        "--train-dataset-dir" "$TRAIN_DIR"
        "--eval-dataset-dir" "$EVAL_DIR"
        "--output-dir" "$STAGE_OUTPUT_DIR"
        "--model-save-dir" "$STAGE_MODEL_SAVE_DIR"
        "--num-train-epochs" "$EPOCHS"
        "--learning-rate" "$LR"
        "--per-device-train-batch-size" "$BATCH_SIZE"
        "--per-device-eval-batch-size" "$BATCH_SIZE"
        "--warmup-steps" "$DEFAULT_WARMUP_STEPS"
        "--logging-steps" "0.02"
        "--eval-steps" "0.02"
        "--save-steps" "0.2"
        "--dataloader-num-workers" "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
        "--dataloader-pin-memory"
        "--report-to" "all"
        $MIXED_PRECISION
        "--semantic-eval-batch-size" "$EVAL_BATCH_SIZE"
    )
    
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
    
    # Set paths for next stage
    PREV_MODEL_DIR="$STAGE_MODEL_SAVE_DIR"
    PREV_TRAINING_ARGS_DIR="$STAGE_MODEL_SAVE_DIR"
done

echo ""
echo "=============================================="
echo "All curriculum stages completed!"
echo "End time: $(date)"
echo "Final model: $PREV_MODEL_DIR"
echo "=============================================="
