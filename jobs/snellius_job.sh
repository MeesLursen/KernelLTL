#!/bin/bash
#SBATCH --job-name=kernelltl
#SBATCH --output=logs/kernelltl_%j.out
#SBATCH --error=logs/kernelltl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G

# ============================================================================
# Snellius Job Script for KernelLTL Curriculum Training
# ============================================================================
#
# Usage:
#   1. Adjust SBATCH parameters above as needed (time, gpus, memory)
#   2. Set the paths in the "USER CONFIGURATION" section below
#   3. Submit with: sbatch snellius_job.sh
#
# For multi-GPU training, change:
#   #SBATCH --gpus=4
#   and the script will automatically use torchrun with all available GPUs
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION - Adjust these paths for your setup
# ============================================================================

# Project directory (where this script and the code reside)
PROJECT_DIR="$HOME/KernelLTL"

# Virtual environment directory
VENV_DIR="$PROJECT_DIR/venv"

# Paths to pre-generated artifacts (adjust these to your data locations)
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"
TRAIN_DATASET_DIR="$PROJECT_DIR/datasets/train"
EVAL_DATASET_DIR="$PROJECT_DIR/datasets/eval"

# Optional: Path to load a previous model checkpoint (for curriculum stages > 1)
# Set to empty string "" if starting from scratch
MODEL_LOAD_DIR=""

# Optional: Path to load training arguments from a previous stage
TRAINING_ARGS_LOAD_DIR=""

# Output directories
OUTPUT_DIR="$PROJECT_DIR/runs/stage1"
MODEL_SAVE_DIR="$OUTPUT_DIR/final_model"

# Training hyperparameters
NUM_TRAIN_EPOCHS=50
LEARNING_RATE=3e-4
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_STEPS=500
LOGGING_STEPS=100
EVAL_STEPS=1000
SAVE_STEPS=1000

# Mixed precision (use bf16 for A100 GPUs on Snellius, fp16 for older GPUs)
MIXED_PRECISION="--bf16"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Load required modules (adjust based on Snellius module availability)
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
echo "Python path: $(which python)"

# ============================================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_SAVE_DIR"

# ============================================================================
# BUILD TRAINING COMMAND
# ============================================================================

# Determine number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs detected: $NUM_GPUS"

# Build the base command arguments
CMD_ARGS=(
    "--kernel-dir" "$KERNEL_DIR"
    "--tokenizer-dir" "$TOKENIZER_DIR"
    "--train-dataset-dir" "$TRAIN_DATASET_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--model-save-dir" "$MODEL_SAVE_DIR"
    "--num-train-epochs" "$NUM_TRAIN_EPOCHS"
    "--learning-rate" "$LEARNING_RATE"
    "--per-device-train-batch-size" "$PER_DEVICE_TRAIN_BATCH_SIZE"
    "--per-device-eval-batch-size" "$PER_DEVICE_EVAL_BATCH_SIZE"
    "--gradient-accumulation-steps" "$GRADIENT_ACCUMULATION_STEPS"
    "--warmup-steps" "$WARMUP_STEPS"
    "--logging-steps" "$LOGGING_STEPS"
    "--eval-steps" "$EVAL_STEPS"
    "--save-steps" "$SAVE_STEPS"
    "--dataloader-num-workers" "$SLURM_CPUS_PER_TASK"
    "--dataloader-pin-memory"
    "--report-to" "tensorboard"
    $MIXED_PRECISION
)

# Add optional evaluation dataset
if [ -n "$EVAL_DATASET_DIR" ] && [ -d "$EVAL_DATASET_DIR" ]; then
    CMD_ARGS+=("--eval-dataset-dir" "$EVAL_DATASET_DIR")
fi

# Add optional model checkpoint (for curriculum stages > 1)
if [ -n "$MODEL_LOAD_DIR" ] && [ -d "$MODEL_LOAD_DIR" ]; then
    CMD_ARGS+=("--model-load-dir" "$MODEL_LOAD_DIR")
fi

# Add optional training args from previous stage
if [ -n "$TRAINING_ARGS_LOAD_DIR" ] && [ -d "$TRAINING_ARGS_LOAD_DIR" ]; then
    CMD_ARGS+=("--training-args-load-dir" "$TRAINING_ARGS_LOAD_DIR")
fi

# Set PYTHONPATH to include project directory
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "=============================================="
echo "Starting training..."
echo "=============================================="

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using distributed training with $NUM_GPUS GPUs via torchrun"
    torchrun --nproc_per_node="$NUM_GPUS" \
        scripts/curriculum_train.py \
        "${CMD_ARGS[@]}"
else
    echo "Using single GPU training"
    python scripts/curriculum_train.py \
        "${CMD_ARGS[@]}"
fi

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

echo "=============================================="
echo "Training completed!"
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Final model saved to: $MODEL_SAVE_DIR"
echo "=============================================="
