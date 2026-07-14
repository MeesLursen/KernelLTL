#!/bin/bash
#SBATCH --job-name=kernelltl-lr-sweep
#SBATCH --output=logs/kernelltl_test_job_%j.out
#SBATCH --error=logs/kernelltl_test_job_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G

# ============================================================================
# Snellius Learning Rate Sweep Script for KernelLTL (Stage 1 CE Training)
# ============================================================================
#
# Runs a sequential sweep over LR_VALUES, each run starting from the same
# base model checkpoint. The best model across all runs (lowest
# eval_semantic_distance) is identified at the end and copied to
# HOME_OUTPUT_DIR/sweep/best_model/.
#
# Early stopping is used to avoid wasting compute on runs that plateau early.
# The metric_for_best_model is eval_semantic_distance (lower is better), so
# each run's saved final_model is already its best checkpoint.
#
# Usage:
#   sbatch snellius_test_job.sh
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"

HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Shared artifacts
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"

# Output directories
HOME_OUTPUT_DIR="$HOME_DIR/artifacts/models/CE/test"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_BASE="$SCRATCH_BASE/models/CE/test"

# Dataset for stage0 (same for all runs)
TRAIN_DIR="$PROJECT_DIR/artifacts/datasets/stage1/train"
EVAL_DIR="$PROJECT_DIR/artifacts/datasets/stage1/eval"

LR="1e-4"

# Max epochs per run. Early stopping will cut this short in practice.
# Keep this high enough that well-behaved runs have time to converge.
EPOCHS=100

# Early stopping patience (in eval steps, not epochs).
# Given the oscillatory semantic_distance behaviour observed in stage1 run3,
# a patience of 15 is enough to outlast ~3 full oscillation cycles without
# stopping prematurely on a spike. Lower this if compute budget is tight.
EARLY_STOPPING_PATIENCE=10

# Minimum improvement in eval_semantic_distance required to reset patience
# counter. 0.0 means any strict improvement counts (recommended here since
# improvements are small near the plateau).
EARLY_STOPPING_THRESHOLD=0.0

# Fixed training settings (shared across all runs for a fair comparison)
DEFAULT_BATCH_SIZE=256
DEFAULT_WARMUP_RATIO=0.05
MIXED_PRECISION="--bf16"
EVAL_BATCH_SIZE="81920"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL Test Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Max epochs per run: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
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
# export PYTHONFAULTHANDLER=1
# export PYTHONUNBUFFERED=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# Use the LR string directly as a tag (hyphens are safe in Linux paths)
LR_TAG="test_metrics_debug${LR}"

STAGE_OUTPUT_DIR="$SCRATCH_OUTPUT_BASE/$LR_TAG"
STAGE_MODEL_SAVE_DIR="$STAGE_OUTPUT_DIR/final_model"
STAGE_HOME_OUTPUT_DIR="$HOME_OUTPUT_DIR/$LR_TAG"
STAGE_HOME_MODEL_SAVE_DIR="$STAGE_HOME_OUTPUT_DIR/final_model"

# eval_steps / save_steps as a fraction of 1 epoch (1 eval per epoch)
STEP_INTERVAL=$(echo "scale=6; 1/$EPOCHS" | bc -l)

mkdir -p "$STAGE_OUTPUT_DIR" "$STAGE_MODEL_SAVE_DIR"
mkdir -p "$STAGE_HOME_OUTPUT_DIR" "$STAGE_HOME_MODEL_SAVE_DIR"

CMD_ARGS=(
    "--kernel-dir"                   "$KERNEL_DIR"
    "--tokenizer-dir"                "$TOKENIZER_DIR"
    "--train-dataset-dir"            "$TRAIN_DIR"
    "--eval-dataset-dir"             "$EVAL_DIR"
    "--output-dir"                   "$STAGE_OUTPUT_DIR"
    "--model-save-dir"               "$STAGE_MODEL_SAVE_DIR"
    "--num-train-epochs"             "$EPOCHS"
    "--learning-rate"                "$LR"
    "--per-device-train-batch-size"  "$DEFAULT_BATCH_SIZE"
    "--per-device-eval-batch-size"   "$DEFAULT_BATCH_SIZE"
    "--warmup-ratio"                 "$DEFAULT_WARMUP_RATIO"
    "--logging-steps"                "$STEP_INTERVAL"
    "--eval-steps"                   "$STEP_INTERVAL"
    "--save-steps"                   "$STEP_INTERVAL"
    "--dataloader-num-workers"       "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
    "--dataloader-pin-memory"
    "--metric-for-best-model"        "eval_semantic_distance"
    "--greater-is-better"            "false"
    "--early-stopping-patience"      "$EARLY_STOPPING_PATIENCE"
    "--early-stopping-threshold"     "$EARLY_STOPPING_THRESHOLD"
    "--semantic-eval-batch-size"     "$EVAL_BATCH_SIZE"
    "--callback-debug"
    "--stage-name"                   "$LR_TAG"   # tags all JSONL records for this run
    $MIXED_PRECISION
)

RUN_START=$(date +%s)

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
        scripts/curriculum_train.py \
        "${CMD_ARGS[@]}"
else
    python scripts/curriculum_train.py \
        "${CMD_ARGS[@]}"
fi

RUN_END=$(date +%s)
RUN_DURATION=$((RUN_END - RUN_START))
echo "LR=$LR completed in $((RUN_DURATION / 3600))h $(((RUN_DURATION % 3600) / 60))m $((RUN_DURATION % 60))s"

# Persist logs and best model to home storage before moving to next run.
echo "Syncing logs and model back to home..."
mkdir -p "$STAGE_HOME_OUTPUT_DIR/logs"
if [ -d "$STAGE_OUTPUT_DIR/logs" ]; then
    rsync -a --delete "$STAGE_OUTPUT_DIR/logs/" "$STAGE_HOME_OUTPUT_DIR/logs/"
fi

echo ""
echo "Cleaning scratch-local..."
rm -rf "$SCRATCH_BASE"

echo ""
echo "=============================================="
echo "Test job completed!"
echo "End time: $(date)"
echo "=============================================="
