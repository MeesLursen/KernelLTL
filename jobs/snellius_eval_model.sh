#!/bin/bash
#SBATCH --job-name=kernelltl_eval
#SBATCH --output=logs/kernelltl_eval_%j.out
#SBATCH --error=logs/kernelltl_eval_%j.err
#SBATCH --time=00:59:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G

# ==========================================================================
# Snellius model evaluation script for KernelLTL
# ==========================================================================

set -e

# ==========================================================================
# USER CONFIGURATION
# ==========================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Shared artifacts
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"

# Output roots
PROJECT_OUTPUT_BASE="$PROJECT_DIR/artifacts/eval"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_BASE="$SCRATCH_BASE/eval"

# Evaluation defaults
CE_REFERENCE_MODEL_DIR="$PROJECT_DIR/artifacts/models/CE/run2/stage4/final_model"
PER_DEVICE_EVAL_BATCH_SIZE=96
SEMANTIC_EVAL_BATCH_SIZE="256000"
MIXED_PRECISION="--bf16"

# ===========================================================================
# EVAL CONFIGURATION
# Format: "RUN_NAME:MODEL_DIR:EVAL_DIR:TRAINER_KIND"
# ===========================================================================

EVAL_CONFIGS=(
    "finetune_rb_eval:$PROJECT_DIR/artifacts/models/RE/rb_momentum_09_lr_5e-8/final_model:$PROJECT_DIR/artifacts/datasets/stage4/eval:rb"
)

# ==========================================================================
# ENVIRONMENT SETUP
# ==========================================================================

echo "=============================================="
echo "KernelLTL Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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

export PYTHONPATH="$HOME_DIR:$PYTHONPATH"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# ==========================================================================
# RUN EVALS
# ==========================================================================

for i in "${!EVAL_CONFIGS[@]}"; do
    IFS=':' read -r RUN_NAME MODEL_DIR EVAL_DIR TRAINER_KIND <<< "${EVAL_CONFIGS[$i]}"

    SCRATCH_MODEL_DIR="$SCRATCH_BASE/models/$RUN_NAME/final_model"
    SCRATCH_EVAL_DIR="$SCRATCH_BASE/datasets/$RUN_NAME/eval"
    SCRATCH_OUTPUT_DIR="$SCRATCH_OUTPUT_BASE/$RUN_NAME"

    PROJECT_OUTPUT_DIR="$PROJECT_OUTPUT_BASE/$RUN_NAME"

    echo ""
    echo "=============================================="
    echo "Starting evaluation: $RUN_NAME"
    echo "  Model dir: $MODEL_DIR"
    echo "  Eval dataset: $EVAL_DIR"
    echo "  Output (scratch): $SCRATCH_OUTPUT_DIR"
    echo "  Output (project): $PROJECT_OUTPUT_DIR"
    echo "=============================================="

    mkdir -p "$SCRATCH_MODEL_DIR"
    mkdir -p "$SCRATCH_EVAL_DIR"
    mkdir -p "$SCRATCH_OUTPUT_DIR"
    mkdir -p "$PROJECT_OUTPUT_DIR"

    echo "Syncing model to scratch..."
    rsync -a --delete "$MODEL_DIR/" "$SCRATCH_MODEL_DIR/"

    echo "Copying eval dataset to scratch..."
    cp -r "$EVAL_DIR/." "$SCRATCH_EVAL_DIR/"

    CMD_ARGS=(
        "--kernel-dir" "$KERNEL_DIR"
        "--tokenizer-dir" "$TOKENIZER_DIR"
        "--eval-dataset-dir" "$SCRATCH_EVAL_DIR"
        "--model-load-dir" "$SCRATCH_MODEL_DIR"
        "--ce-reference-model-dir" "$CE_REFERENCE_MODEL_DIR"
        "--output-dir" "$SCRATCH_OUTPUT_DIR"
        "--trainer-kind" "$TRAINER_KIND"
        "--per-device-eval-batch-size" "$PER_DEVICE_EVAL_BATCH_SIZE"
        "--semantic-eval-batch-size" "$SEMANTIC_EVAL_BATCH_SIZE"
        "--stage-name" "$RUN_NAME"
        $MIXED_PRECISION
    )

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
            scripts/evaluate_model.py \
            "${CMD_ARGS[@]}"
    else
        python scripts/evaluate_model.py \
            "${CMD_ARGS[@]}"
    fi

    echo "Syncing evaluation outputs to project storage..."
    rsync -a --delete "$SCRATCH_OUTPUT_DIR/" "$PROJECT_OUTPUT_DIR/"

done

echo "Cleaning scratch-local"
rm -rf "$SCRATCH_BASE"

echo ""
echo "=============================================="
echo "All evaluations completed!"
echo "End time: $(date)"
