#!/bin/bash
#SBATCH --job-name=kernelltl_validate
#SBATCH --output=logs/kernelltl_validate_%j.out
#SBATCH --error=logs/kernelltl_validate_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G

# ==========================================================================
# Snellius validation script for KernelLTL
#
# Iterates over a list of trained models, runs greedy + top-K validation
# on the held-out validation dataset, and rsyncs per-sample JSONLs and
# summary JSONs back to project storage.
# ==========================================================================

set -e

# ==========================================================================
# USER CONFIGURATION
# ==========================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Shared artifacts
KERNEL_DIR="$HOME_DIR/artifacts/kernel"
TOKENIZER_DIR="$HOME_DIR/artifacts/tokenizer"

# Validation dataset (5k formulas at depths {2,3,4,5})
VALIDATION_DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"

# CE base / reference model used for KL computation
CE_REFERENCE_MODEL_DIR="$PROJECT_DIR/artifacts/models/CE/run2/stage4/final_model"

# Output roots
PROJECT_OUTPUT_BASE="$PROJECT_DIR/artifacts/validation"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_BASE="$SCRATCH_BASE/validation"

# Validation defaults
PER_DEVICE_EVAL_BATCH_SIZE=96
SEMANTIC_EVAL_BATCH_SIZE="256000"
TOP_K=5
MIXED_PRECISION="--bf16"

# ==========================================================================
# VALIDATION CONFIGURATION
# Format: "RUN_NAME:MODEL_DIR[:FLAGS]"
#
# FLAGS (third field, optional):
#   no_kl   -- skip KL-from-base computation (use for the base model itself,
#              where comparing it against itself is uninformative).
#
# When FLAGS is omitted, KL is computed against $CE_REFERENCE_MODEL_DIR.
# ==========================================================================

VALIDATE_CONFIGS=(
    "ce_base:$PROJECT_DIR/artifacts/models/CE/run2/stage4/final_model:no_kl"
    "ce_finetune:$PROJECT_DIR/artifacts/models/CE/finetune/final_model"
    "rb_momentum_09_lr_5e-8:$PROJECT_DIR/artifacts/models/RE/rb_momentum_09_lr_5e-8/final_model"
    "gae_lambda_09_lr_5e-8_crlr_1e-3:$PROJECT_DIR/artifacts/models/RE/gae_lambda_09_lr_5e-8_crlr_1e-3/final_model"
    "gae_lambda_1_lr_5e-8_crlr_5e-3:$PROJECT_DIR/artifacts/models/RE/gae_lambda_1_lr_5e-8_crlr_5e-3/final_model"
)

# ==========================================================================
# ENVIRONMENT SETUP
# ==========================================================================

echo "=============================================="
echo "KernelLTL Validation"
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

export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# Stage shared artifacts (kernel, tokenizer, validation dataset, CE reference)
# onto scratch-local once. Per-model directories are staged inside the loop.
SCRATCH_KERNEL_DIR="$SCRATCH_BASE/kernel"
SCRATCH_TOKENIZER_DIR="$SCRATCH_BASE/tokenizer"
SCRATCH_VALIDATION_DIR="$SCRATCH_BASE/datasets/validation"
SCRATCH_CE_REF_DIR="$SCRATCH_BASE/models/ce_reference"

mkdir -p "$SCRATCH_KERNEL_DIR" "$SCRATCH_TOKENIZER_DIR" "$SCRATCH_VALIDATION_DIR" "$SCRATCH_CE_REF_DIR"

echo "Staging shared artifacts to scratch-local..."
rsync -a --delete "$KERNEL_DIR/" "$SCRATCH_KERNEL_DIR/"
rsync -a --delete "$TOKENIZER_DIR/" "$SCRATCH_TOKENIZER_DIR/"
rsync -a --delete "$VALIDATION_DATASET_DIR/" "$SCRATCH_VALIDATION_DIR/"
rsync -a --delete "$CE_REFERENCE_MODEL_DIR/" "$SCRATCH_CE_REF_DIR/"

# ==========================================================================
# RUN VALIDATIONS
# ==========================================================================

for i in "${!VALIDATE_CONFIGS[@]}"; do
    IFS=':' read -r RUN_NAME MODEL_DIR FLAGS <<< "${VALIDATE_CONFIGS[$i]}"

    SKIP_KL=0
    if [ "$FLAGS" = "no_kl" ]; then
        SKIP_KL=1
    fi

    SCRATCH_MODEL_DIR="$SCRATCH_BASE/models/$RUN_NAME/final_model"
    SCRATCH_OUTPUT_DIR="$SCRATCH_OUTPUT_BASE/$RUN_NAME"
    PROJECT_OUTPUT_DIR="$PROJECT_OUTPUT_BASE/$RUN_NAME"

    echo ""
    echo "=============================================="
    echo "Validating: $RUN_NAME ($((i+1)) of ${#VALIDATE_CONFIGS[@]})"
    echo "  Model dir:   $MODEL_DIR"
    echo "  KL from base: $([ "$SKIP_KL" = "1" ] && echo "disabled" || echo "enabled (vs $CE_REFERENCE_MODEL_DIR)")"
    echo "  Output (scratch): $SCRATCH_OUTPUT_DIR"
    echo "  Output (project): $PROJECT_OUTPUT_DIR"
    echo "=============================================="

    mkdir -p "$SCRATCH_MODEL_DIR" "$SCRATCH_OUTPUT_DIR" "$PROJECT_OUTPUT_DIR"

    echo "Syncing model to scratch..."
    rsync -a --delete "$MODEL_DIR/" "$SCRATCH_MODEL_DIR/"

    CMD_ARGS=(
        "--kernel-dir" "$SCRATCH_KERNEL_DIR"
        "--tokenizer-dir" "$SCRATCH_TOKENIZER_DIR"
        "--eval-dataset-dir" "$SCRATCH_VALIDATION_DIR"
        "--model-load-dir" "$SCRATCH_MODEL_DIR"
        "--output-dir" "$SCRATCH_OUTPUT_DIR"
        "--per-device-eval-batch-size" "$PER_DEVICE_EVAL_BATCH_SIZE"
        "--semantic-eval-batch-size" "$SEMANTIC_EVAL_BATCH_SIZE"
        "--top-k" "$TOP_K"
        $MIXED_PRECISION
    )

    if [ "$SKIP_KL" != "1" ]; then
        CMD_ARGS+=("--ce-reference-model-dir" "$SCRATCH_CE_REF_DIR")
    fi

    STAGE_START=$(date +%s)

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
            scripts/validate_model.py \
            "${CMD_ARGS[@]}"
    else
        python scripts/validate_model.py \
            "${CMD_ARGS[@]}"
    fi

    STAGE_END=$(date +%s)
    STAGE_DURATION=$((STAGE_END - STAGE_START))

    echo "Syncing validation outputs to project storage..."
    rsync -a --delete "$SCRATCH_OUTPUT_DIR/" "$PROJECT_OUTPUT_DIR/"

    echo "$RUN_NAME completed in $((STAGE_DURATION / 3600))h $(((STAGE_DURATION % 3600) / 60))m $((STAGE_DURATION % 60))s"
done

echo ""
echo "=============================================="
echo "Cleaning scratch-local"
rm -rf "$SCRATCH_BASE"

echo "All validations completed!"
echo "End time: $(date)"
echo "=============================================="
