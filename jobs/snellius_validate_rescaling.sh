#!/bin/bash
#SBATCH --job-name=kernelltl_rescaling
#SBATCH --output=logs/kernelltl_rescaling_%j.out
#SBATCH --error=logs/kernelltl_rescaling_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G

# ==========================================================================
# Experiment 3, Test 2: norm-rescaling intervention on the CE base model.
#
# Runs scripts/validation_variance_rescaling.py -- one greedy pass per scale
# c over the validation set, on c * emb(phi) -- and rsyncs the per-scale
# JSONLs and the sweep manifest back to project storage under
# artifacts/validation/<RUN_NAME>_rescaling/scale_<c>/.
#
# Same staging pattern as snellius_validate_models.sh. Scale 1 reproduces
# the conditioned run and scale 0 the zero ablation, so the sweep carries
# both references.
# ==========================================================================

set -e

# ==========================================================================
# USER CONFIGURATION
# ==========================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"
VALIDATION_DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"

PROJECT_OUTPUT_BASE="$PROJECT_DIR/artifacts/validation"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_BASE="$SCRATCH_BASE/validation"

RUN_NAME="ce_base"
MODEL_DIR="$PROJECT_DIR/artifacts/models/CE/final_pretrain/stage4/final_model/"

# The sweep. Log-spaced around 1; 0 and 1 are the references.
SCALES="0 0.25 0.5 0.75 1 1.5 2 4"

PER_DEVICE_EVAL_BATCH_SIZE=96
SEMANTIC_EVAL_BATCH_SIZE="256000"
EMBEDDING_BATCH_SIZE=256
MIXED_PRECISION="--bf16"

# ==========================================================================
# ENVIRONMENT SETUP
# ==========================================================================

echo "=============================================="
echo "KernelLTL Experiment 3 -- norm rescaling sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Scales: $SCALES"
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

SCRATCH_KERNEL_DIR="$SCRATCH_BASE/kernel"
SCRATCH_TOKENIZER_DIR="$SCRATCH_BASE/tokenizer"
SCRATCH_VALIDATION_DIR="$SCRATCH_BASE/datasets/validation"
SCRATCH_MODEL_DIR="$SCRATCH_BASE/models/$RUN_NAME/final_model"
SCRATCH_OUTPUT_DIR="$SCRATCH_OUTPUT_BASE/${RUN_NAME}_rescaling"
PROJECT_OUTPUT_DIR="$PROJECT_OUTPUT_BASE/${RUN_NAME}_rescaling"

mkdir -p "$SCRATCH_KERNEL_DIR" "$SCRATCH_TOKENIZER_DIR" "$SCRATCH_VALIDATION_DIR" \
         "$SCRATCH_MODEL_DIR" "$SCRATCH_OUTPUT_DIR" "$PROJECT_OUTPUT_DIR"

echo "Staging artifacts to scratch-local..."
rsync -a --delete "$KERNEL_DIR/" "$SCRATCH_KERNEL_DIR/"
rsync -a --delete "$TOKENIZER_DIR/" "$SCRATCH_TOKENIZER_DIR/"
rsync -a --delete "$VALIDATION_DATASET_DIR/" "$SCRATCH_VALIDATION_DIR/"
rsync -a --delete "$MODEL_DIR/" "$SCRATCH_MODEL_DIR/"

# ==========================================================================
# RUN
# ==========================================================================

CMD_ARGS=(
    "--kernel-dir" "$SCRATCH_KERNEL_DIR"
    "--tokenizer-dir" "$SCRATCH_TOKENIZER_DIR"
    "--eval-dataset-dir" "$SCRATCH_VALIDATION_DIR"
    "--model-load-dir" "$SCRATCH_MODEL_DIR"
    "--output-dir" "$SCRATCH_OUTPUT_DIR"
    "--scales" $SCALES
    "--per-device-eval-batch-size" "$PER_DEVICE_EVAL_BATCH_SIZE"
    "--semantic-eval-batch-size" "$SEMANTIC_EVAL_BATCH_SIZE"
    "--embedding-batch-size" "$EMBEDDING_BATCH_SIZE"
    $MIXED_PRECISION
)

START=$(date +%s)
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" scripts/validation_variance_rescaling.py "${CMD_ARGS[@]}"
else
    python scripts/validation_variance_rescaling.py "${CMD_ARGS[@]}"
fi
END=$(date +%s)

echo "Syncing sweep outputs to project storage..."
rsync -a --delete "$SCRATCH_OUTPUT_DIR/" "$PROJECT_OUTPUT_DIR/"

DUR=$((END - START))
echo "Sweep completed in $((DUR / 3600))h $(((DUR % 3600) / 60))m $((DUR % 60))s"

echo "Cleaning scratch-local"
rm -rf "$SCRATCH_BASE"

echo "=============================================="
echo "Outputs under: $PROJECT_OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================="
