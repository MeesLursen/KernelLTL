#!/bin/bash
#SBATCH --job-name=kernelltl_validate_ablation
#SBATCH --output=logs/kernelltl_validate_ablation_%j.out
#SBATCH --error=logs/kernelltl_validate_ablation_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G

# ==========================================================================
# Embedding-ablation FLOOR (G1b).
#
# Re-runs the GREEDY validation pass of the CE base under destroyed/corrupted
# conditioning, to establish the floor above which "the model conditions on the
# embedding" is a meaningful claim (RQ1). The corruption is NOT applied in the
# pipeline anymore: it is baked into three pre-built datasets whose embeddings
# are already corrupted (over the non-trivial targets only), while every row /
# formula_id stays aligned with the original validation set:
#   validation_ablation_zero    -- unconditional prior (zero embedding)
#   validation_ablation_mean    -- constant non-trivial-mean embedding
#   validation_ablation_shuffle -- another non-trivial target's embedding
#
# Each dataset's satisfactions.pt is a relative symlink to ../validation, so we
# stage the original validation dir alongside the ablation dirs and rsync WITHOUT
# -L (symlinks preserved, satvecs not triplicated). Output folders
# ce_base_ablation_<mode> sit next to the real runs and are loadable by
# visualize_validation_geometry.py as extra "runs" for the feasibility floor.
# Greedy-only by construction (validation_ablation.py): no top-K, no KL ref.
# ==========================================================================

set -e

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"
DATASETS_DIR="$PROJECT_DIR/artifacts/datasets"
CE_BASE_MODEL_DIR="$PROJECT_DIR/artifacts/models/CE/final_pretrain/stage4/final_model/"

PROJECT_OUTPUT_BASE="$PROJECT_DIR/artifacts/validation"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"

PER_DEVICE_EVAL_BATCH_SIZE=96
SEMANTIC_EVAL_BATCH_SIZE="256000"
MIXED_PRECISION="--bf16"

ABLATIONS=(zero mean shuffle)

echo "=============================================="
echo "KernelLTL embedding-ablation floor (G1b)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Node: ${SLURMD_NODENAME:-N/A}  Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"; source "$VENV_DIR/bin/activate"
    pip install --upgrade pip; pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# Stage shared artifacts + the CE base model onto scratch-local once.
SCRATCH_KERNEL_DIR="$SCRATCH_BASE/kernel"
SCRATCH_TOKENIZER_DIR="$SCRATCH_BASE/tokenizer"
SCRATCH_DATASETS_DIR="$SCRATCH_BASE/datasets"
SCRATCH_MODEL_DIR="$SCRATCH_BASE/models/ce_base/final_model"
mkdir -p "$SCRATCH_KERNEL_DIR" "$SCRATCH_TOKENIZER_DIR" "$SCRATCH_DATASETS_DIR" "$SCRATCH_MODEL_DIR"

echo "Staging shared artifacts to scratch-local..."
rsync -a --delete "$KERNEL_DIR/" "$SCRATCH_KERNEL_DIR/"
rsync -a --delete "$TOKENIZER_DIR/" "$SCRATCH_TOKENIZER_DIR/"
rsync -a --delete "$CE_BASE_MODEL_DIR/" "$SCRATCH_MODEL_DIR/"

# The original validation dir holds the shared satisfactions.pt that each ablation
# dataset symlinks to (../validation/satisfactions.pt). Stage it first so the
# relative symlinks resolve on scratch. rsync -a (NOT -L) preserves the symlinks
# and avoids triplicating the satvecs.
echo "Staging validation + ablation datasets to scratch-local..."
rsync -a --delete "$DATASETS_DIR/validation/" "$SCRATCH_DATASETS_DIR/validation/"
for ABL in "${ABLATIONS[@]}"; do
    rsync -a --delete "$DATASETS_DIR/validation_ablation_$ABL/" \
        "$SCRATCH_DATASETS_DIR/validation_ablation_$ABL/"
done

for ABL in "${ABLATIONS[@]}"; do
    RUN_NAME="ce_base_ablation_$ABL"
    SCRATCH_DATASET_DIR="$SCRATCH_DATASETS_DIR/validation_ablation_$ABL"
    SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/validation/$RUN_NAME"
    PROJECT_OUTPUT_DIR="$PROJECT_OUTPUT_BASE/$RUN_NAME"
    mkdir -p "$SCRATCH_OUTPUT_DIR" "$PROJECT_OUTPUT_DIR"

    echo ""
    echo "=============================================="
    echo "Ablation: $ABL  ->  $RUN_NAME"
    echo "Dataset:  $SCRATCH_DATASET_DIR"
    echo "=============================================="

    CMD_ARGS=(
        "--kernel-dir" "$SCRATCH_KERNEL_DIR"
        "--tokenizer-dir" "$SCRATCH_TOKENIZER_DIR"
        "--eval-dataset-dir" "$SCRATCH_DATASET_DIR"
        "--model-load-dir" "$SCRATCH_MODEL_DIR"
        "--output-dir" "$SCRATCH_OUTPUT_DIR"
        "--per-device-eval-batch-size" "$PER_DEVICE_EVAL_BATCH_SIZE"
        "--semantic-eval-batch-size" "$SEMANTIC_EVAL_BATCH_SIZE"
        $MIXED_PRECISION
    )

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" scripts/validation_ablation.py "${CMD_ARGS[@]}"
    else
        python scripts/validation_ablation.py "${CMD_ARGS[@]}"
    fi

    echo "Syncing $RUN_NAME outputs to project storage..."
    rsync -a --delete "$SCRATCH_OUTPUT_DIR/" "$PROJECT_OUTPUT_DIR/"
done

echo ""
echo "Cleaning scratch-local"
rm -rf "$SCRATCH_BASE"
echo "All ablation runs completed! End: $(date)"
