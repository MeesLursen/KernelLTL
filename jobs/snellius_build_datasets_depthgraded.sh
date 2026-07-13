#!/bin/bash
#SBATCH --job-name=kernelltl_build_depthgraded
#SBATCH --output=logs/kernelltl_build_depthgraded_%j.out
#SBATCH --error=logs/kernelltl_build_depthgraded_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

# ============================================================================
# Depth-graded rebuild of curriculum / eval / validation / finetune datasets.
#
# Reuses (does NOT recompute): the regenerated kernel_v2, the recomputed stage4
# superset embeddings (datasets_v2/curriculum/stage4/train), and its row-aligned
# satvec hashes (datasets_v2/_hashes/stage4_train.npy). No 445 GB re-read.
#
#   stage_i train = depth <= (i+1)   (sizes ~99k/209k/429k/890k, minus/plus depth-2 edits)
#   eval per depth = 250/250/500/1000 (exact-doubling stage totals 250/500/1000/2000)
#   validation     = 1000 per depth  (4000)
#   depth-2 holdout = 1250 classes, drawn from the finite depth-2 census
#   finetune       = exactly 60,000 equivalents + 30,000 near-miss
#
# Resumable: the census + per-row depths are cached under OUTPUT_ROOT/_cache, and the
# finetune object is skipped if already saved. Writes a fresh datasets_v3 tree
# (non-destructive; reads from datasets_v2).
#
# Usage:  sbatch jobs/snellius_build_datasets_depthgraded.sh
# ============================================================================

# ---------------------------- CONFIGURATION --------------------------------
PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel_v2"
RECOMPUTE_DIR="$PROJECT_DIR/artifacts/datasets_v2/curriculum/stage4/train"   # formulas + recomputed embeddings
STAGE4_HASHES="$PROJECT_DIR/artifacts/datasets_v2/_hashes/stage4_train.npy"  # row-aligned satvec hashes
OLD_EVAL_DIR="$PROJECT_DIR/artifacts/datasets/stage4/eval"                    # seeds (free reps + depths 3-5)
OLD_VAL_DIR="$PROJECT_DIR/artifacts/datasets/validation"                      # seeds
OUTPUT_ROOT="$PROJECT_DIR/artifacts/datasets_v3"

# Sizes
EVAL_PER_DEPTH=(250 250 500 1000)   # d2 d3 d4 d5
VAL_PER_DEPTH=(1000 1000 1000 1000) # d2 d3 d4 d5
FINETUNE_SAMPLE_COUNT=30000

# Sampling / batching
P_LEAF_LOW=0.1
P_LEAF_HIGH=0.5
SAMPLE_BATCH=51200
ATTEMPT_BUDGET=5000000
SATISFACTION_BATCH_SIZE=256000
EMBED_BATCH=512
SEED=0

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Depth-Graded Dataset Build"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"
mkdir -p "$OUTPUT_ROOT"

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
echo "Python: $(python --version)"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $NUM_GPUS"
if [ "$NUM_GPUS" -lt 1 ]; then echo "Expected >=1 GPU, found $NUM_GPUS"; exit 1; fi

# ---------------------------- RUN ------------------------------------------
CMD=(
    python -u scripts/build_datasets_depthgraded.py
    --kernel-dir "$KERNEL_DIR"
    --stage4-recompute-dir "$RECOMPUTE_DIR"
    --stage4-hashes "$STAGE4_HASHES"
    --old-eval-dir "$OLD_EVAL_DIR"
    --old-val-dir "$OLD_VAL_DIR"
    --output-root "$OUTPUT_ROOT"
    --eval-per-depth "${EVAL_PER_DEPTH[@]}"
    --val-per-depth "${VAL_PER_DEPTH[@]}"
    --finetune-sample-count "$FINETUNE_SAMPLE_COUNT"
    --p-leaf-range "$P_LEAF_LOW" "$P_LEAF_HIGH"
    --sample-batch "$SAMPLE_BATCH"
    --attempt-budget "$ATTEMPT_BUDGET"
    --satisfaction-batch-size "$SATISFACTION_BATCH_SIZE"
    --embed-batch "$EMBED_BATCH"
    --seed "$SEED"
    --device cuda
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Depth-graded build complete!  End: $(date)"
echo "Output root: $OUTPUT_ROOT"
ls -R "$OUTPUT_ROOT" | grep -v "^$" | head -60
echo "=============================================="
