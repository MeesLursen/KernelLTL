#!/bin/bash
#SBATCH --job-name=kernelltl_rebuild_datasets
#SBATCH --output=logs/kernelltl_rebuild_datasets_%j.out
#SBATCH --error=logs/kernelltl_rebuild_datasets_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

# ============================================================================
# Rebuild curriculum / finetune / eval / validation datasets against the
# regenerated kernel, with satvec-level disjointness and reproducible embeddings.
#
# Phases (each is a resumable object; re-running skips completed ones):
#   1. recompute curriculum embeddings (re-project stored satvecs through new F)
#   2. build finetune (30k unique non-trivial bases, 2 equiv rewrites + 1 negation)
#   3. build the semantic exclusion set (curriculum U finetune satvec hashes)
#   4. fill depth bins 2..5 (reverse order, satvec-disjoint + unique + non-trivial)
#   5. split -> validation + new stage4/eval; derive stage1..3 eval by depth
#
# Reads stored satvecs via mmap directly from project storage (~445 GB for stage4
# train); nothing is staged to scratch. This is a long job -- if it hits the wall
# clock, just resubmit: every finished object (metadata.json / _DONE) is skipped.
#
# Usage:  sbatch jobs/snellius_rebuild_datasets.sh
# ============================================================================

# ---------------------------- CONFIGURATION --------------------------------
PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel_v2"           # regenerated kernel (from snellius_regenerate_kernel.sh)
DATASETS_DIR="$PROJECT_DIR/artifacts/datasets"          # existing datasets (source; must hold stored satvecs)
OUTPUT_ROOT="$PROJECT_DIR/artifacts/datasets_v2"        # rebuilt datasets (non-destructive)

STAGE_TRAIN_DIRS=(
    "$DATASETS_DIR/stage0/train"
    "$DATASETS_DIR/stage1/train"
    "$DATASETS_DIR/stage2/train"
    "$DATASETS_DIR/stage3/train"
    "$DATASETS_DIR/stage4/train"
)
STAGE4_TRAIN_DIR="$DATASETS_DIR/stage4/train"           # finetune base pool
STAGE4_EVAL_DIR="$DATASETS_DIR/stage4/eval"             # seeds the depth bins

# Generation parameters
FINETUNE_SAMPLE_COUNT=30000
BIN_TARGET=10000
P_LEAF_LOW=0.1
P_LEAF_HIGH=0.5
SAMPLE_BATCH=51200
ATTEMPT_BUDGET=50000000
SATISFACTION_BATCH_SIZE=256000
EMBED_BATCH=512
SEED=0

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Dataset Rebuild"
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
    python -u scripts/rebuild_datasets.py
    --kernel-dir "$KERNEL_DIR"
    --stage-train-dirs "${STAGE_TRAIN_DIRS[@]}"
    --stage4-train-dir "$STAGE4_TRAIN_DIR"
    --stage4-eval-dir "$STAGE4_EVAL_DIR"
    --output-root "$OUTPUT_ROOT"
    --finetune-sample-count "$FINETUNE_SAMPLE_COUNT"
    --bin-target "$BIN_TARGET"
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
echo "Dataset rebuild complete!  End: $(date)"
echo "Output root: $OUTPUT_ROOT"
ls -R "$OUTPUT_ROOT" | head -60
echo "=============================================="
