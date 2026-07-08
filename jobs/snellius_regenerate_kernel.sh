#!/bin/bash
#SBATCH --job-name=kernelltl_regenerate_kernel
#SBATCH --output=logs/kernelltl_regenerate_kernel_%j.out
#SBATCH --error=logs/kernelltl_regenerate_kernel_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

# ============================================================================
# Regenerate the kernel's anchor set + feature matrix F, reusing the SOURCE
# kernel's traces so every stored satvec stays valid and only embeddings change.
#
# The anchors are resampled under the symmetric Hamming-band gate (|sim^pm| <= tau)
# with explicit trivial rejection; F is rebuilt from 0/1 anchor satisfactions.
#
# Resumable per object via the saved-kernel state (metadata.json is the sentinel):
# if the output already has anchors, they are reused and only F is (re)built.
# To force a clean rerun, delete OUTPUT_KERNEL_DIR first.
#
# Usage:  sbatch jobs/snellius_regenerate_kernel.sh
# ============================================================================

# ---------------------------- CONFIGURATION --------------------------------
PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

SOURCE_KERNEL_DIR="$HOME_DIR/artifacts/kernel"          # existing kernel (traces reused, never resampled)
OUTPUT_KERNEL_DIR="$PROJECT_DIR/artifacts/kernel_v2"    # regenerated kernel (F.pt ~2GB -> project storage)

# Anchor sampling (thesis values: M=1024, tau=0.6, max depth 6)
ANCHOR_COUNT=1024
THRESHOLD=0.5
ANCHOR_MAX_DEPTH=5
ANCHOR_P_LEAF_LOW=0.4
ANCHOR_P_LEAF_HIGH=0.6
COSINE_BATCH_SIZE=10240
MAX_ATTEMPTS=10000
SEED=1                                                  # RNG seed for anchor sampling
BUILD_F_BATCH_SIZE=512000

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Kernel Regeneration"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"
mkdir -p "$(dirname "$OUTPUT_KERNEL_DIR")"

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
    python -u scripts/regenerate_kernel.py
    --source-kernel-dir "$SOURCE_KERNEL_DIR"
    --output-dir "$OUTPUT_KERNEL_DIR"
    --anchor-count "$ANCHOR_COUNT"
    --threshold "$THRESHOLD"
    --anchor-max-depth "$ANCHOR_MAX_DEPTH"
    --anchor-p-leaf-range "$ANCHOR_P_LEAF_LOW" "$ANCHOR_P_LEAF_HIGH"
    --cosine-batch-size "$COSINE_BATCH_SIZE"
    --max-attempts "$MAX_ATTEMPTS"
    --seed "$SEED"
    --build-f-batch-size "$BUILD_F_BATCH_SIZE"
    --device cuda
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Kernel regeneration complete!  End: $(date)"
echo "New kernel: $OUTPUT_KERNEL_DIR"
ls -lh "$OUTPUT_KERNEL_DIR/"
echo "Next: sbatch jobs/snellius_rebuild_datasets.sh"
echo "=============================================="
