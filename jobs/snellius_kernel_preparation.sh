#!/bin/bash
#SBATCH --job-name=kernelltl_kernel
#SBATCH --output=logs/kernelltl_kernel_%j.out
#SBATCH --error=logs/kernelltl_kernel_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G

# ============================================================================
# Snellius Job Script for KernelLTL Kernel Generation
# ============================================================================
#
# This script generates the LTL kernel (traces + anchor formulas + feature matrix F)
# and copies it from scratch to the home directory for persistence.
#
# Usage:
#   sbatch snellius_kernel_preparation.sh
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION
# ============================================================================

PROJECT_DIR="$HOME/KernelLTL"
VENV_DIR="$PROJECT_DIR/venv"

# Scratch directory for fast I/O during generation
SCRATCH_OUTPUT="/scratch-shared/$USER/KernelLTL/kernel"

# Final destination in home directory (persistent storage)
FINAL_OUTPUT="$PROJECT_DIR/artifacts/kernel"

# ============================================================================
# KERNEL PARAMETERS
# ============================================================================

# Basic parameters
TRACE_LENGTH=20
NUM_ATOMIC_PROPS=5
SEED=1  # For reproducibility; set to empty string for random

# Trace sampling parameters
# Leave NUM_TRACES empty to use epsilon/delta heuristic
NUM_TRACES="500000"
EPSILON=0.01
DELTA=0.01
TRACE_SAMPLER="correlated"  # Options: "iid", "correlated"
LOW_VARIANCE_RATIO=0.3
LOW_VAR_SWITCH_PROB=0.1

# Anchor formula parameters
ANCHOR_COUNT=1024
ANCHOR_SAMPLER="cosine"  # Options: "uniform", "cosine"
ANCHOR_P_LEAF=0.5
ANCHOR_MAX_DEPTH=6
COSINE_BATCH_SIZE=10240
COSINE_THRESHOLD=0.6
COSINE_MAX_ATTEMPTS=1000

# Feature matrix construction
BUILD_F_BATCH_SIZE=1024

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL Kernel Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Create necessary directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$SCRATCH_OUTPUT"
mkdir -p "$(dirname "$FINAL_OUTPUT")"

# Load modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

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

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ============================================================================
# BUILD KERNEL COMMAND
# ============================================================================

CMD=(
    python scripts/prepare_kernel.py
    --output-dir "$SCRATCH_OUTPUT"
    --trace-length "$TRACE_LENGTH"
    --num-atomic-props "$NUM_ATOMIC_PROPS"
    --epsilon "$EPSILON"
    --delta "$DELTA"
    --trace-sampler "$TRACE_SAMPLER"
    --low-variance-ratio "$LOW_VARIANCE_RATIO"
    --low-var-switch-prob "$LOW_VAR_SWITCH_PROB"
    --anchor-count "$ANCHOR_COUNT"
    --anchor-sampler "$ANCHOR_SAMPLER"
    --anchor-p-leaf "$ANCHOR_P_LEAF"
    --anchor-max-depth "$ANCHOR_MAX_DEPTH"
    --cosine-batch-size "$COSINE_BATCH_SIZE"
    --cosine-threshold "$COSINE_THRESHOLD"
    --cosine-max-attempts "$COSINE_MAX_ATTEMPTS"
    --build-f-batch-size "$BUILD_F_BATCH_SIZE"
)

# Add seed if specified
if [ -n "$SEED" ]; then
    CMD+=(--seed "$SEED")
fi

# Add explicit num-traces if specified (overrides epsilon/delta)
if [ -n "$NUM_TRACES" ]; then
    CMD+=(--num-traces "$NUM_TRACES")
fi

# ============================================================================
# GENERATE KERNEL
# ============================================================================

echo ""
echo "=============================================="
echo "Generating kernel with the following parameters:"
echo "  Trace length: $TRACE_LENGTH"
echo "  Atomic propositions: $NUM_ATOMIC_PROPS"
echo "  Trace sampler: $TRACE_SAMPLER"
echo "  Anchor count: $ANCHOR_COUNT"
echo "  Anchor sampler: $ANCHOR_SAMPLER"
echo "  Anchor max depth: $ANCHOR_MAX_DEPTH"
echo "  Seed: ${SEED:-random}"
echo "  Output (scratch): $SCRATCH_OUTPUT"
echo "=============================================="
echo ""

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# ============================================================================
# COPY TO HOME DIRECTORY
# ============================================================================

echo ""
echo "=============================================="
echo "Copying kernel from scratch to home..."
echo "  From: $SCRATCH_OUTPUT"
echo "  To:   $FINAL_OUTPUT"
echo "=============================================="

# Remove old kernel if it exists
if [ -d "$FINAL_OUTPUT" ]; then
    echo "Removing existing kernel at $FINAL_OUTPUT..."
    rm -rf "$FINAL_OUTPUT"
fi

# Copy from scratch to home
cp -r "$SCRATCH_OUTPUT" "$FINAL_OUTPUT"

# Verify the copy
echo ""
echo "Verifying copied kernel..."
ls -la "$FINAL_OUTPUT/"

# ============================================================================
# CLEANUP
# ============================================================================

echo "Cleaning up scratch directory..."
rm -rf "$SCRATCH_OUTPUT"
echo "Scratch cleanup complete."

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "Kernel generation complete!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Kernel saved to: $FINAL_OUTPUT"
echo ""
echo "Contents:"
ls -lh "$FINAL_OUTPUT/"
echo ""
echo "Next step: Run dataset preparation with:"
echo "  sbatch jobs/snellius_dataset_preparation.sh"
