#!/bin/bash
#SBATCH --job-name=kernelltl_analysis
#SBATCH --output=logs/kernelltl_analysis_%j.out
#SBATCH --error=logs/kernelltl_analysis_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# ==========================================================================
# Snellius CPU analysis script for KernelLTL
#
# Runs:
#   1. analyze_dataset.py  — dataset characterisation (depth/length/operator
#      distributions, tree shape-bias diagnostics via Boltzmann P1 reference)
#   2. visualize_validation.py — cross-model comparison of validation outputs
#      across all 5 trained models (bars, ECDFs, per-depth, Pareto, radar,
#      paired statistics with bootstrap CIs)
#
# Pure CPU job — no GPU required.
# ==========================================================================

set -e

# ==========================================================================
# CONFIGURATION
# ==========================================================================

HOME_DIR="$HOME/KernelLTL"
PROJECT_DIR="/projects/prjs2029/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

VALIDATION_ROOT="$PROJECT_DIR/artifacts/validation"
DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"
TOKENIZER_DIR="$HOME_DIR/artifacts/tokenizer"

ANALYSIS_OUTPUT_DIR="$PROJECT_DIR/artifacts/validation/_analysis"
DATASET_ANALYSIS_OUTPUT_DIR="$ANALYSIS_OUTPUT_DIR/dataset_analysis/validation"

RUNS=(
    "ce_base"
    "ce_finetune"
    "rb_momentum_09_lr_5e-8"
    "gae_lambda_09_lr_5e-8_crlr_1e-3"
    "gae_lambda_1_lr_5e-8_crlr_5e-3"
)
REFERENCE_RUN="ce_base"

BOOTSTRAP_N=10000
ALPHA=0.05
MC_SHAPE_N=100000
RNG_SEED=0
DPI=200

# ==========================================================================
# ENVIRONMENT SETUP
# ==========================================================================

echo "=============================================="
echo "KernelLTL Analysis"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"
mkdir -p "$ANALYSIS_OUTPUT_DIR"
mkdir -p "$DATASET_ANALYSIS_OUTPUT_DIR"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

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
echo "PYTHONPATH: $PYTHONPATH"

# ==========================================================================
# 1. DATASET ANALYSIS
# ==========================================================================

echo ""
echo "=============================================="
echo "Step 1: Dataset analysis"
echo "  Input:  $DATASET_DIR"
echo "  Output: $DATASET_ANALYSIS_OUTPUT_DIR"
echo "=============================================="

STEP1_START=$(date +%s)

python -u scripts/analyze_dataset.py \
    --dataset-dir "$DATASET_DIR" \
    --output-dir  "$DATASET_ANALYSIS_OUTPUT_DIR" \
    --mc-shape-n  "$MC_SHAPE_N" \
    --rng-seed    "$RNG_SEED" \
    --dpi         "$DPI"

STEP1_END=$(date +%s)
STEP1_DUR=$((STEP1_END - STEP1_START))
echo "Dataset analysis done in $((STEP1_DUR / 60))m $((STEP1_DUR % 60))s"

# ==========================================================================
# 2. VALIDATION ANALYSIS
# ==========================================================================

echo ""
echo "=============================================="
echo "Step 2: Validation analysis"
echo "  Runs:      ${RUNS[*]}"
echo "  Reference: $REFERENCE_RUN"
echo "  Output:    $ANALYSIS_OUTPUT_DIR"
echo "=============================================="

STEP2_START=$(date +%s)

python -u scripts/visualize_validation.py \
    --validation-root "$VALIDATION_ROOT" \
    --dataset-dir     "$DATASET_DIR" \
    --output-dir      "$ANALYSIS_OUTPUT_DIR" \
    --runs            "${RUNS[@]}" \
    --reference-run   "$REFERENCE_RUN" \
    --tokenizer-dir   "$TOKENIZER_DIR" \
    --bootstrap-n     "$BOOTSTRAP_N" \
    --alpha           "$ALPHA" \
    --rng-seed        "$RNG_SEED" \
    --dpi             "$DPI"

STEP2_END=$(date +%s)
STEP2_DUR=$((STEP2_END - STEP2_START))
echo "Validation analysis done in $((STEP2_DUR / 60))m $((STEP2_DUR % 60))s"

# ==========================================================================
# SUMMARY
# ==========================================================================

TOTAL_DUR=$((STEP2_END - STEP1_START))
echo ""
echo "=============================================="
echo "All analyses complete!"
echo "Total time: $((TOTAL_DUR / 60))m $((TOTAL_DUR % 60))s"
echo "End: $(date)"
echo ""
echo "Outputs:"
echo "  Dataset analysis: $DATASET_ANALYSIS_OUTPUT_DIR"
echo "  Validation:       $ANALYSIS_OUTPUT_DIR"
echo "  Summary:          $ANALYSIS_OUTPUT_DIR/summary.md"
echo "=============================================="
