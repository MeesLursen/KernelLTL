#!/bin/bash
#SBATCH --job-name=kernelltl_analysis
#SBATCH --output=logs/kernelltl_analysis_%j.out
#SBATCH --error=logs/kernelltl_analysis_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G

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
DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"   # holds trivial_ids.csv (auto-drops tautologies/contradictions)
TOKENIZER_DIR="$HOME_DIR/artifacts/tokenizer"

ANALYSIS_OUTPUT_DIR="$PROJECT_DIR/artifacts/validation/_analysis"

# Each entry is "split_name:dataset_path" — the split name becomes the
# output subdirectory under dataset_analysis/.
DATASET_SPLITS=(
    # "validation:$PROJECT_DIR/artifacts/datasets/validation"
    # "finetune:$PROJECT_DIR/artifacts/datasets/finetune/train"
    "stage4_train:$PROJECT_DIR/artifacts/datasets/stage4/train"
    "stage4_eval:$PROJECT_DIR/artifacts/datasets/stage4/eval"
)

RUNS=(
    "ce_base"
    "ce_finetune"
    "rb_momentum_09"
    "gae_lambda_09"
    "gae_lambda_1"
)
REFERENCE_RUN="ce_base"

# Dataset analysis is the heaviest step and is independent of the trivial-target
# filtering, so it rarely needs recomputing. Control it without editing the file:
#   auto  (default) — run a split only if its output dir is missing/empty
#   force           — always recompute every split
#   skip            — never run (assume dataset_analysis/ is already there)
# e.g.  RUN_DATASET_ANALYSIS=skip sbatch jobs/snellius_analysis.sh
RUN_DATASET_ANALYSIS="${RUN_DATASET_ANALYSIS:-skip}"

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
echo "Step 1: Dataset analysis (mode: $RUN_DATASET_ANALYSIS)"
echo "  Splits: ${DATASET_SPLITS[*]}"
echo "=============================================="

STEP1_START=$(date +%s)

if [ "$RUN_DATASET_ANALYSIS" = "skip" ]; then
    echo "  RUN_DATASET_ANALYSIS=skip — leaving existing dataset_analysis/ untouched."
else
    for SPLIT_ENTRY in "${DATASET_SPLITS[@]}"; do
        SPLIT_NAME="${SPLIT_ENTRY%%:*}"
        SPLIT_PATH="${SPLIT_ENTRY#*:}"
        SPLIT_OUT="$ANALYSIS_OUTPUT_DIR/dataset_analysis/$SPLIT_NAME"
        if [ "$RUN_DATASET_ANALYSIS" = "auto" ] && [ -d "$SPLIT_OUT" ] && [ -n "$(ls -A "$SPLIT_OUT" 2>/dev/null)" ]; then
            echo "  Split '$SPLIT_NAME' already analysed ($SPLIT_OUT non-empty) — skipping. Use RUN_DATASET_ANALYSIS=force to recompute."
            continue
        fi
        echo "  Analysing split '$SPLIT_NAME' -> $SPLIT_PATH"
        mkdir -p "$SPLIT_OUT"
        python -u scripts/analyze_dataset.py \
            --dataset-dir "$SPLIT_PATH" \
            --output-dir "$SPLIT_OUT" \
            --mc-shape-n "$MC_SHAPE_N" \
            --rng-seed "$RNG_SEED" \
            --dpi "$DPI" \
            --mmap-satisfactions
    done
fi

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
