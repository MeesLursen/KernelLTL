#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_extra
#SBATCH --output=logs/kernelltl_analysis_extra_%j.out
#SBATCH --error=logs/kernelltl_analysis_extra_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G

# ==========================================================================
# Snellius CPU analysis script for KernelLTL — extra validation analyses
#
# Runs visualize_validation_extra.py, which produces the conditional
# descriptive metrics (depth/length gaps and semantic distance sliced by
# correctness), top-K diagnostics (pass@k', distinct-correct counts),
# target-side operator analysis (KL, decomposition, log-odds, logistic
# regression), and the contrast studies (per-target paired diffs, pairwise
# Cohen's κ + McNemar agreement matrices, pairwise output-similarity heatmap)
# across the same five trained models as the main analysis.
#
# Pure CPU workload; the partition request matches snellius_analysis.sh for
# consistency, no GPU is actually used. Outputs go under the same
# ``_analysis`` directory as the main script, in ``figures/extra/`` and
# ``stats/extra/`` subdirectories.
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
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"

ANALYSIS_OUTPUT_DIR="$PROJECT_DIR/artifacts/validation/_analysis"

RUNS=(
    "ce_base"
    "ce_finetune"
    "rb_momentum_09"
    "gae_lambda_09"
    "gae_lambda_1"
)
REFERENCE_RUN="ce_base"

BOOTSTRAP_N=2000
ALPHA=0.05
RNG_SEED=0
DPI=200

# ==========================================================================
# ENVIRONMENT SETUP
# ==========================================================================

echo "=============================================="
echo "KernelLTL Extra Validation Analysis"
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
# EXTRA VALIDATION ANALYSIS
# ==========================================================================

echo ""
echo "=============================================="
echo "Running visualize_validation_extra.py"
echo "  Runs:      ${RUNS[*]}"
echo "  Reference: $REFERENCE_RUN"
echo "  Output:    $ANALYSIS_OUTPUT_DIR"
echo "=============================================="

START=$(date +%s)

python -u scripts/visualize_validation_extra.py \
    --validation-root "$VALIDATION_ROOT" \
    --dataset-dir     "$DATASET_DIR" \
    --output-dir      "$ANALYSIS_OUTPUT_DIR" \
    --runs            "${RUNS[@]}" \
    --reference-run   "$REFERENCE_RUN" \
    --tokenizer-dir   "$TOKENIZER_DIR" \
    --bootstrap-n     "$BOOTSTRAP_N" \
    --alpha           "$ALPHA" \
    --rng-seed        "$RNG_SEED" \
    --dpi             "$DPI" \
    --logistic-regularized-fallback

# --- dual-reference: ce_finetune (isolates the RL objective from finetuning), separate dir ---
REF2="ce_finetune"
ANALYSIS_OUTPUT_DIR2="$ANALYSIS_OUTPUT_DIR/ref_ce_finetune"
mkdir -p "$ANALYSIS_OUTPUT_DIR2"
echo "Second reference: $REF2 -> $ANALYSIS_OUTPUT_DIR2"
python -u scripts/visualize_validation_extra.py \
    --validation-root "$VALIDATION_ROOT" \
    --dataset-dir     "$DATASET_DIR" \
    --output-dir      "$ANALYSIS_OUTPUT_DIR2" \
    --runs            "${RUNS[@]}" \
    --reference-run   "$REF2" \
    --tokenizer-dir   "$TOKENIZER_DIR" \
    --bootstrap-n     "$BOOTSTRAP_N" \
    --alpha           "$ALPHA" \
    --rng-seed        "$RNG_SEED" \
    --dpi             "$DPI" \
    --logistic-regularized-fallback

END=$(date +%s)
DURATION=$((END - START))

# ==========================================================================
# SUMMARY
# ==========================================================================

echo ""
echo "=============================================="
echo "Extra analysis complete!"
echo "Duration:  $((DURATION / 60))m $((DURATION % 60))s"
echo "End:       $(date)"
echo ""
echo "Outputs:"
echo "  Figures:  $ANALYSIS_OUTPUT_DIR/figures/extra/"
echo "  Stats:    $ANALYSIS_OUTPUT_DIR/stats/extra/"
echo "  Metadata: $ANALYSIS_OUTPUT_DIR/run_metadata_extra.json"
echo "=============================================="
