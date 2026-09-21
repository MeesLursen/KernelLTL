#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_exp3
#SBATCH --output=logs/kernelltl_analysis_exp3_%j.out
#SBATCH --error=logs/kernelltl_analysis_exp3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

# ============================================================================
# Experiment 3 post-hoc analysis: does the decoder read the embedding norm?
#
# Test 1 (observational) reads the conditioned greedy run, the zero-ablation
# run (for V_0), the Experiment 2 features (u, variance bins) and the kernel's
# trace sample (to evaluate the ~1.6k generated misses -- CPU, minutes).
# Test 2 (interventional) reads the rescaling sweep of
# snellius_validate_rescaling.sh. Emits tidy CSVs with percentile-bootstrap
# CIs, a manifest, and the figures.
#
# Run AFTER snellius_exp2_features.sh, the validation + ablation runs, and
# snellius_validate_rescaling.sh.
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

FEATURES_DIR="$PROJECT_DIR/artifacts/analysis/exp2/features"
VALIDATION_ROOT="$PROJECT_DIR/artifacts/validation"
RUN_NAME="ce_base"
DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/exp3"

TEST="both"              # 1 | 2 | both
BOOTSTRAP_SAMPLES=10000
SEED=0
EVAL_BATCH_SIZE=65536    # traces per evaluation chunk (Test 1)

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Experiment 3 analysis (test=$TEST)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs" "$OUTPUT_DIR"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "$HOME_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

# ---------------------------- RUN ------------------------------------------
CMD=(
    python -u scripts/analysis_exp3/run_exp3.py
    --test "$TEST"
    --features-dir "$FEATURES_DIR"
    --dataset-dir "$DATASET_DIR"
    --run-dir "$VALIDATION_ROOT/$RUN_NAME"
    --zero-run-dir "$VALIDATION_ROOT/${RUN_NAME}_ablation_zero"
    --kernel-dir "$KERNEL_DIR"
    --rescaling-dir "$VALIDATION_ROOT/${RUN_NAME}_rescaling"
    --output-dir "$OUTPUT_DIR"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --seed "$SEED"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Analysis complete!  End: $(date)"
echo "Tables under: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
echo "=============================================="
