#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_exp2
#SBATCH --output=logs/kernelltl_analysis_exp2_%j.out
#SBATCH --error=logs/kernelltl_analysis_exp2_%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

# ============================================================================
# Experiment 2 post-hoc analysis: cached geometry features + validation
# JSONLs -> statistical tables (Part II, Stages A/B/C).
#
# Pure CPU: reads exp2_features.csv (snellius_exp2_features.sh), the greedy
# records of the conditioned validation run, and the shuffle-ablation run
# (guessability null). Emits tidy CSVs -- the M0/M1/contrast/M2 ladder with
# whole-pipeline percentile-bootstrap CIs, the u-decile curve, the Stage A
# audit tables -- plus a manifest freezing tiers, constants, and the estimand.
# No plotting: data-viz happens locally on the synced tables.
#
# Run AFTER snellius_exp2_features.sh and the validation + ablation runs.
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

FEATURES_DIR="$PROJECT_DIR/artifacts/analysis/exp2/features"
VALIDATION_ROOT="$PROJECT_DIR/artifacts/validation"
RUN_NAME="ce_base"                                   # conditioned run to analyse
DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/exp2/tables"

BOOTSTRAP_SAMPLES=10000
SEED=0

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Experiment 2 analysis"
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
    python -u scripts/analysis_exp2/run_exp2.py
    --features-dir "$FEATURES_DIR"
    --run-dir "$VALIDATION_ROOT/$RUN_NAME"
    --shuffle-run-dir "$VALIDATION_ROOT/${RUN_NAME}_ablation_shuffle"
    --dataset-dir "$DATASET_DIR"
    --output-dir "$OUTPUT_DIR"
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
