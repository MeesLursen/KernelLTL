#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_exp1
#SBATCH --output=logs/kernelltl_analysis_exp1_%j.out
#SBATCH --error=logs/kernelltl_analysis_exp1_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

# ============================================================================
# Experiment 1 post-hoc analysis: validation JSONLs -> statistical tables.
#
# Pure CPU: reads the per-generation records written by snellius_validate_models.sh
# and snellius_validate_ablation.sh, emits tidy CSVs (point estimates + 95%
# percentile-bootstrap CIs) plus a manifest mapping thesis objects to files.
# No plotting -- data-viz happens locally on the synced tables.
#
# Run AFTER the conditioned validation run and the three ablation runs.
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

VALIDATION_ROOT="$PROJECT_DIR/artifacts/validation"
RUN_NAME="ce_base"                                   # conditioned run to analyse
DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/exp1"

TOP_K=5
BOOTSTRAP_SAMPLES=10000
SEED=0

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL Experiment 1 analysis"
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
    python -u scripts/analysis_exp1/run_exp1.py
    --run-dir "$VALIDATION_ROOT/$RUN_NAME"
    --ablation-dir "zero=$VALIDATION_ROOT/${RUN_NAME}_ablation_zero"
    --ablation-dir "mean=$VALIDATION_ROOT/${RUN_NAME}_ablation_mean"
    --ablation-dir "shuffle=$VALIDATION_ROOT/${RUN_NAME}_ablation_shuffle"
    --dataset-dir "$DATASET_DIR"
    --output-dir "$OUTPUT_DIR"
    --top-k "$TOP_K"
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
