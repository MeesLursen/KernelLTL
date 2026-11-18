#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_exp2
#SBATCH --output=logs/kernelltl_analysis_exp2_%j.out
#SBATCH --error=logs/kernelltl_analysis_exp2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

# ============================================================================
# Experiment 2, step 2 of 2 (CPU): feature tables -> statistical tables.
#
# What the representation keeps, the misses (eq. miss_terms), and the regression
# ladder M0-M3 with its specification search, checks and 10,000-resample
# bootstrap (parallel over the allocated cores). Needs statsmodels and scipy in
# the venv. Run AFTER jobs/snellius_analysis_exp2_features.sh.
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

FEATURES_DIR="$PROJECT_DIR/artifacts/analysis/exp2/features"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/exp2"

K=10
SENSITIVITY_KS="1,5"
BOOTSTRAP_SAMPLES=10000
SEED=0

echo "=============================================="
echo "KernelLTL Experiment 2 analysis"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "$HOME_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"
# one BLAS thread per bootstrap worker
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -c "import statsmodels, scipy" || {
    echo "statsmodels / scipy missing in $VENV_DIR (see requirements.txt)"; exit 1; }
echo "Code: $(git -C "$HOME_DIR" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"

CMD=(
    python -u -m analysis.exp2.run_exp2
    --features-dir "$FEATURES_DIR"
    --output-dir "$OUTPUT_DIR"
    --k "$K"
    --sensitivity-ks "$SENSITIVITY_KS"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --seed "$SEED"
    --workers "${SLURM_CPUS_PER_TASK:-16}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Experiment 2 analysis complete!  End: $(date)"
ls -la "$OUTPUT_DIR"
echo "=============================================="
