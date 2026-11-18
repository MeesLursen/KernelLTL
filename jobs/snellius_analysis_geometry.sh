#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_geometry
#SBATCH --output=logs/kernelltl_analysis_geometry_%j.out
#SBATCH --error=logs/kernelltl_analysis_geometry_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

# ============================================================================
# Geometry analysis of the kernel instance (thesis Sec. kernel_geometry).
#
# Kernel tensors -> rank, leverage division, random sets, reference differences,
# plus the fixed objects (spectrum, leverage, trace processes) that Experiment 2
# reuses. Needs one GPU; counts are exact fp32, all algebra float64.
#
# Run AFTER jobs/snellius_recover_trace_provenance.sh (for the per-trace process
# labels; without them the flip-count rule is used and the manifest says so).
# Afterwards: jobs/snellius_analysis_exp2_features.sh.
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TRACE_ORIGIN="$PROJECT_DIR/artifacts/kernel_provenance/trace_origin.npy"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/geometry"

SEED=0
N_SIZES=25
N_REPS=20

echo "=============================================="
echo "KernelLTL geometry analysis"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

nvidia-smi -L
echo "Code: $(git -C "$HOME_DIR" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"

ORIGIN_ARGS=()
if [ -f "$TRACE_ORIGIN" ]; then
    ORIGIN_ARGS=(--trace-origin "$TRACE_ORIGIN")
else
    echo "WARNING: $TRACE_ORIGIN not found; trace processes will come from the flip-count rule."
fi

CMD=(
    python -u -m analysis.geometry.run_geometry
    --kernel-dir "$KERNEL_DIR"
    --output-dir "$OUTPUT_DIR"
    --device cuda
    --seed "$SEED"
    --n-sizes "$N_SIZES"
    --n-reps "$N_REPS"
    ${ORIGIN_ARGS[@]+"${ORIGIN_ARGS[@]}"}
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Geometry analysis complete!  End: $(date)"
ls -la "$OUTPUT_DIR" "$OUTPUT_DIR/fixed"
echo "=============================================="
