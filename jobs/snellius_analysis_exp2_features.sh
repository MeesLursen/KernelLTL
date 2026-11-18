#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_exp2_features
#SBATCH --output=logs/kernelltl_analysis_exp2_features_%j.out
#SBATCH --error=logs/kernelltl_analysis_exp2_features_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

# ============================================================================
# Experiment 2, step 1 of 2: the representation quantities of every validation
# target, of its 10 nearest validation neighbours, and of every greedy miss.
#
# Uses the fixed objects of the geometry analysis as they are, so run it AFTER
# jobs/snellius_analysis_geometry.sh. Most of the time goes into ~40k group
# leverages (one eigenvalue problem each). Afterwards: jobs/snellius_analysis_exp2.sh.
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
GEOMETRY_DIR="$PROJECT_DIR/artifacts/analysis/geometry"
DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"
RUN_DIR="$PROJECT_DIR/artifacts/validation/ce_base"
OUTPUT_DIR="$PROJECT_DIR/artifacts/analysis/exp2/features"

KS="1,5,10"            # neighbour counts; 10 is reported, 1 and 5 are the sensitivity check

echo "=============================================="
echo "KernelLTL Experiment 2 features"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

if [ ! -f "$GEOMETRY_DIR/fixed/spectrum.npz" ]; then
    echo "Missing $GEOMETRY_DIR/fixed/spectrum.npz: run jobs/snellius_analysis_geometry.sh first."
    exit 1
fi

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

nvidia-smi -L
echo "Code: $(git -C "$HOME_DIR" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"

CMD=(
    python -u -m analysis.exp2.features
    --kernel-dir "$KERNEL_DIR"
    --geometry-dir "$GEOMETRY_DIR"
    --dataset-dir "$DATASET_DIR"
    --run-dir "$RUN_DIR"
    --output-dir "$OUTPUT_DIR"
    --ks "$KS"
    --device cuda
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Experiment 2 features complete!  End: $(date)"
ls -la "$OUTPUT_DIR"
echo "=============================================="
