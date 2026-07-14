#!/bin/bash
#SBATCH --job-name=kernelltl_build_val_ablation
#SBATCH --output=logs/kernelltl_build_val_ablation_%j.out
#SBATCH --error=logs/kernelltl_build_val_ablation_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# ============================================================================
# Build the three embedding-ablation datasets for the G1b feasibility floor:
#   validation_ablation_{zero,mean,shuffle}  (siblings of validation, satvecs symlinked).
#
# CPU-only and near-instant (reads a ~16 MB embedding tensor, writes three corrupted
# copies + relative symlinks). The GPU allocation is only to match the working job
# convention; no GPU is used. Run AFTER the depth-graded build produced datasets/validation
# and BEFORE snellius_validate_ablation.sh (which rsyncs these dirs onto scratch).
#
# Usage:  sbatch jobs/snellius_build_validation_ablation.sh
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

VALIDATION_DIR="$PROJECT_DIR/artifacts/datasets/validation"
SEED=0

echo "=============================================="
echo "KernelLTL validation-ablation build"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"

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

python -u scripts/build_validation_ablation.py --validation-dir "$VALIDATION_DIR" --seed "$SEED"

echo ""
echo "=============================================="
echo "Ablation datasets built!  End: $(date)"
ls -la "$(dirname "$VALIDATION_DIR")"/validation_ablation_* 2>/dev/null
echo "=============================================="
