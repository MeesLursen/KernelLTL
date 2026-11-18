#!/bin/bash
#SBATCH --job-name=kernelltl_trace_provenance
#SBATCH --output=logs/kernelltl_trace_provenance_%j.out
#SBATCH --error=logs/kernelltl_trace_provenance_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G

set -euo pipefail

# ============================================================================
# One-off: replay the December 2025 trace draw (job 17762902, gpu_a100) and label
# every row of the thesis kernel's traces.pt with the process that drew it.
#
# Must run on an A100 with torch 2.8.0: the CUDA random stream depends on both.
# Labels are written only if the replay reproduces traces.pt bit for bit; on a
# mismatch the script writes report.json with diagnostics and exits with status 2.
#
# Usage:  sbatch jobs/snellius_recover_trace_provenance.sh
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"                 # thesis kernel (its traces.pt is the December sample)
OUTPUT_DIR="$PROJECT_DIR/artifacts/kernel_provenance"      # new directory; the kernel dir is left untouched

echo "=============================================="
echo "KernelLTL trace provenance (one-off replay)"
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

python -u scripts/recover_trace_provenance.py \
    --kernel-dir "$KERNEL_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Done: $(date)"
ls -la "$OUTPUT_DIR"
