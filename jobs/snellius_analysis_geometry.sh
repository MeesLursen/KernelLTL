#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_geometry
#SBATCH --output=logs/kernelltl_analysis_geometry_%j.out
#SBATCH --error=logs/kernelltl_analysis_geometry_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

# ============================================================================
# Embedding-geometry vs. correctness analysis.
#   Step 1 (heavy): compute_geometry_features.py streams the ~10 GB satisfactions
#                   tensor -> small geometry_features.csv (cached).
#   Step 2 (light): visualize_validation_geometry.py -> stats/extra + figures/extra.
# CPU-only; mmap keeps peak memory modest (~a few GB).
# ============================================================================

HOME_DIR="$HOME/KernelLTL"
PROJECT_DIR="/projects/prjs2029/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# --- assets (adjust if your validation outputs live elsewhere) --------------
KERNEL_DIR="$HOME_DIR/artifacts/kernel"
VAL_DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"   # embeddings.pt + satisfactions.pt
VAL_ROOT="$PROJECT_DIR/artifacts/validation"                   # per-run folders (per_sample/*.jsonl)
ANALYSIS_DIR="$VAL_ROOT/_analysis"
FEATURES_CSV="$ANALYSIS_DIR/geometry_features.csv"

RUNS=(ce_base ce_finetune rb_momentum_09 gae_lambda_09 gae_lambda_1)
REFERENCE_RUN="ce_base"
# Embedding-ablation floor runs (G1b), produced by jobs/snellius_validate_ablation.sh.
# Leave empty to skip the RQ1 feasibility-floor step.
ABLATION_RUNS=(ce_base_ablation_zero ce_base_ablation_mean ce_base_ablation_shuffle)

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Node: ${SLURMD_NODENAME:-N/A}  Start: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs" "$ANALYSIS_DIR"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"; source "$VENV_DIR/bin/activate"
    pip install --upgrade pip; pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi
export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

# --- Step 1: features (skip if cached) --------------------------------------
if [ -f "$FEATURES_CSV" ]; then
    echo "geometry_features.csv exists; skipping feature computation."
else
    echo "Computing geometry features ..."
    python -u scripts/compute_geometry_features.py \
        --validation-dataset-dir "$VAL_DATASET_DIR" \
        --kernel-dir "$KERNEL_DIR" \
        --output "$FEATURES_CSV" \
        --chunk-size 500
fi

# --- Step 2: analysis + figures ---------------------------------------------
echo "Running geometry analysis ..."
python -u scripts/visualize_validation_geometry.py \
    --validation-root "$VAL_ROOT" \
    --geometry-features "$FEATURES_CSV" \
    --output-dir "$ANALYSIS_DIR" \
    --runs "${RUNS[@]}" \
    --ablation-runs "${ABLATION_RUNS[@]}" \
    --reference-run "$REFERENCE_RUN" \
    --bootstrap-n 10000 \
    --alpha 0.05 \
    --rng-seed 0

# --- dual-reference: ce_finetune for the cross-model interactions (RL-objective isolation).
#     Separate output dir; --ablation-runs omitted so the RQ1 floor (tied to the conditioned
#     base) is skipped -- only the reference-dependent interactions matter for this pass. ---
ANALYSIS_DIR2="$ANALYSIS_DIR/ref_ce_finetune"
mkdir -p "$ANALYSIS_DIR2"
echo "Second reference: ce_finetune -> $ANALYSIS_DIR2 (floor skipped)"
python -u scripts/visualize_validation_geometry.py \
    --validation-root "$VAL_ROOT" \
    --geometry-features "$FEATURES_CSV" \
    --output-dir "$ANALYSIS_DIR2" \
    --runs "${RUNS[@]}" \
    --reference-run "ce_finetune" \
    --bootstrap-n 10000 \
    --alpha 0.05 \
    --rng-seed 0

echo "Done. End: $(date)"
