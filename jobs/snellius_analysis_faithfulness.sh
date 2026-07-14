#!/bin/bash
#SBATCH --job-name=kernelltl_analysis_faithfulness
#SBATCH --output=logs/kernelltl_analysis_faithfulness_%j.out
#SBATCH --error=logs/kernelltl_analysis_faithfulness_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G

set -euo pipefail

# ============================================================================
# RQ1a representation-faithfulness pipeline (study I1) + the full geometry +
# faithfulness analysis (mediation, faithfulness-conditioned interaction).
#   Step 1a (heavy): compute_geometry_features.py     -> geometry_features.csv
#   Step 1b (heavy): compute_faithfulness_features.py -> faithfulness_features.csv
#                    (streams the ~10 GB satisfactions tensor; landmark-partner matmul)
#   Step 2  (light): visualize_validation_geometry.py with BOTH feature CSVs
#                    -> stats/extra/geometry_* + geometry_faith_* + figures.
# CPU-only matmuls; mmap keeps peak memory modest.
# ============================================================================

HOME_DIR="$HOME/KernelLTL"
PROJECT_DIR="/projects/prjs2029/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
VAL_DATASET_DIR="$PROJECT_DIR/artifacts/datasets/validation"   # embeddings.pt + satisfactions.pt
VAL_ROOT="$PROJECT_DIR/artifacts/validation"
ANALYSIS_DIR="$VAL_ROOT/_analysis"
GEO_CSV="$ANALYSIS_DIR/geometry_features.csv"
FAITH_CSV="$ANALYSIS_DIR/faithfulness_features.csv"

RUNS=(ce_base ce_finetune rb_momentum_09 gae_lambda_09 gae_lambda_1)
REFERENCE_RUN="ce_base"

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

# --- Step 1a: geometry features (skip if cached) ----------------------------
if [ -f "$GEO_CSV" ]; then
    echo "geometry_features.csv exists; skipping."
else
    echo "Computing geometry features ..."
    python -u scripts/compute_geometry_features.py \
        --validation-dataset-dir "$VAL_DATASET_DIR" \
        --kernel-dir "$KERNEL_DIR" \
        --output "$GEO_CSV" \
        --chunk-size 500
fi

# --- Step 1b: faithfulness features (skip if cached) ------------------------
if [ -f "$FAITH_CSV" ]; then
    echo "faithfulness_features.csv exists; skipping."
else
    echo "Computing faithfulness features ..."
    python -u scripts/compute_faithfulness_features.py \
        --validation-dataset-dir "$VAL_DATASET_DIR" \
        --output "$FAITH_CSV" \
        --n-landmarks 256 \
        --chunk-size 500 \
        --seed 0
fi

# --- Step 2: geometry + faithfulness analysis + figures ---------------------
echo "Running geometry + faithfulness analysis ..."
python -u scripts/visualize_validation_geometry.py \
    --validation-root "$VAL_ROOT" \
    --geometry-features "$GEO_CSV" \
    --faithfulness-features "$FAITH_CSV" \
    --output-dir "$ANALYSIS_DIR" \
    --runs "${RUNS[@]}" \
    --reference-run "$REFERENCE_RUN" \
    --bootstrap-n 10000 \
    --alpha 0.05 \
    --rng-seed 0

echo "Done. End: $(date)"
