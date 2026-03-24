#!/bin/bash
#SBATCH --job-name=kernelltl-lr-sweep
#SBATCH --output=logs/kernelltl_lr_sweep_%j.out
#SBATCH --error=logs/kernelltl_lr_sweep_%j.err
#SBATCH --time=72:00:00          # 5 runs * ~14h each worst-case; early stopping reduces this significantly
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G

# ============================================================================
# Snellius Learning Rate Sweep Script for KernelLTL (Stage 1 CE Training)
# ============================================================================
#
# Runs a sequential sweep over LR_VALUES, each run starting from the same
# base model checkpoint. The best model across all runs (lowest
# eval_semantic_distance) is identified at the end and copied to
# HOME_OUTPUT_DIR/sweep/best_model/.
#
# Early stopping is used to avoid wasting compute on runs that plateau early.
# The metric_for_best_model is eval_semantic_distance (lower is better), so
# each run's saved final_model is already its best checkpoint.
#
# Usage:
#   sbatch snellius_ce_hyperparam.sh
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION
# ============================================================================

PROJECT_DIR="$HOME/KernelLTL"
VENV_DIR="$PROJECT_DIR/venv"

# Shared artifacts
KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"

# Output directories
HOME_OUTPUT_DIR="$PROJECT_DIR/artifacts/models/CE/sweep"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_BASE="$SCRATCH_BASE/models/CE/sweep"

# Dataset for stage1 (same for all runs)
TRAIN_DIR="$PROJECT_DIR/artifacts/datasets/stage1/train"
EVAL_DIR="$PROJECT_DIR/artifacts/datasets/stage1/eval"

# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Five learning rates on a log-uniform scale spanning two decades.
# Adjust these values to focus the sweep on a narrower range once you have
# a rough idea of where the optimum lies.
LR_VALUES=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3")

# Max epochs per run. Early stopping will cut this short in practice.
# Keep this high enough that well-behaved runs have time to converge.
EPOCHS=100

# Early stopping patience (in eval steps, not epochs).
# Given the oscillatory semantic_distance behaviour observed in stage1 run3,
# a patience of 15 is enough to outlast ~3 full oscillation cycles without
# stopping prematurely on a spike. Lower this if compute budget is tight.
EARLY_STOPPING_PATIENCE=10

# Minimum improvement in eval_semantic_distance required to reset patience
# counter. 0.0 means any strict improvement counts (recommended here since
# improvements are small near the plateau).
EARLY_STOPPING_THRESHOLD=0.0

# Fixed training settings (shared across all runs for a fair comparison)
DEFAULT_BATCH_SIZE=64
DEFAULT_WARMUP_STEPS=500
MIXED_PRECISION="--bf16"
EVAL_BATCH_SIZE="81920"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL Learning Rate Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "LR values: ${LR_VALUES[*]}"
echo "Max epochs per run: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "=============================================="

mkdir -p "$PROJECT_DIR/logs"

# Load modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$PROJECT_DIR"

# Setup virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# ============================================================================
# COPY DATASETS TO SCRATCH
# ============================================================================
# Copy once and reuse across all LR runs to avoid redundant I/O.

SCRATCH_TRAIN_DIR="$SCRATCH_BASE/datasets/stage1/train"
SCRATCH_EVAL_DIR="$SCRATCH_BASE/datasets/stage1/eval"

echo ""
echo "Copying datasets to scratch..."
mkdir -p "$SCRATCH_TRAIN_DIR" "$SCRATCH_EVAL_DIR"
cp -r "$TRAIN_DIR/." "$SCRATCH_TRAIN_DIR/"
cp -r "$EVAL_DIR/." "$SCRATCH_EVAL_DIR/"

# ============================================================================
# RUN SWEEP
# ============================================================================

for LR in "${LR_VALUES[@]}"; do

    # Use the LR string directly as a tag (hyphens are safe in Linux paths)
    LR_TAG="lr_${LR}"

    STAGE_OUTPUT_DIR="$SCRATCH_OUTPUT_BASE/$LR_TAG"
    STAGE_MODEL_SAVE_DIR="$STAGE_OUTPUT_DIR/final_model"
    STAGE_HOME_OUTPUT_DIR="$HOME_OUTPUT_DIR/$LR_TAG"
    STAGE_HOME_MODEL_SAVE_DIR="$STAGE_HOME_OUTPUT_DIR/final_model"

    # eval_steps / save_steps as a fraction of 1 epoch (1 eval per epoch)
    STEP_INTERVAL=$(echo "scale=6; 1/$EPOCHS" | bc -l)

    echo ""
    echo "=============================================="
    echo "Starting sweep run: LR = $LR"
    echo "  Output (scratch): $STAGE_OUTPUT_DIR"
    echo "  Output (home):    $STAGE_HOME_OUTPUT_DIR"
    echo "=============================================="

    mkdir -p "$STAGE_OUTPUT_DIR" "$STAGE_MODEL_SAVE_DIR"
    mkdir -p "$STAGE_HOME_OUTPUT_DIR" "$STAGE_HOME_MODEL_SAVE_DIR"

    CMD_ARGS=(
        "--kernel-dir"                   "$KERNEL_DIR"
        "--tokenizer-dir"                "$TOKENIZER_DIR"
        "--train-dataset-dir"            "$SCRATCH_TRAIN_DIR"
        "--eval-dataset-dir"             "$SCRATCH_EVAL_DIR"
        "--output-dir"                   "$STAGE_OUTPUT_DIR"
        "--model-save-dir"               "$STAGE_MODEL_SAVE_DIR"
        "--num-train-epochs"             "$EPOCHS"
        "--learning-rate"                "$LR"
        "--per-device-train-batch-size"  "$DEFAULT_BATCH_SIZE"
        "--per-device-eval-batch-size"   "$DEFAULT_BATCH_SIZE"
        "--warmup-steps"                 "$DEFAULT_WARMUP_STEPS"
        "--logging-steps"                "$STEP_INTERVAL"
        "--eval-steps"                   "$STEP_INTERVAL"
        "--save-steps"                   "$STEP_INTERVAL"
        "--dataloader-num-workers"       "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
        "--dataloader-pin-memory"
        "--metric-for-best-model"        "eval_semantic_distance"
        "--greater-is-better"            "false"
        "--early-stopping-patience"      "$EARLY_STOPPING_PATIENCE"
        "--early-stopping-threshold"     "$EARLY_STOPPING_THRESHOLD"
        "--semantic-eval-batch-size"     "$EVAL_BATCH_SIZE"
        "--stage-name"                   "$LR_TAG"   # tags all JSONL records for this run
        $MIXED_PRECISION
    )

    RUN_START=$(date +%s)

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
            scripts/curriculum_train.py \
            "${CMD_ARGS[@]}"
    else
        python scripts/curriculum_train.py \
            "${CMD_ARGS[@]}"
    fi

    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))
    echo "LR=$LR completed in $((RUN_DURATION / 3600))h $(((RUN_DURATION % 3600) / 60))m $((RUN_DURATION % 60))s"

    # Persist logs and best model to home storage before moving to next run.
    # If the node crashes mid-sweep we still have the completed runs.
    echo "Syncing logs and model back to home..."
    mkdir -p "$STAGE_HOME_OUTPUT_DIR/logs"
    if [ -d "$STAGE_OUTPUT_DIR/logs" ]; then
        rsync -a --delete "$STAGE_OUTPUT_DIR/logs/" "$STAGE_HOME_OUTPUT_DIR/logs/"
    fi
    if [ -d "$STAGE_MODEL_SAVE_DIR" ]; then
        rsync -a --delete "$STAGE_MODEL_SAVE_DIR/" "$STAGE_HOME_MODEL_SAVE_DIR/"
    fi

done

# ============================================================================
# FIND BEST LR AND COPY BEST MODEL
# ============================================================================
# Parse each run's metrics_history.jsonl to find the lowest
# eval_semantic_distance recorded across all eval steps. Because
# load_best_model_at_end=True, the saved final_model for each run is already
# that run's best checkpoint, so we just need to identify which run won.

echo ""
echo "=============================================="
echo "Comparing runs to find best learning rate..."
echo "=============================================="

BEST_LR_TAG=$(python3 - <<PYEOF
import json
import os
import glob

home_sweep_base = "$HOME_OUTPUT_DIR"
lr_tags = [d for d in os.listdir(home_sweep_base)
           if d.startswith("lr_") and os.path.isdir(os.path.join(home_sweep_base, d))]

results = {}

for lr_tag in sorted(lr_tags):
    metrics_path = os.path.join(home_sweep_base, lr_tag, "logs", "metrics_history.jsonl")
    if not os.path.exists(metrics_path):
        print(f"[WARNING] No metrics file found for {lr_tag}, skipping.")
        continue

    best_dist = float("inf")
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            dist = record.get("eval_semantic_distance")
            if dist is not None and float(dist) < best_dist:
                best_dist = float(dist)

    results[lr_tag] = best_dist
    print(f"  {lr_tag:20s}  best eval_semantic_distance = {best_dist:.6f}")

if not results:
    print("No results found — cannot determine best LR.")
    raise SystemExit(1)

best_tag = min(results, key=results.get)
print(f"\n  --> Best run: {best_tag}  (eval_semantic_distance = {results[best_tag]:.6f})")
print(best_tag)   # last line is consumed by the shell assignment below
PYEOF
)

# The Python script prints the winning lr_tag as its very last line.
BEST_LR_TAG=$(echo "$BEST_LR_TAG" | tail -n 1)

echo ""
echo "Best LR tag: $BEST_LR_TAG"

BEST_MODEL_SRC="$HOME_OUTPUT_DIR/$BEST_LR_TAG/final_model"
BEST_MODEL_DST="$HOME_OUTPUT_DIR/best_model"

echo "Copying best model to: $BEST_MODEL_DST"
mkdir -p "$BEST_MODEL_DST"
rsync -a --delete "$BEST_MODEL_SRC/" "$BEST_MODEL_DST/"

# Write a small provenance file so we know which run won
cat > "$BEST_MODEL_DST/sweep_provenance.json" <<PROVEOF
{
  "best_lr_tag": "$BEST_LR_TAG",
  "lr_values_swept": ["${LR_VALUES[*]}"],
  "metric": "eval_semantic_distance",
  "source_model": "$BEST_MODEL_SRC",
  "slurm_job_id": "$SLURM_JOB_ID"
}
PROVEOF

# ============================================================================
# CLEANUP
# ============================================================================
echo ""
echo "Cleaning scratch-local..."
rm -rf "$SCRATCH_BASE"

echo ""
echo "=============================================="
echo "LR sweep completed!"
echo "End time: $(date)"
echo "Best model: $BEST_MODEL_DST  (run: $BEST_LR_TAG)"
echo "=============================================="
