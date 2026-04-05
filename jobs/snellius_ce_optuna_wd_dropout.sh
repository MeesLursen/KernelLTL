#!/bin/bash
#SBATCH --job-name=kernelltl-ce-hpo-optuna
#SBATCH --output=logs/kernelltl_ce_hpo_optuna_%j.out
#SBATCH --error=logs/kernelltl_ce_hpo_optuna_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

# ============================================================================
# Snellius CE Hyperparameter Search (HF Trainer.hyperparameter_search + Optuna)
# ============================================================================
#
# Runs scripts/hpo_optuna_wd_dropout.py to optimize dropout + weight decay.
# The script performs:
#   1) Optuna trial search
#   2) Final training pass with best hyperparameters
#   3) Save final_model and hpo_best_run.json
#
# Usage:
#   sbatch jobs/snellius_ce_hpo_optuna.sh
#
# ============================================================================

set -e

# ============================================================================
# USER CONFIGURATION
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Shared artifacts
KERNEL_DIR="$HOME_DIR/artifacts/kernel"
TOKENIZER_DIR="$HOME_DIR/artifacts/tokenizer"

# Stage/dataset selection
TRAIN_DIR="$PROJECT_DIR/artifacts/datasets/stage1/train"
EVAL_DIR="$PROJECT_DIR/artifacts/datasets/stage1/eval"

# Output locations
HOME_OUTPUT_DIR="$HOME_DIR/artifacts/models/CE/hpo_optuna"
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/models/CE/hpo_optuna"

# Training defaults
EPOCHS=100
LEARNING_RATE=5e-4
BATCH_SIZE=256
WARMUP_RATIO=0.05
MIXED_PRECISION="--bf16"
EVAL_BATCH_SIZE="81920"
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_THRESHOLD=0.0

# HPO controls
N_TRIALS=20
OBJECTIVE_METRIC="eval_semantic_distance"
HPO_DIRECTION="minimize"

# Search space
DROPOUT_MIN=0.0
DROPOUT_MAX=0.2
DROPOUT_STEP=0.05
WEIGHT_DECAY_MIN=0.005
WEIGHT_DECAY_MAX=0.02
WEIGHT_DECAY_LOG="--weight-decay-log"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL CE HPO (Optuna via HF Trainer)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Trials: $N_TRIALS"
echo "Search: dropout in [$DROPOUT_MIN, $DROPOUT_MAX] step $DROPOUT_STEP"
echo "Search: weight_decay in [$WEIGHT_DECAY_MIN, $WEIGHT_DECAY_MAX]"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"
mkdir -p "$HOME_OUTPUT_DIR"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

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

export PYTHONPATH="$HOME_DIR:$PYTHONPATH"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Visible GPUs on node: $NUM_GPUS"

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU visible on this node."
    exit 1
fi

mkdir -p "$SCRATCH_OUTPUT_DIR"

# ============================================================================
# RUN HPO
# ============================================================================

STEP_INTERVAL=$(echo "scale=6; 1/$EPOCHS" | bc -l)
STAGE_TAG="hpo_optuna_stage1"

CMD_ARGS=(
    "--kernel-dir" "$KERNEL_DIR"
    "--tokenizer-dir" "$TOKENIZER_DIR"
    "--train-dataset-dir" "$TRAIN_DIR"
    "--eval-dataset-dir" "$EVAL_DIR"
    "--output-dir" "$SCRATCH_OUTPUT_DIR"
    "--model-save-dir" "$SCRATCH_OUTPUT_DIR/final_model"
    "--stage-name" "$STAGE_TAG"
    "--num-train-epochs" "$EPOCHS"
    "--learning-rate" "$LEARNING_RATE"
    "--per-device-train-batch-size" "$BATCH_SIZE"
    "--per-device-eval-batch-size" "$BATCH_SIZE"
    "--warmup-ratio" "$WARMUP_RATIO"
    "--logging-steps" "$STEP_INTERVAL"
    "--eval-steps" "$STEP_INTERVAL"
    "--save-steps" "$STEP_INTERVAL"
    "--dataloader-num-workers" "$SLURM_CPUS_PER_TASK"
    "--dataloader-pin-memory"
    "--semantic-eval-batch-size" "$EVAL_BATCH_SIZE"
    "--metric-for-best-model" "$OBJECTIVE_METRIC"
    "--greater-is-better" "false"
    "--early-stopping-patience" "$EARLY_STOPPING_PATIENCE"
    "--early-stopping-threshold" "$EARLY_STOPPING_THRESHOLD"
    "--n-trials" "$N_TRIALS"
    "--objective-metric" "$OBJECTIVE_METRIC"
    "--hpo-direction" "$HPO_DIRECTION"
    "--dropout-min" "$DROPOUT_MIN"
    "--dropout-max" "$DROPOUT_MAX"
    "--dropout-step" "$DROPOUT_STEP"
    "--weight-decay-min" "$WEIGHT_DECAY_MIN"
    "--weight-decay-max" "$WEIGHT_DECAY_MAX"
    $WEIGHT_DECAY_LOG
    $MIXED_PRECISION
)

RUN_START=$(date +%s)

# Important: run as a single process (no torchrun). The HPO script enforces this.
python scripts/hpo_optuna_wd_dropout.py "${CMD_ARGS[@]}"

RUN_END=$(date +%s)
RUN_DURATION=$((RUN_END - RUN_START))
echo "HPO completed in $((RUN_DURATION / 3600))h $(((RUN_DURATION % 3600) / 60))m $((RUN_DURATION % 60))s"

# ============================================================================
# SYNC RESULTS TO HOME
# ============================================================================

echo "Syncing HPO artifacts to home..."
mkdir -p "$HOME_OUTPUT_DIR"
cp -r "$SCRATCH_OUTPUT_DIR/" "$HOME_OUTPUT_DIR/"

# ============================================================================
# CLEANUP
# ============================================================================

echo "Cleaning scratch-local..."
rm -rf "$SCRATCH_BASE"

echo ""
echo "=============================================="
echo "CE HPO run completed!"
echo "End time: $(date)"
echo "Saved artifacts: $HOME_OUTPUT_DIR"
echo "=============================================="
