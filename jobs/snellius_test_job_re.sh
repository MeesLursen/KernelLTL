#!/bin/bash  
#SBATCH --job-name=kernelltl_test_job_re
#SBATCH --output=logs/kernelltl_test_job_re_gae_lambda_1_wu_4_crlr_5e-4_value_test_%j.out
#SBATCH --error=logs/kernelltl_test_job_re_gae_lambda_1_wu_4_crlr_5e-4_value_test_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G

# ============================================================================
# Snellius Multi-Stage Curriculum Training Script for KernelLTL
# ============================================================================
#
# This script runs the specified STAGE_CONFIGS as a test job.
#
# Usage:
#   1. Configure the STAGE_CONFIGS array below with your stage parameters
#   2. Submit with: sbatch snellius_test_job.sh
#   
# ============================================================================

set -e  # Exit on error

# ============================================================================
# USER CONFIGURATION
# ============================================================================

PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

# Shared artifacts
KERNEL_DIR="$HOME_DIR/artifacts/kernel"
TOKENIZER_DIR="$HOME_DIR/artifacts/tokenizer"

# Shared model root with independent RE/CE branches
BASE_MODELS_ROOT="$PROJECT_DIR/artifacts/models"
BASE_RE_OUTPUT_DIR="$BASE_MODELS_ROOT/RE/test"
BASE_CE_OUTPUT_DIR="$BASE_MODELS_ROOT/CE"

# Training defaults (can be overridden per stage)
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_TRAIN_BATCH_SIZE=256
DEFAULT_EVAL_BATCH_SIZE=96
DEFAULT_WARMUP_RATIO=1.0

# Mixed precision
MIXED_PRECISION="--bf16"

# Semantic Eval Callback Settings
DISABLE_SEMANTIC_CALLBACK=1
DISABLE_TRAIN_END_SEMANTIC_EVAL=0
EVAL_BATCH_SIZE="256000"

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE=15
EARLY_STOPPING_THRESHOLD=0.0

# RL trainer mode: gae or rb
DEFAULT_RL_TRAINER="rb"

# Train Dataset Satisfaction mmap Loading
TRAIN_SATS_MMAP=0

# Shared RL controls
DEFAULT_RL_CLIP="1.0"

# RB-specific controls
DEFAULT_RB_BASELINE_MOMENTUM="0.9"

# GAE-specific controls
DEFAULT_GAE_GAMMA="1.0"
DEFAULT_GAE_LAMBDA="1.0"
DEFAULT_CRITIC_LR="5e-6"
DEFAULT_CRITIC_HIDDEN_DIM="256"
DEFAULT_CRITIC_WEIGHT_DECAY="0.0"

# ============================================================================
# STAGE CONFIGURATION
# ============================================================================
# Define your curriculum stages here
# Format: "STAGE_NAME|TRAIN_DIR|EVAL_DIR|EPOCHS|LEARNING_RATE|RL_TRAINER|CRITIC_LR"

# ============================================================================

STAGE_CONFIGS=(
    "critic_lr_5e-3_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:5e-3"
    "critic_lr_1e-3_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:1e-3"
    "critic_lr_5e-4_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:5e-4"
    "critic_lr_1e-4_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:1e-4"
    "critic_lr_5e-5_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:5e-5"
    "critic_lr_1e-5_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:1e-5"
    "critic_lr_5e-6_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:5e-6"
    "critic_lr_1e-6_test_gae_lamdba_1:$PROJECT_DIR/artifacts/datasets/finetune/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:5:5e-8:gae:1e-6"
)   
    # "stage0:$PROJECT_DIR/artifacts/datasets/stage0/train:$PROJECT_DIR/artifacts/datasets/stage0/eval:100:1e-4:rb"
    # 
    # "stage2:$PROJECT_DIR/artifacts/datasets/stage2/train:$PROJECT_DIR/artifacts//datasets/stage2/eval:100:5e-5:gae"
    # "stage3:$PROJECT_DIR/artifacts/datasets/stage3/train:$PROJECT_DIR/artifacts/datasets/stage3/eval:100:1e-5:gae"
    # "stage4:$PROJECT_DIR/artifacts/datasets/stage4/train:$PROJECT_DIR/artifacts/datasets/stage4/eval:100:5e-6:rb"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "KernelLTL Multi-Stage Curriculum Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p "$HOME_DIR/logs"

# Load modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"

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

export PYTHONPATH="$HOME_DIR:$PYTHONPATH"


NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

if [ -z "$TMPDIR" ]; then
    echo "TMPDIR is not set. scratch-node is required for this job."
    exit 1
fi

case "$TMPDIR" in
    /scratch-node/*)
        ;;
    *)
        echo "TMPDIR is not on /scratch-node ($TMPDIR). This job requires node-local scratch."
        exit 1
        ;;
esac

SCRATCH_ROOT="$TMPDIR/KernelLTL"
mkdir -p "$SCRATCH_ROOT"
SCRATCH_RE_OUTPUT_ROOT="$SCRATCH_ROOT/models/RE/test"
mkdir -p "$SCRATCH_RE_OUTPUT_ROOT"

# ============================================================================
# RUN CURRICULUM STAGES
# ============================================================================

PREV_MODEL_DIR="$BASE_CE_OUTPUT_DIR/run2/stage4/final_model"
PREV_TRAINING_ARGS_DIR="$BASE_CE_OUTPUT_DIR/run2/stage4/final_model"
DEBUG_OPTION=""

for i in "${!STAGE_CONFIGS[@]}"; do
    # Parse stage configuration
    IFS=':' read -r STAGE_NAME TRAIN_DIR EVAL_DIR EPOCHS LR STAGE_RL_TRAINER CRITIC_LR <<< "${STAGE_CONFIGS[$i]}"
    
    # Use defaults if not specified
    LR=${LR:-$DEFAULT_LEARNING_RATE}
    STAGE_RL_TRAINER=${STAGE_RL_TRAINER:-$DEFAULT_RL_TRAINER}
    STEP_INTERVAL=$(echo "scale=6; 1/$EPOCHS" | bc -l)
    PRETRAIN_STEP_INTERVAL=$(echo "scale=6; 5/$EPOCHS" | bc -l)

    STAGE_OUTPUT_DIR="$BASE_RE_OUTPUT_DIR/$STAGE_NAME"
    STAGE_MODEL_SAVE_DIR="$STAGE_OUTPUT_DIR/final_model"
    SCRATCH_STAGE_OUTPUT_DIR="$SCRATCH_RE_OUTPUT_ROOT/$STAGE_NAME"
    SCRATCH_STAGE_MODEL_SAVE_DIR="$SCRATCH_STAGE_OUTPUT_DIR/final_model"
    CE_REFERENCE_MODEL_DIR="$PREV_MODEL_DIR"
    
    echo ""
    echo "=============================================="
    echo "Starting $STAGE_NAME (Run $((i+1)) of ${#STAGE_CONFIGS[@]})"
    echo "  Train dataset: $TRAIN_DIR"
    echo "  Eval dataset: $EVAL_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Learning rate: $LR"
    echo "  Batch size: $BATCH_SIZE"
    echo "  RL trainer: $STAGE_RL_TRAINER"
    echo "  CE reference model: $CE_REFERENCE_MODEL_DIR"
    echo "  RE output (scratch): $SCRATCH_STAGE_OUTPUT_DIR"
    echo "  RE output (project): $STAGE_OUTPUT_DIR"
    echo "=============================================="
    
    mkdir -p "$STAGE_OUTPUT_DIR"
    mkdir -p "$STAGE_MODEL_SAVE_DIR"
    mkdir -p "$SCRATCH_STAGE_OUTPUT_DIR"
    mkdir -p "$SCRATCH_STAGE_MODEL_SAVE_DIR"

    if [ -d "$STAGE_OUTPUT_DIR" ]; then
        echo "Syncing existing stage output from project to scratch (resume support)..."
        rsync -a "$STAGE_OUTPUT_DIR/" "$SCRATCH_STAGE_OUTPUT_DIR/"
    fi

    if [ ! -d "$CE_REFERENCE_MODEL_DIR" ]; then
        echo "Missing CE reference model directory for $STAGE_NAME: $CE_REFERENCE_MODEL_DIR".
        exit 1
    fi

    SCRATCH_TRAIN_DIR="$SCRATCH_ROOT/datasets/$STAGE_NAME/train"
    SCRATCH_EVAL_DIR="$SCRATCH_ROOT/datasets/$STAGE_NAME/eval"

    echo ""
    echo "=============================================="
    echo "Copying train+eval datasets from home to scratch..."
    echo "  From: $TRAIN_DIR" and "$EVAL_DIR"
    echo "  To:   $SCRATCH_TRAIN_DIR" and "$SCRATCH_EVAL_DIR"
    echo "=============================================="

    mkdir -p "$SCRATCH_TRAIN_DIR"
    mkdir -p "$SCRATCH_EVAL_DIR"

    cp -r "$TRAIN_DIR/." "$SCRATCH_TRAIN_DIR/"
    cp -r "$EVAL_DIR/." "$SCRATCH_EVAL_DIR/"

    # Build command arguments
    CMD_ARGS=(
        "--kernel-dir" "$KERNEL_DIR"
        "--tokenizer-dir" "$TOKENIZER_DIR"
        "--train-dataset-dir" "$SCRATCH_TRAIN_DIR"
        "--eval-dataset-dir" "$SCRATCH_EVAL_DIR"
        "--output-dir" "$SCRATCH_STAGE_OUTPUT_DIR"
        "--model-save-dir" "$SCRATCH_STAGE_MODEL_SAVE_DIR"
        "--num-train-epochs" "$EPOCHS"
        "--learning-rate" "$LR"
        "--per-device-train-batch-size" "$DEFAULT_TRAIN_BATCH_SIZE"
        "--per-device-eval-batch-size" "$DEFAULT_EVAL_BATCH_SIZE"
        "--critic-pretraining-ratio" "$PRETRAIN_STEP_INTERVAL"
        "--logging-steps" "$STEP_INTERVAL"
        "--eval-steps" "$STEP_INTERVAL"
        "--save-steps" "$STEP_INTERVAL"
        "--dataloader-num-workers" "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
        "--dataloader-pin-memory"
        $MIXED_PRECISION
        "--semantic-eval-batch-size" "$EVAL_BATCH_SIZE"
        "--rl-trainer" "$STAGE_RL_TRAINER"
        "--reinforce-reward-clip" "$DEFAULT_RL_CLIP"
        "--ce-reference-model-dir" "$CE_REFERENCE_MODEL_DIR"

    )

    if [ "$STAGE_RL_TRAINER" = "rb" ]; then
        CMD_ARGS+=("--reinforce-baseline-momentum" "$DEFAULT_RB_BASELINE_MOMENTUM")
    elif [ "$STAGE_RL_TRAINER" = "gae" ]; then
        CMD_ARGS+=(
            "--gae-gamma" "$DEFAULT_GAE_GAMMA"
            "--gae-lambda" "$DEFAULT_GAE_LAMBDA"
            "--critic-lr" "$CRITIC_LR"
            "--critic-hidden-dim" "$DEFAULT_CRITIC_HIDDEN_DIM"
            "--critic-weight-decay" "$DEFAULT_CRITIC_WEIGHT_DECAY"
        )
    else
        echo "Unknown RL trainer '$STAGE_RL_TRAINER' for $STAGE_NAME. Expected 'rb' or 'gae'."
        exit 1
    fi

    # Set debugging options
    if [ -n "$DEBUG_OPTION" ]; then
        echo "  Running with debug option: $DEBUG_OPTION"
        CMD_ARGS+=("--debug" "$DEBUG_OPTION")
    fi

    # Load previous stage model (if not first stage)
    if [ -n "$PREV_MODEL_DIR" ] && [ -d "$PREV_MODEL_DIR" ]; then
        echo "  Loading model from previous stage: $PREV_MODEL_DIR"
        CMD_ARGS+=("--model-load-dir" "$PREV_MODEL_DIR")
    fi
    
    # Load previous stage training args (if not first stage)
    if [ -n "$PREV_TRAINING_ARGS_DIR" ] && [ -d "$PREV_TRAINING_ARGS_DIR" ]; then
        CMD_ARGS+=("--training-args-load-dir" "$PREV_TRAINING_ARGS_DIR")
    fi
    
    # Train sats mmap storage
    if [ "$TRAIN_SATS_MMAP" = "1" ]; then
        echo "  Training dataset satisfactions will be loaded as mmap."
        CMD_ARGS+=(
            "--satisfactions-mmap"
        )
    fi

    # Disable Eval Controls
    if [ "$DISABLE_SEMANTIC_CALLBACK" = "1" ]; then
        echo "  Disabled Semantic Evaluation Callback."
        CMD_ARGS+=(
            "--disable-semantic-callback"
        )
    fi

    if [ "$DISABLE_TRAIN_END_SEMANTIC_EVAL" = "1" ]; then
        echo "  Disabled final model evaluation in the SemanticEvaluationCallback."
        CMD_ARGS+=(
            "--disable-train-end-semantic-eval"
        )
    fi

    # Run training
    STAGE_START=$(date +%s)
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
            scripts/curriculum_train_reinforce.py \
            "${CMD_ARGS[@]}"
    else
        python scripts/curriculum_train_reinforce.py \
            "${CMD_ARGS[@]}"
    fi
    
    STAGE_END=$(date +%s)
    STAGE_DURATION=$((STAGE_END - STAGE_START))

    echo "Syncing stage outputs from scratch to project..."
    mkdir -p "$STAGE_OUTPUT_DIR"
    rsync -a --delete "$SCRATCH_STAGE_OUTPUT_DIR/" "$STAGE_OUTPUT_DIR/"
    
    echo "$STAGE_NAME completed in $((STAGE_DURATION / 3600))h $(((STAGE_DURATION % 3600) / 60))m $((STAGE_DURATION % 60))s"
done

echo ""
echo "=============================================="
echo "All curriculum stages completed!"
echo "End time: $(date)"
echo "Final model (project): $STAGE_MODEL_SAVE_DIR"
echo "=============================================="
