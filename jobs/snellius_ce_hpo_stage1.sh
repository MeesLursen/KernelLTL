#!/bin/bash
#SBATCH --job-name=kernelltl-ce-hpo-stage1
#SBATCH --output=logs/kernelltl_ce_hpo_stage1_%j.out
#SBATCH --error=logs/kernelltl_ce_hpo_stage1_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G

set -euo pipefail

# ============================================================================
# Staged CE hyperparameter selection on stage 1 (depth-graded datasets + regenerated kernel).
#
# Mirrors the thesis protocol (main.tex, "Prior to training..."):
#   Phase A: LR grid {5e-6, 1e-5, 5e-5, 1e-4, 5e-4} at dropout=0.1, wd=0.01
#            (4-GPU DDP, full-fidelity runs, early stopping).
#   Phase B: Optuna (20 sequential trials) over dropout in [0, 0.2] (step .05)
#            and weight_decay in [0.005, 0.02], at the Phase-A best LR. Each
#            trial trains with DDP across all 4 GPUs (same effective batch as
#            Phase A). The standard (dropout, wd) is enqueued as the study's
#            first trial, so the Phase-C trigger compares Optuna's best
#            against the baseline under the exact same trial protocol.
#   Phase C: only if Optuna's best (dropout, wd) beat the baseline reference
#            AND differ from the standard values: rerun the LR grid with the
#            new dropout/wd. Final pick = global best across A and C.
#
# Resumable: completed runs are skipped (a run is complete once its
# logs/metrics_history.jsonl has been synced to HPO_ROOT; Optuna is complete
# once hpo_best_run.json exists). Resubmit on wall-clock timeout.
#
# Phase B DDP note: HF's optuna backend runs the study on rank 0 and
# broadcasts each trial's parameters to the other ranks, so a trial spans
# all GPUs. The driver forces NopPruner (a pruned trial would abort rank 0
# mid-run while other ranks keep training, deadlocking the process group)
# and suggests every searched parameter inside hp_space so the broadcast
# carries the full trial.
# ============================================================================

# ---------------------------- CONFIGURATION --------------------------------
PROJECT_DIR="/projects/prjs2029/KernelLTL"
HOME_DIR="$HOME/KernelLTL"
VENV_DIR="$HOME_DIR/venv"

KERNEL_DIR="$PROJECT_DIR/artifacts/kernel"
TOKENIZER_DIR="$PROJECT_DIR/artifacts/tokenizer"
TRAIN_DIR="$PROJECT_DIR/artifacts/datasets/stage1/train"
EVAL_DIR="$PROJECT_DIR/artifacts/datasets/stage1/eval"

HPO_ROOT="$PROJECT_DIR/artifacts/models/CE/hpo_v2"     # persistent results
SCRATCH_BASE="/scratch-local/$USER/KernelLTL"
SCRATCH_ROOT="$SCRATCH_BASE/models/CE/hpo_v2"

# Selection objective. Thesis + existing infra select on eval_semantic_distance;
# switch to "eval_loss" only if you deliberately change the selection criterion.
OBJECTIVE_METRIC="eval_loss"

# Standard settings (thesis): the Phase-A regularisation, and Phase-C trigger reference.
STD_DROPOUT=0.1
STD_WD=0.01

# Phase A / C: LR grid at full fidelity (thesis values).
LR_VALUES=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3")
FULL_EPOCHS=100
FULL_PATIENCE=10
FULL_BATCH_SIZE=256            # per device, 4 GPUs -> effective 1024
WARMUP_RATIO=$(echo "scale=6; 1/$FULL_EPOCHS" | bc -l)

# Phase B: Optuna over dropout/weight decay (thesis ranges), DDP per trial.
N_TRIALS=20
DROPOUT_MIN=0.0
DROPOUT_MAX=0.2
DROPOUT_STEP=0.05
WD_MIN=0.005
WD_MAX=0.02
HPO_EPOCHS=100                  # trial budget; early stopping cuts most trials short
HPO_PATIENCE=5
HPO_BATCH_SIZE=256             # per device; x4 GPUs -> effective 1024, same as Phase A

# Phase-C trigger tolerances: "different from standard" means dropout not equal
# to STD_DROPOUT (grid-stepped, so exact compare) or wd relatively off by > this.
WD_REL_TOL=0.10

MIXED_PRECISION="--bf16"
EVAL_BATCH_SIZE="256000"       # trace-eval batch inside the semantic callback

# ---------------------------- ENVIRONMENT ----------------------------------
echo "=============================================="
echo "KernelLTL staged CE HPO (stage 1, v2 data)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURMD_NODENAME:-N/A}   Start: $(date)"
echo "Objective: $OBJECTIVE_METRIC"
echo "=============================================="

mkdir -p "$HOME_DIR/logs" "$HPO_ROOT"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0

cd "$HOME_DIR"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $NUM_GPUS"

# Stage datasets to scratch once (all phases share them).
SCRATCH_TRAIN_DIR="$SCRATCH_BASE/datasets/stage1/train"
SCRATCH_EVAL_DIR="$SCRATCH_BASE/datasets/stage1/eval"
mkdir -p "$SCRATCH_TRAIN_DIR" "$SCRATCH_EVAL_DIR"
cp -r "$TRAIN_DIR/." "$SCRATCH_TRAIN_DIR/"
cp -r "$EVAL_DIR/." "$SCRATCH_EVAL_DIR/"

# ---------------------------- HELPERS --------------------------------------

# One full-fidelity CE run (4-GPU DDP). Skipped if already synced to HPO_ROOT.
#   $1 tag   $2 learning rate   $3 dropout   $4 weight decay
run_full() {
    local TAG="$1" LR="$2" DROPOUT="$3" WD="$4"
    local OUT_SCRATCH="$SCRATCH_ROOT/$TAG"
    local OUT_HOME="$HPO_ROOT/$TAG"

    if [ -f "$OUT_HOME/logs/metrics_history.jsonl" ]; then
        echo "[skip] $TAG already complete."
        return 0
    fi

    echo ""
    echo "===== run_full $TAG: lr=$LR dropout=$DROPOUT wd=$WD ====="
    mkdir -p "$OUT_SCRATCH/final_model" "$OUT_HOME/logs" "$OUT_HOME/final_model"
    local STEP_INTERVAL
    STEP_INTERVAL=$(echo "scale=6; 1/$FULL_EPOCHS" | bc -l)

    local CMD_ARGS=(
        "--kernel-dir"                  "$KERNEL_DIR"
        "--tokenizer-dir"               "$TOKENIZER_DIR"
        "--train-dataset-dir"           "$SCRATCH_TRAIN_DIR"
        "--eval-dataset-dir"            "$SCRATCH_EVAL_DIR"
        "--output-dir"                  "$OUT_SCRATCH"
        "--model-save-dir"              "$OUT_SCRATCH/final_model"
        "--num-train-epochs"            "$FULL_EPOCHS"
        "--learning-rate"               "$LR"
        "--dropout"                     "$DROPOUT"
        "--weight-decay"                "$WD"
        "--per-device-train-batch-size" "$FULL_BATCH_SIZE"
        "--per-device-eval-batch-size"  "$FULL_BATCH_SIZE"
        "--warmup-ratio"                "$WARMUP_RATIO"
        "--logging-steps"               "$STEP_INTERVAL"
        "--eval-steps"                  "$STEP_INTERVAL"
        "--save-steps"                  "$STEP_INTERVAL"
        "--dataloader-num-workers"      "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
        "--dataloader-pin-memory"
        "--metric-for-best-model"       "$OBJECTIVE_METRIC"
        "--greater-is-better"           "false"
        "--early-stopping-patience"     "$FULL_PATIENCE"
        "--early-stopping-threshold"    "0.0"
        "--semantic-eval-batch-size"    "$EVAL_BATCH_SIZE"
        "--stage-name"                  "$TAG"
        $MIXED_PRECISION
    )

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" scripts/curriculum_train.py "${CMD_ARGS[@]}"
    else
        python scripts/curriculum_train.py "${CMD_ARGS[@]}"
    fi

    rsync -a --delete "$OUT_SCRATCH/logs/" "$OUT_HOME/logs/"
    rsync -a --delete "$OUT_SCRATCH/final_model/" "$OUT_HOME/final_model/"
}

# Best (minimum) objective value recorded in a run's metrics_history.jsonl.
#   $1 run dir under HPO_ROOT ; prints the value (or "inf")
best_of_run() {
    python3 - "$HPO_ROOT/$1/logs/metrics_history.jsonl" "$OBJECTIVE_METRIC" <<'PYEOF'
import json, sys
path, key = sys.argv[1], sys.argv[2]
best = float("inf")
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line).get(key)
            except json.JSONDecodeError:
                continue
            if v is not None and float(v) < best:
                best = float(v)
except FileNotFoundError:
    pass
print(best)
PYEOF
}

# Winner among a list of run tags: prints "tag value".
#   $@ tags
best_tag_of() {
    local BEST_TAG="" BEST_VAL="inf" TAG VAL
    for TAG in "$@"; do
        VAL=$(best_of_run "$TAG")
        echo "  $TAG: best $OBJECTIVE_METRIC = $VAL" >&2
        if python3 -c "import sys; sys.exit(0 if float('$VAL') < float('$BEST_VAL') else 1)"; then
            BEST_TAG="$TAG"; BEST_VAL="$VAL"
        fi
    done
    echo "$BEST_TAG $BEST_VAL"
}

# ============================================================================
# PHASE A: LR grid at standard dropout / weight decay
# ============================================================================
echo ""
echo "########## PHASE A: LR sweep @ dropout=$STD_DROPOUT wd=$STD_WD ##########"
A_TAGS=()
for LR in "${LR_VALUES[@]}"; do
    TAG="a_lr_${LR}"
    A_TAGS+=("$TAG")
    run_full "$TAG" "$LR" "$STD_DROPOUT" "$STD_WD"
done

read -r BEST_A_TAG BEST_A_VAL <<< "$(best_tag_of "${A_TAGS[@]}")"
BEST_LR="${BEST_A_TAG#a_lr_}"
echo ""
echo "Phase A winner: $BEST_A_TAG  ($OBJECTIVE_METRIC = $BEST_A_VAL)  ->  BEST_LR = $BEST_LR"

# ============================================================================
# PHASE B: Optuna over dropout / weight decay at BEST_LR (single GPU)
# ============================================================================
echo ""
echo "########## PHASE B: Optuna wd/dropout @ lr=$BEST_LR ##########"

# Phase B runs each trial as a DDP job across ALL GPUs (rank 0 drives the Optuna
# study and broadcasts each trial's parameters to the other ranks; the driver
# disables pruning, which would desync the ranks). Trials therefore use the same
# effective batch size as Phase A (per-device x NUM_GPUS).
# The STANDARD settings (dropout=STD_DROPOUT, wd=STD_WD) are enqueued as the
# study's first trial: the baseline then runs under exactly the trial protocol,
# making "did Optuna beat the standard settings" an apples-to-apples comparison.
# NOTE: the baseline consumes 1 of the N_TRIALS trials (so N_TRIALS-1 are TPE-sampled).
OPTUNA_TAG="b_optuna_wd_dropout"
OPTUNA_OUT_SCRATCH="$SCRATCH_ROOT/$OPTUNA_TAG"
OPTUNA_OUT_HOME="$HPO_ROOT/$OPTUNA_TAG"
BEST_RUN_JSON="$OPTUNA_OUT_HOME/logs/hpo_best_run.json"
STUDY_NAME="ce_hpo_stage1_wd_dropout"
STUDY_DB="$OPTUNA_OUT_SCRATCH/optuna_study.db"
STUDY_STORAGE="sqlite:///$STUDY_DB"

if [ -f "$BEST_RUN_JSON" ]; then
    echo "[skip] $OPTUNA_TAG already complete."
else
    mkdir -p "$OPTUNA_OUT_SCRATCH" "$OPTUNA_OUT_HOME/logs"
    STEP_INTERVAL=$(echo "scale=6; 1/$HPO_EPOCHS" | bc -l)

    # B.1 -- create the shared study and enqueue the standard-settings baseline trial.
    python3 - "$STUDY_STORAGE" "$STUDY_NAME" "$STD_DROPOUT" "$STD_WD" <<'PYEOF'
import sys
import optuna
storage, name, std_dropout, std_wd = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
study = optuna.create_study(study_name=name, storage=storage, direction="minimize", load_if_exists=True)
if len(study.trials) == 0:
    study.enqueue_trial({"dropout": std_dropout, "weight_decay": std_wd},
                        user_attrs={"baseline": True})
    print(f"[study] created '{name}', enqueued baseline trial (dropout={std_dropout}, wd={std_wd})")
else:
    print(f"[study] '{name}' already has {len(study.trials)} trials; not enqueueing.")
PYEOF

    # B.2 -- one DDP invocation; rank 0 runs the study, all ranks train each trial.
    HPO_ARGS=(
        "--kernel-dir"                  "$KERNEL_DIR"
        "--tokenizer-dir"               "$TOKENIZER_DIR"
        "--train-dataset-dir"           "$SCRATCH_TRAIN_DIR"
        "--eval-dataset-dir"            "$SCRATCH_EVAL_DIR"
        "--output-dir"                  "$OPTUNA_OUT_SCRATCH"
        "--study-name"                  "$STUDY_NAME"
        "--study-storage"               "$STUDY_STORAGE"
        "--load-if-exists"
        "--skip-final-train"
        "--n-trials"                    "$N_TRIALS"
        "--objective-metric"            "$OBJECTIVE_METRIC"
        "--hpo-direction"               "minimize"
        "--dropout-min"                 "$DROPOUT_MIN"
        "--dropout-max"                 "$DROPOUT_MAX"
        "--dropout-step"                "$DROPOUT_STEP"
        "--weight-decay-min"            "$WD_MIN"
        "--weight-decay-max"            "$WD_MAX"
        "--learning-rate"               "$BEST_LR"
        "--num-train-epochs"            "$HPO_EPOCHS"
        "--warmup-ratio"                "$WARMUP_RATIO"
        "--per-device-train-batch-size" "$HPO_BATCH_SIZE"
        "--per-device-eval-batch-size"  "$HPO_BATCH_SIZE"
        "--logging-steps"               "$STEP_INTERVAL"
        "--eval-steps"                  "$STEP_INTERVAL"
        "--save-steps"                  "$STEP_INTERVAL"
        "--dataloader-num-workers"      "$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
        "--dataloader-pin-memory"
        "--metric-for-best-model"       "$OBJECTIVE_METRIC"
        "--greater-is-better"           "false"
        "--early-stopping-patience"     "$HPO_PATIENCE"
        "--early-stopping-threshold"    "0.0"
        "--semantic-eval-batch-size"    "$EVAL_BATCH_SIZE"
        "--disable-train-end-semantic-eval"
        "--stage-name"                  "$OPTUNA_TAG"
        $MIXED_PRECISION
    )

    HPO_FAIL=0
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" scripts/hpo_optuna_wd_dropout.py "${HPO_ARGS[@]}" || HPO_FAIL=1
    else
        python scripts/hpo_optuna_wd_dropout.py "${HPO_ARGS[@]}" || HPO_FAIL=1
    fi

    # Persist logs + the study db regardless of outcome (checkpoints stay on scratch).
    cp -f "$STUDY_DB" "$OPTUNA_OUT_HOME/" 2>/dev/null || true
    if [ -d "$OPTUNA_OUT_SCRATCH/logs" ]; then
        rsync -a "$OPTUNA_OUT_SCRATCH/logs/" "$OPTUNA_OUT_HOME/logs/"
    fi
    if [ "$HPO_FAIL" -ne 0 ]; then
        echo "ERROR: Optuna DDP run failed."
        exit 1
    fi

    # B.3 -- query the finished study: global best, baseline value, Phase-C decision.
    python3 - "$STUDY_STORAGE" "$STUDY_NAME" "$BEST_RUN_JSON" "$STD_DROPOUT" "$STD_WD" "$WD_REL_TOL" "$OBJECTIVE_METRIC" <<'PYEOF'
import json, os, sys
import optuna
storage, name, out_json, std_dropout, std_wd, wd_rel_tol, metric = (
    sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]),
    float(sys.argv[6]), sys.argv[7])
study = optuna.load_study(study_name=name, storage=storage)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
baseline = [t for t in done if t.user_attrs.get("baseline")]
if not baseline:
    raise SystemExit("Baseline trial missing or failed -- cannot make the Phase-C decision.")
baseline_val = float(baseline[0].value)
best = study.best_trial
dropout = float(best.params.get("dropout", std_dropout))
wd = float(best.params.get("weight_decay", std_wd))
objective = float(best.value)
improved = objective < baseline_val
differs = (abs(dropout - std_dropout) > 1e-9) or (abs(wd - std_wd) / std_wd > wd_rel_tol)
decision = "yes" if (improved and differs) else "no"
print(f"[study] {len(done)} completed trials; baseline={baseline_val:.6f}; "
      f"best trial #{best.number}: objective={objective:.6f}, dropout={dropout}, wd={wd:.6g}; "
      f"improved={improved}, differs={differs} -> phase C: {decision}")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump({
        "run_id": str(best.number),
        "objective": objective,
        "objective_metric": metric,
        "direction": "minimize",
        "hyperparameters": {"dropout": dropout, "weight_decay": wd},
        "baseline_objective": baseline_val,
        "completed_trials": len(done),
        "phase_c_decision": decision,
    }, f, indent=2)
PYEOF
fi

# Decision variables come from the persisted json (works identically on resume).
read -r RUN_PHASE_C BEST_DROPOUT BEST_WD BASELINE_VAL <<< "$(python3 - "$BEST_RUN_JSON" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    best = json.load(f)
hp = best["hyperparameters"]
print(best["phase_c_decision"], hp["dropout"], hp["weight_decay"], best["baseline_objective"])
PYEOF
)"
echo "Phase C trigger: $RUN_PHASE_C  (best dropout=$BEST_DROPOUT, wd=$BEST_WD, baseline=$BASELINE_VAL)"

# ============================================================================
# PHASE C (conditional): LR grid at the tuned dropout / weight decay
# ============================================================================
ALL_TAGS=("${A_TAGS[@]}")
if [ "$RUN_PHASE_C" = "yes" ]; then
    echo ""
    echo "########## PHASE C: LR re-sweep @ dropout=$BEST_DROPOUT wd=$BEST_WD ##########"
    for LR in "${LR_VALUES[@]}"; do
        TAG="c_lr_${LR}"
        ALL_TAGS+=("$TAG")
        run_full "$TAG" "$LR" "$BEST_DROPOUT" "$BEST_WD"
    done
else
    echo "Phase C skipped: tuned dropout/wd did not beat the standard settings."
fi

# ============================================================================
# FINAL SELECTION: global best across all full-fidelity runs
# ============================================================================
echo ""
echo "########## FINAL SELECTION ##########"
read -r WINNER_TAG WINNER_VAL <<< "$(best_tag_of "${ALL_TAGS[@]}")"

WINNER_LR="${WINNER_TAG#a_lr_}"; WINNER_LR="${WINNER_LR#c_lr_}"
if [[ "$WINNER_TAG" == c_* ]]; then
    WINNER_DROPOUT="$BEST_DROPOUT"; WINNER_WD="$BEST_WD"
else
    WINNER_DROPOUT="$STD_DROPOUT"; WINNER_WD="$STD_WD"
fi

cat > "$HPO_ROOT/hpo_summary.json" <<SUMEOF
{
  "objective_metric": "$OBJECTIVE_METRIC",
  "winner_tag": "$WINNER_TAG",
  "winner_objective": $WINNER_VAL,
  "selected": {
    "learning_rate": "$WINNER_LR",
    "dropout": $WINNER_DROPOUT,
    "weight_decay": $WINNER_WD
  },
  "phase_a_best_lr": "$BEST_LR",
  "phase_c_ran": "$RUN_PHASE_C",
  "optuna_best": { "dropout": $BEST_DROPOUT, "weight_decay": $BEST_WD },
  "baseline_trial_objective": $BASELINE_VAL,
  "slurm_job_id": "${SLURM_JOB_ID:-N/A}"
}
SUMEOF

echo "Selected hyperparameters -> $HPO_ROOT/hpo_summary.json"
cat "$HPO_ROOT/hpo_summary.json"

echo ""
echo "Cleaning scratch-local..."
rm -rf "$SCRATCH_BASE"
echo "Done. End: $(date)"
