# Curriculum Training Workflow

This repository uses a stage-based workflow where kernel/tokenizer/datasets are prepared once,
then CE or RE stages are trained sequentially and saved as `final_model` artifacts.

## Scripts

1. `scripts/prepare_kernel.py`
   - Builds traces, anchor formulas, and kernel feature matrix `F`.
2. `scripts/prepare_tokenizer.py`
   - Builds/saves `LTLTokenizer` from kernel metadata.
3. `scripts/prepare_datasets.py`
   - Builds train/eval datasets per stage and supports disjoint split mode.
4. `scripts/curriculum_train.py`
   - CE stage training entrypoint.
5. `scripts/curriculum_train_reinforce.py`
   - RE stage training entrypoint (`rb` or `gae`).

## Typical Local Workflow

### 1. Build kernel once

```bash
python scripts/prepare_kernel.py \
  --output-dir artifacts/kernel \
  --trace-length 20 \
  --num-atomic-props 5 \
  --anchor-count 1024 \
  --epsilon 0.01 \
  --delta 0.01
```

### 2. Build/save tokenizer

```bash
python scripts/prepare_tokenizer.py \
  --output-dir artifacts/tokenizer \
  --kernel-dir artifacts/kernel
```

### 3. Build per-stage datasets

Example independent sampling mode:

```bash
python scripts/prepare_datasets.py \
  --kernel-dir artifacts/kernel \
  --train-out artifacts/datasets/stage0/train \
  --train-k 50000 \
  --train-p-leaf-range 0.45 0.45 \
  --train-max-depth 2 \
  --eval-out artifacts/datasets/stage0/eval \
  --eval-k 2000 \
  --eval-p-leaf-range 0.45 0.45 \
  --eval-max-depth 2 \
  --eval-store-formula-str \
  --eval-store-satisfaction
```

Example disjoint split mode:

```bash
python scripts/prepare_datasets.py \
  --kernel-dir artifacts/kernel \
  --disjoint-split \
  --eval-ratio 0.05 \
  --train-out artifacts/datasets/stage1/train \
  --eval-out artifacts/datasets/stage1/eval \
  --train-k 60000 \
  --train-p-leaf-range 0.35 0.55 \
  --train-max-depth 3
```

### 4A. Train a CE stage

```bash
python scripts/curriculum_train.py \
  --kernel-dir artifacts/kernel \
  --tokenizer-dir artifacts/tokenizer \
  --train-dataset-dir artifacts/datasets/stage0/train \
  --eval-dataset-dir artifacts/datasets/stage0/eval \
  --output-dir artifacts/models/CE/stage0 \
  --model-save-dir artifacts/models/CE/stage0/final_model \
  --stage-name stage0 \
  --num-train-epochs 50 \
  --learning-rate 5e-4
```

### 4B. Train an RE stage (RB or GAE)

```bash
python scripts/curriculum_train_reinforce.py \
  --kernel-dir artifacts/kernel \
  --tokenizer-dir artifacts/tokenizer \
  --train-dataset-dir artifacts/datasets/stage0/train \
  --eval-dataset-dir artifacts/datasets/stage0/eval \
  --output-dir artifacts/models/RE/stage0 \
  --model-save-dir artifacts/models/RE/stage0/final_model \
  --model-load-dir artifacts/models/RE/stage_prev/final_model \
  --ce-reference-model-dir artifacts/models/CE/stage0/final_model \
  --stage-name stage0 \
  --rl-trainer gae
```

## Stage Chaining

- `--model-load-dir`: warm-starts current stage from a previous model checkpoint.
- `--training-args-load-dir`: loads previous stage `training_args.bin`, then applies CLI overrides.
- `--model-save-dir`: defaults to `<output-dir>/final_model` if not set.
- RE KL metrics use `--ce-reference-model-dir` and do not assume it equals `--model-load-dir`.

## Metrics And Callbacks

- Both training scripts use:
  - `SemanticEvaluationCallback`
  - `UnifiedMetricsLoggerCallback` (JSONL logs at `<output-dir>/logs/metrics_history.jsonl`)
- Stage metadata can be labeled with `--stage-name`.
- Early stopping defaults to `eval_semantic_distance` (`greater_is_better=false`).

## Multi-GPU And Cluster Notes

- Local multi-GPU:

```bash
torchrun --nproc_per_node=<N> scripts/curriculum_train_reinforce.py ...
```

- Snellius workflows are in `jobs/` and assume scratch-node usage.
- Current RE Slurm workflow writes stage outputs to local scratch first, then synchronizes back to project storage using `rsync`.

## Practical Tips

- Keep CE and RE model roots separate (for example `artifacts/models/CE/...` and `artifacts/models/RE/...`).
- Use consistent stage names across dataset, CE model, and RE model directories.
- For full option details, run `--help` on each script.
