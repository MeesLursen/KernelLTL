# Curriculum Training Utilities

This repo now contains four small CLIs that let you precompute artefacts once and reuse
them across multiple curriculum stages:

1. `prepare_kernel.py` – builds traces, anchor formulas and the kernel feature matrix `F`, then saves everything.
2. `prepare_tokenizer.py` – instantiates (or reloads) an `LTLTokenizer` and saves it in Hugging Face format.
3. `prepare_datasets.py` – samples train/eval formula datasets using a saved kernel and persists them via `LTLDataset.save`.
4. `curriculum_train.py` – loads the saved kernel/tokenizer/datasets (and optionally a previous stage checkpoint) to run a training stage, then saves the resulting model for the next stage.

## Typical workflow

### 1. Build the kernel once

```bash
python prepare_kernel.py \
  --output-dir artifacts/kernel_stage0 \
  --trace-length 20 \
  --num-atomic-props 5 \
  --anchor-count 1024 \
  --epsilon 0.01 --delta 0.01 \
  --trace-sampler correlated \
  --anchor-sampler cosine \
  --cosine-batch-size 10240 --cosine-max-attempts 500
```

### 2. Save a tokenizer (either from `n_ap` or the kernel metadata)

```bash
python prepare_tokenizer.py \
  --output-dir artifacts/tokenizer_stage0 \
  --kernel-dir artifacts/kernel_stage0
```

### 3. Pre-generate datasets for each curriculum stage

```bash
python prepare_datasets.py \
  --kernel-dir artifacts/kernel_stage0 \
  --train-out datasets/stage1/train \
  --train-k 50000 --train-p-leaf 0.45 --train-max-depth 2 \
  --eval-out datasets/stage1/eval \
  --eval-k 2000 --eval-p-leaf 0.45 --eval-max-depth 2 \
  --eval-store-formula-str --eval-store-satisfaction
```

Repeat this step with different `k`, `p_leaf`, or `max_depth` to assemble the curriculum.

### 4. Run a curriculum training stage

```bash
python curriculum_train.py \
  --kernel-dir artifacts/kernel_stage0 \
  --tokenizer-dir artifacts/tokenizer_stage0 \
  --train-dataset-dir datasets/stage1/train \
  --eval-dataset-dir datasets/stage1/eval \
  --output-dir runs/stage1 \
  --model-save-dir runs/stage1/final_model \
  --num-train-epochs 20 --learning-rate 3e-4
```

For later stages, point `--model-load-dir` to the previous stage's `final_model` directory so the new training run starts from that checkpoint. Override any `TrainingArguments` either via command-line flags (`--num-train-epochs`, `--per-device-train-batch-size`, etc.) or by supplying a JSON file through `--training-args-file`.

Refer to each script's `--help` output for the full list of options and defaults.
