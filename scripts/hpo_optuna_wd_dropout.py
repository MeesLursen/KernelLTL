"""Hyperparameter search driver for CE curriculum training using HF Trainer + Optuna.

This script mirrors `scripts/curriculum_train.py` for data/model loading, but runs
`Trainer.hyperparameter_search()` to tune:
- dropout (applied to attn/resid/embd dropout in model config)
- weight_decay (TrainingArguments field)

After HPO, it trains one final run with the best hyperparameters and saves it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from transformers import EarlyStoppingCallback, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from ce_trainer import CETrainer
from config_class import LTLConfig
from dataset_class import LTLDataset
from kernel_class import LTLKernel
from model_class import LTLModel
from tokenizer_pretrained_class import LTLTokenizer
from training_utils import SemanticEvaluationCallback, UnifiedMetricsLoggerCallback


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _non_negative_float(value: str) -> float:
    fval = float(value)
    if fval < 0.0:
        raise argparse.ArgumentTypeError("Value must be non-negative")
    return fval


def _dropout_rate(value: str) -> float:
    fval = float(value)
    if fval < 0.0 or fval >= 1.0:
        raise argparse.ArgumentTypeError("Dropout must be in [0.0, 1.0)")
    return fval


def _positive_dropout_step(value: str) -> float:
    fval = float(value)
    if fval <= 0.0:
        raise argparse.ArgumentTypeError("Dropout step must be > 0")
    return fval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna-based hyperparameter search for CE curriculum training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--kernel-dir", required=True, help="Directory of the persisted kernel for semantic embeddings")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory containing the tokenizer (saved via prepare_tokenizer.py)")
    parser.add_argument("--train-dataset-dir", required=True, help="Directory containing a saved training dataset")
    parser.add_argument("--eval-dataset-dir", required=True, help="Directory containing a saved evaluation dataset")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Trainer artifacts (checkpoints, logs, etc.)",
    )
    parser.add_argument(
        "--model-save-dir",
        default=None,
        help="Directory where the best-hparams final model is stored. Defaults to <output-dir>/final_model",
    )
    parser.add_argument("--config-json", help="Optional JSON file with LTLConfig parameters for fresh initialisations")
    parser.add_argument("--training-args-load-dir", help="Directory containing a saved training_args.bin to seed TrainingArguments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to transformers.TrainingArguments")
    parser.add_argument("--stage-name", type=str, default=None, help="Optional stage identifier used in metric logs")

    # Model config controls
    config_group = parser.add_argument_group("Model config overrides")
    config_group.add_argument("--n-positions", type=_positive_int, default=512)
    config_group.add_argument("--n-layer", type=_positive_int, default=12)
    config_group.add_argument("--n-head", type=_positive_int, default=16)

    # TrainingArgument overrides
    train_group = parser.add_argument_group("Training argument overrides")
    train_group.add_argument("--num-train-epochs", type=float, default=None)
    train_group.add_argument("--learning-rate", type=float, default=None)
    train_group.add_argument("--per-device-train-batch-size", type=_positive_int, default=None)
    train_group.add_argument("--per-device-eval-batch-size", type=_positive_int, default=None)
    train_group.add_argument("--warmup-steps", type=_positive_int, default=None)
    train_group.add_argument("--warmup-ratio", type=float, default=None)
    train_group.add_argument("--logging-steps", type=float, default=None)
    train_group.add_argument("--eval-steps", type=float, default=None)
    train_group.add_argument("--save-steps", type=float, default=None)
    train_group.add_argument("--gradient-accumulation-steps", type=_positive_int, default=None)
    train_group.add_argument("--dataloader-num-workers", type=int, default=None)
    train_group.add_argument("--dataloader-pin-memory", action="store_true")
    train_group.add_argument("--no-dataloader-pin-memory", action="store_false", dest="dataloader_pin_memory")
    train_group.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision if supported")
    train_group.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision if supported")
    train_group.add_argument("--report-to", nargs="*", default=None, help="Backends to report metrics to (e.g. tensorboard)")
    train_group.add_argument("--metric-for-best-model", type=str, default="eval_semantic_distance")
    train_group.add_argument("--greater-is-better", choices=["true", "false"], default="false")
    train_group.add_argument("--early-stopping-patience", type=int, default=None)
    train_group.add_argument("--early-stopping-threshold", type=float, default=None)
    train_group.set_defaults(dataloader_pin_memory=True)

    callback_group = parser.add_argument_group("Semantic evaluation callback")
    callback_group.add_argument("--disable-semantic-callback", action="store_true")
    callback_group.add_argument(
        "--disable-train-end-semantic-eval",
        action="store_true",
        help="Skip SemanticEvaluationCallback.on_train_end computations (useful for faster HPO trials)",
    )
    callback_group.add_argument("--semantic-eval-batch-size", type=_positive_int, default=10240)

    hpo_group = parser.add_argument_group("HPO controls")
    hpo_group.add_argument("--n-trials", type=_positive_int, default=12, help="Number of Optuna trials")
    hpo_group.add_argument("--hpo-direction", choices=["minimize", "maximize"], default="minimize")
    hpo_group.add_argument(
        "--objective-metric",
        type=str,
        default="eval_semantic_distance",
        help="Metric key used as optimization objective (e.g. eval_semantic_distance)",
    )
    hpo_group.add_argument("--dropout-min", type=_dropout_rate, default=0.0)
    hpo_group.add_argument("--dropout-max", type=_dropout_rate, default=0.3)
    hpo_group.add_argument("--dropout-step", type=_positive_dropout_step, default=0.05)
    hpo_group.add_argument("--weight-decay-min", type=_non_negative_float, default=0.0)
    hpo_group.add_argument("--weight-decay-max", type=_non_negative_float, default=0.05)
    hpo_group.add_argument("--weight-decay-log", action="store_true", help="Use log sampling for weight decay")

    # Shared-study controls: multiple single-GPU worker processes can join one
    # study (e.g. sqlite storage on node-local scratch), each running a share
    # of the trials in parallel.
    hpo_group.add_argument("--study-name", type=str, default=None, help="Optuna study name (required for shared studies)")
    hpo_group.add_argument("--study-storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:////scratch/.../study.db")
    hpo_group.add_argument("--load-if-exists", action="store_true", help="Join an existing study instead of failing on name collision")
    hpo_group.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Skip the final full training pass after HPO (use when workers share a study and selection happens outside)",
    )

    return parser.parse_args()


def _validate_hpo_args(args: argparse.Namespace) -> None:
    if args.dropout_min > args.dropout_max:
        raise ValueError("--dropout-min must be <= --dropout-max")
    if args.weight_decay_min > args.weight_decay_max:
        raise ValueError("--weight-decay-min must be <= --weight-decay-max")
    if args.dropout_min == args.dropout_max and args.weight_decay_min == args.weight_decay_max:
        raise ValueError("At least one of dropout or weight decay ranges must span multiple values")
    if args.weight_decay_log and args.weight_decay_min <= 0.0:
        raise ValueError("--weight-decay-log requires --weight-decay-min > 0")


def _load_kernel(kernel_dir: str) -> LTLKernel:
    kernel = LTLKernel.load(kernel_dir)
    if kernel.F is None or kernel.traces is None or kernel.m is None:
        raise RuntimeError(
            "Kernel must include traces, anchor formulas, and the feature matrix F. Recreate it via prepare_kernel.py."
        )
    return kernel


def _load_dataset(path: str, load_satisfactions: bool) -> LTLDataset:
    dataset = LTLDataset.load(path, load_satisfactions=load_satisfactions)
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {path} is empty")
    return dataset


def _load_tokenizer(path: str) -> LTLTokenizer:
    return LTLTokenizer.from_pretrained(path)


def _load_training_args(args: argparse.Namespace) -> TrainingArguments:
    if args.training_args_load_dir:
        load_path = os.path.join(args.training_args_load_dir, TRAINING_ARGS_NAME)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Could not find {TRAINING_ARGS_NAME} in {args.training_args_load_dir}")
        loaded_args = torch.load(load_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded_args, TrainingArguments):
            raise TypeError(
                f"Expected {TRAINING_ARGS_NAME} to contain a TrainingArguments object, got {type(loaded_args)!r}"
            )
        base_kwargs: Dict[str, Any] = loaded_args.to_dict()
        base_kwargs["output_dir"] = args.output_dir
        base_kwargs["logging_dir"] = os.path.join(args.output_dir, "logs")
        print("Loaded training_args from directory.")
    else:
        base_kwargs = {
            "output_dir": args.output_dir,
            "num_train_epochs": 75,
            "learning_rate": 5e-4,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_strategy": "steps",
            "logging_steps": 0.02,
            "logging_dir": os.path.join(args.output_dir, "logs"),
            "eval_strategy": "steps",
            "eval_steps": 0.02,
            "save_strategy": "steps",
            "save_steps": 0.1,
            "save_safetensors": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_semantic_distance",
            "greater_is_better": False,
            "remove_unused_columns": False,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "report_to": ["all"],
            "ddp_find_unused_parameters": False,
        }
        print("Built training_args from scratch.")

    override_fields = {
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
    }
    for key, value in override_fields.items():
        if value is not None:
            base_kwargs[key] = value

    # HF picks warmup_steps when it is > 0, otherwise warmup_ratio. The from-scratch
    # defaults set warmup_steps=500, so honour an explicit --warmup-ratio by clearing
    # steps (unless --warmup-steps was also given, which then takes precedence as usual).
    if args.warmup_ratio is not None and args.warmup_steps is None:
        base_kwargs["warmup_steps"] = 0

    if args.report_to is not None:
        base_kwargs["report_to"] = args.report_to
    if args.metric_for_best_model is not None:
        base_kwargs["metric_for_best_model"] = args.metric_for_best_model
    if args.greater_is_better is not None:
        base_kwargs["greater_is_better"] = args.greater_is_better.lower() == "true"
    if args.fp16:
        base_kwargs["fp16"] = True
    if args.bf16:
        base_kwargs["bf16"] = True
    if args.seed is not None:
        base_kwargs["seed"] = args.seed

    return TrainingArguments(**base_kwargs)


def _build_model(
    args: argparse.Namespace,
    kernel: LTLKernel,
    tokenizer: LTLTokenizer,
    dropout: float,
) -> LTLModel:
    if kernel.m is None:
        raise ValueError("Kernel anchor set size 'm' is unknown")

    if args.config_json:
        config = LTLConfig.from_pretrained(args.config_json)
        config.n_embd = kernel.m
        config.vocab_size = tokenizer.vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    else:
        config = LTLConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=args.n_positions,
            n_embd=kernel.m,
            n_layer=args.n_layer,
            n_head=args.n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    config.attn_pdrop = dropout
    config.resid_pdrop = dropout
    config.embd_pdrop = dropout

    if config.n_embd % config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    return LTLModel(config, semantic_emb_dim=kernel.m)


def main() -> None:
    args = parse_args()
    _validate_hpo_args(args)

    # DDP is supported: HF's optuna backend runs the study on rank 0 and broadcasts
    # each trial's parameters to the other ranks as a FixedTrial. Consequences honoured
    # below: (1) every searched parameter must be suggested inside hp_space, because
    # only trial.params survives the broadcast (model_init runs after it); (2) pruning
    # must be disabled -- a pruned trial would abort rank 0 mid-run while other ranks
    # keep training, deadlocking the process group.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank != -1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if args.eval_dataset_dir is None:
        raise ValueError("--eval-dataset-dir is required for objective computation during HPO")

    kernel = _load_kernel(args.kernel_dir)
    tokenizer = _load_tokenizer(args.tokenizer_dir)
    train_dataset = _load_dataset(args.train_dataset_dir, load_satisfactions=False)
    eval_dataset = _load_dataset(args.eval_dataset_dir, load_satisfactions=True)

    if args.config_json:
        base_config = LTLConfig.from_pretrained(args.config_json)
        collate_max_length = int(base_config.n_positions)
    else:
        collate_max_length = int(args.n_positions)

    training_args = _load_training_args(args)

    # Current trial dropout, set by hp_space (which runs on every rank via
    # Trainer._hp_search_setup before model_init); updated again for the final run.
    current_dropout = {"value": args.dropout_min}

    def model_init(trial=None):
        return _build_model(args, kernel, tokenizer, dropout=float(current_dropout["value"]))

    max_length_hint = collate_max_length
    if isinstance(max_length_hint, int) and max_length_hint > 0:
        tokenizer.model_max_length = max_length_hint

    callbacks = []
    semantic_callback = None
    if not args.disable_semantic_callback:
        semantic_callback = SemanticEvaluationCallback(
            tokenizer=tokenizer,
            enable_train_end_eval=not args.disable_train_end_semantic_eval,
        )
        callbacks.append(semantic_callback)

    metrics_logger_callback = UnifiedMetricsLoggerCallback(
        output_dir=args.output_dir,
        stage_name=args.stage_name,
    )
    if semantic_callback is not None:
        semantic_callback.attach_metrics_logger(metrics_logger_callback)
    callbacks.append(metrics_logger_callback)

    if args.early_stopping_patience is not None:
        if args.early_stopping_patience <= 0:
            raise ValueError("--early-stopping-patience must be > 0")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=(
                    args.early_stopping_threshold if args.early_stopping_threshold is not None else 0.0
                ),
            )
        )

    trainer = CETrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        data_collator=lambda batch: tokenizer.collate_batch(batch, collate_max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        processing_class=tokenizer,
        kernel=kernel,
        semantic_eval_batch_size=args.semantic_eval_batch_size,
    )
    metrics_logger_callback.attach_trainer(trainer)
    if semantic_callback is not None:
        semantic_callback.attach_trainer(trainer)

    def hp_space(trial):
        # ALL searched parameters are suggested here so they land in trial.params
        # before the DDP broadcast. Dropout is not a TrainingArguments field, so it
        # is deliberately NOT part of the returned dict (Trainer would only warn);
        # it reaches the model through current_dropout -> model_init, which
        # _hp_search_setup guarantees is set on every rank (FixedTrial re-suggest
        # returns the broadcast value).
        if args.dropout_min == args.dropout_max:
            trial_dropout = float(args.dropout_min)
        else:
            trial_dropout = float(
                trial.suggest_float(
                    "dropout",
                    args.dropout_min,
                    args.dropout_max,
                    step=args.dropout_step,
                )
            )
        current_dropout["value"] = trial_dropout

        if args.weight_decay_min == args.weight_decay_max:
            weight_decay = float(args.weight_decay_min)
        else:
            weight_decay = float(
                trial.suggest_float(
                    "weight_decay",
                    args.weight_decay_min,
                    args.weight_decay_max,
                    log=args.weight_decay_log,
                )
            )
        print(f"[HPO trial {trial.number}] dropout={trial_dropout:.4f}, weight_decay={weight_decay:.6g}")
        return {"weight_decay": weight_decay}

    def compute_objective(metrics: Dict[str, float]) -> float:
        if args.objective_metric not in metrics:
            raise KeyError(
                f"Objective metric '{args.objective_metric}' not found in eval metrics. "
                f"Available keys: {sorted(metrics.keys())}"
            )
        return float(metrics[args.objective_metric])

    import optuna

    study_kwargs: Dict[str, Any] = {"pruner": optuna.pruners.NopPruner()}
    if args.study_name is not None:
        study_kwargs["study_name"] = args.study_name
    if args.study_storage is not None:
        study_kwargs["storage"] = args.study_storage
    if args.load_if_exists:
        study_kwargs["load_if_exists"] = True

    print(f"Starting HF hyperparameter_search (backend=optuna, world_size={world_size}, study_kwargs={study_kwargs})...")
    best_run = trainer.hyperparameter_search(
        backend="optuna",
        direction=args.hpo_direction,
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=args.n_trials,
        **study_kwargs,
    )

    # Under DDP, hyperparameter_search returns the BestRun on rank 0 only.
    best_dropout: float | None = None
    best_weight_decay: float | None = None
    if best_run is not None:
        best_hparams = dict(best_run.hyperparameters)
        best_dropout = float(best_hparams.get("dropout", args.dropout_min))
        best_weight_decay = float(best_hparams.get("weight_decay", args.weight_decay_min))

        print("HPO completed.")
        print(f"Best run id: {best_run.run_id}")
        print(f"Best objective: {best_run.objective}")
        print(f"Best dropout: {best_dropout}")
        print(f"Best weight_decay: {best_weight_decay}")

        os.makedirs(training_args.logging_dir, exist_ok=True)
        best_run_path = os.path.join(training_args.logging_dir, "hpo_best_run.json")
        with open(best_run_path, "w") as f:
            json.dump(
                {
                    "run_id": best_run.run_id,
                    "objective": float(best_run.objective),
                    "objective_metric": args.objective_metric,
                    "direction": args.hpo_direction,
                    "hyperparameters": best_hparams,
                },
                f,
                indent=2,
            )
    else:
        print(f"[rank {local_rank}] hyperparameter_search finished; results live on rank 0.")

    if args.skip_final_train:
        print("Skipping final training pass (--skip-final-train); selection happens outside this script.")
        return

    # Final training pass with best discovered hyperparameters (all ranks train,
    # so under DDP the best parameters are broadcast from rank 0 first).
    if world_size > 1:
        import torch.distributed as dist

        payload = [best_dropout, best_weight_decay]
        dist.broadcast_object_list(payload, src=0)
        best_dropout, best_weight_decay = float(payload[0]), float(payload[1])

    trainer.args.weight_decay = best_weight_decay
    current_dropout["value"] = best_dropout

    train_result = trainer.train()
    print(train_result)

    if best_run is not None:
        log_dir = training_args.logging_dir
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "trainer_log_history.log")
        with open(log_path, "w") as f:
            for entry in trainer.state.log_history:
                f.write(json.dumps(entry) + "\n")

    save_dir = args.model_save_dir or os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_dir)


if __name__ == "__main__":
    main()
