"""Driver script for curriculum training across pre-generated datasets and kernels.

Highlights:
- accepts ``--training-args-load-dir`` to seed :class:`transformers.TrainingArguments`
    from a previous stage's ``training_args.bin`` before applying any overrides.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from transformers import EarlyStoppingCallback, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from config_class import LTLConfig
from reinforce_trainer import REINFORCETrainerRB, REINFORCETrainerGAE
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


def _dropout_rate(value: str) -> float:
    fval = float(value)
    if fval < 0.0 or fval >= 1.0:
        raise argparse.ArgumentTypeError("Dropout must be in [0.0, 1.0)")
    return fval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load kernel/tokenizer/datasets/checkpoints and run a curriculum training stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--kernel-dir", required=True, help="Directory of the persisted kernel for semantic embeddings")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory containing the tokenizer (saved via prepare_tokenizer.py)")
    parser.add_argument("--train-dataset-dir", required=True, help="Directory containing a saved training dataset")
    parser.add_argument("--eval-dataset-dir", help="Directory containing a saved evaluation dataset")
    parser.add_argument("--model-load-dir", help="Directory with a previously saved curriculum checkpoint to resume from")
    parser.add_argument(
        "--ce-reference-model-dir",
        default=None,
        help="Directory of the CE reference model used for stage-end KL metric computation",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Trainer artifacts (checkpoints, logs, etc.)",
    )
    parser.add_argument(
        "--model-save-dir",
        default=None,
        help="Directory where the trained model for this stage will be stored. Defaults to <output-dir>/final_model",
    )
    parser.add_argument("--config-json", help="Optional JSON file with LTLConfig parameters for fresh initialisations")
    parser.add_argument("--training-args-load-dir", help="Directory containing a saved training_args.bin to seed TrainingArguments")
    parser.add_argument("--resume-from-checkpoint", help="Checkpoint directory passed to Trainer.train")
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to transformers.TrainingArguments")
    parser.add_argument("--stage-name", type=str, default=None, help="Optional stage identifier used in metric logs")

    # Config overrides when starting from scratch
    config_group = parser.add_argument_group("Model config overrides (when not loading a checkpoint)")
    config_group.add_argument("--n-positions", type=_positive_int, default=512)
    config_group.add_argument("--n-layer", type=_positive_int, default=12)
    config_group.add_argument("--n-head", type=_positive_int, default=16)
    config_group.add_argument(
        "--dropout",
        type=_dropout_rate,
        default=None,
        help="Optional dropout override applied to attn_pdrop/resid_pdrop/embd_pdrop for fresh model initialisation",
    )

    # TrainingArgument overrides
    train_group = parser.add_argument_group("Training argument overrides")
    train_group.add_argument("--num-train-epochs", type=float, default=None)
    train_group.add_argument("--learning-rate", type=float, default=None)
    train_group.add_argument("--per-device-train-batch-size", type=_positive_int, default=None)
    train_group.add_argument("--per-device-eval-batch-size", type=_positive_int, default=None)
    train_group.add_argument("--warmup-steps", type=float, default=None)
    train_group.add_argument("--weight-decay", type=float, default=None)
    train_group.add_argument("--logging-steps", type=float, default=None)
    train_group.add_argument("--eval-steps", type=float, default=None)
    train_group.add_argument("--save-steps", type=float, default=None)
    train_group.add_argument(
        "--debug", nargs="*", choices=["underflow_overflow"], default=None,
        help="Enable HF debug utilities (e.g., underflow/overflow checks)."
    )
    train_group.add_argument("--gradient-accumulation-steps", type=_positive_int, default=None)
    train_group.add_argument("--dataloader-num-workers", type=int, default=None)
    train_group.add_argument("--dataloader-pin-memory", action="store_true")
    train_group.add_argument("--no-dataloader-pin-memory", action="store_false", dest="dataloader_pin_memory")
    train_group.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision if supported")
    train_group.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision if supported")
    train_group.add_argument("--report-to", nargs="*", default=None, help="Backends to report metrics to (e.g. tensorboard)")
    train_group.add_argument("--metric-for-best-model", type=str, default="eval_semantic_distance", help="Metric used for best model selection, e.g. eval_loss or eval_semantic_distance")
    train_group.add_argument("--greater-is-better", choices=["true", "false"], default="false", help="Whether larger metric values are better for best model selection")
    train_group.add_argument("--early-stopping-patience", type=int, default=None, help="Number of evaluation calls with no significant improvement before stopping")
    train_group.add_argument("--early-stopping-threshold", type=float, default=None, help="Minimum metric improvement required to reset early stopping patience")
    train_group.set_defaults(dataloader_pin_memory=True)

    # Semantic evaluation callback controls
    callback_group = parser.add_argument_group("Semantic evaluation callback")
    callback_group.add_argument("--disable-semantic-callback", action="store_true")
    callback_group.add_argument("--semantic-eval-batch-size", type=_positive_int, default=10240)
    callback_group.add_argument(
        "--callback-debug",
        action="store_true",
        help="Enable detailed debug prints for metrics gathering/aggregation in callbacks",
    )

    # RL trainer controls
    rl_group = parser.add_argument_group("RL trainer controls")
    rl_group.add_argument(
        "--rl-trainer",
        choices=["rb", "gae"],
        default="gae",
        help="RL trainer variant: rb=running baseline REINFORCE, gae=actor-critic with GAE",
    )
    rl_group.add_argument("--reinforce-baseline-momentum", type=float, default=0.9)
    rl_group.add_argument("--reinforce-reward-clip", type=float, default=1.0)
    rl_group.add_argument("--gae-gamma", type=float, default=0.99)
    rl_group.add_argument("--gae-lambda", type=float, default=0.95)
    rl_group.add_argument("--critic-loss-coef", type=float, default=0.5)
    rl_group.add_argument("--critic-lr", type=float, default=None)
    rl_group.add_argument("--critic-hidden-dim", type=_positive_int, default=256)
    rl_group.add_argument("--critic-weight-decay", type=float, default=0.0)

    return parser.parse_args()


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


def _resolve_satisfactions_path(args: argparse.Namespace) -> str:
    default_path = os.path.join(args.train_dataset_dir, "satisfactions.pt")
    if not os.path.exists(default_path):
        raise FileNotFoundError(
            f"Could not find satisfactions.pt in train dataset directory: {default_path}"
        )
    return default_path


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
        print('Loaded training_args from directory.')
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
        print('Built training_args from scratch.')

    override_fields = {
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
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


def _build_model(args: argparse.Namespace, kernel: LTLKernel, tokenizer: LTLTokenizer) -> LTLModel:
    if kernel.m is None:
        raise ValueError("Kernel anchor set size 'm' is unknown")

    if args.model_load_dir:
        if args.dropout is not None:
            raise ValueError("--dropout cannot be used together with --model-load-dir because loaded checkpoints keep their saved dropout settings")
        print(f"Loading model from {args.model_load_dir}")
        model = LTLModel.from_pretrained(args.model_load_dir)
        if model.config.n_embd != kernel.m:
            raise ValueError(
                f"Loaded model embedding dim ({model.config.n_embd}) does not match kernel anchor count ({kernel.m})."
            )
        return model

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

    if args.dropout is not None:
        config.attn_pdrop = args.dropout
        config.resid_pdrop = args.dropout
        config.embd_pdrop = args.dropout

    if config.n_embd % config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    model = LTLModel(config, semantic_emb_dim=kernel.m)
    return model


def main() -> None:
    args = parse_args()

    # Respect DDP local rank if applicable
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    kernel = _load_kernel(args.kernel_dir)
    tokenizer = _load_tokenizer(args.tokenizer_dir)
    train_dataset = _load_dataset(args.train_dataset_dir, load_satisfactions=False)
    eval_dataset = _load_dataset(args.eval_dataset_dir, load_satisfactions=True) if args.eval_dataset_dir else None

    training_args = _load_training_args(args)
    model = _build_model(args, kernel, tokenizer)
    satisfactions_path = _resolve_satisfactions_path(args)

    max_length_hint = getattr(model.config, "n_positions", None)
    if isinstance(max_length_hint, int) and max_length_hint > 0:
        tokenizer.model_max_length = max_length_hint


    callbacks = []
    semantic_callback = None
    if not args.disable_semantic_callback and eval_dataset is not None:
        semantic_callback = SemanticEvaluationCallback(
            tokenizer=tokenizer,
            debug_metrics=args.callback_debug,
        )
        callbacks.append(semantic_callback)
    metrics_logger = UnifiedMetricsLoggerCallback(
        output_dir=args.output_dir,
        stage_name=args.stage_name,
        trainer_kind=args.rl_trainer,
        debug_metrics=args.callback_debug,
    )
    if semantic_callback is not None:
        semantic_callback.attach_metrics_logger(metrics_logger)
    callbacks.append(metrics_logger)
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


    trainer_common_kwargs = {
        "model": model,
        "args": training_args,
        "data_collator": lambda batch: tokenizer.collate_batch(batch, model.config.n_positions),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": callbacks,
        "processing_class": tokenizer,
        "kernel": kernel,
        "tokenizer": tokenizer,
        "reward_clip": args.reinforce_reward_clip,
        "semantic_eval_batch_size": args.semantic_eval_batch_size,
        "satisfactions_path": satisfactions_path,
    }

    if args.rl_trainer == "rb":
        print("Using REINFORCETrainerRB")
        trainer = REINFORCETrainerRB(
            **trainer_common_kwargs,
            baseline_momentum=args.reinforce_baseline_momentum,
        )
    else:
        print("Using REINFORCETrainerGAE")
        trainer = REINFORCETrainerGAE(
            **trainer_common_kwargs,
            gae_gamma=args.gae_gamma,
            gae_lambda=args.gae_lambda,
            critic_loss_coef=args.critic_loss_coef,
            critic_lr=args.critic_lr,
            critic_hidden_dim=args.critic_hidden_dim,
            critic_weight_decay=args.critic_weight_decay,
        )

    if args.model_load_dir is not None and hasattr(trainer, "load_trainer_state"):
        trainer.load_trainer_state(args.model_load_dir, load_optimizer=False, strict_critic=True)
    ce_reference_model_dir = args.ce_reference_model_dir or args.model_load_dir
    if ce_reference_model_dir is not None and not os.path.isdir(ce_reference_model_dir):
        raise FileNotFoundError(
            f"CE reference model directory not found: {ce_reference_model_dir}"
        )
    trainer._ce_reference_model_path = ce_reference_model_dir
    metrics_logger.attach_trainer(trainer)
    semantic_callback.attach_trainer(trainer)
    
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(train_result)

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
