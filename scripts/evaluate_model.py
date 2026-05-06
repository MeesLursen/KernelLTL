"""Standalone evaluation script for KernelLTL models."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from transformers import TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from ce_trainer import CETrainer
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved KernelLTL model on an evaluation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--kernel-dir", required=True, help="Directory of the persisted kernel for semantic embeddings")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory containing the tokenizer (saved via prepare_tokenizer.py)")
    parser.add_argument("--eval-dataset-dir", required=True, help="Directory containing a saved evaluation dataset")
    parser.add_argument("--model-load-dir", required=True, help="Directory with a saved model to evaluate")
    parser.add_argument(
        "--ce-reference-model-dir",
        default=None,
        help="Directory of the CE reference model used for stage-end KL metric computation",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for evaluation artifacts (logs, metrics, etc.)",
    )
    parser.add_argument("--training-args-load-dir", help="Directory containing a saved training_args.bin to seed TrainingArguments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to transformers.TrainingArguments")
    parser.add_argument("--stage-name", type=str, default=None, help="Optional stage identifier used in metric logs")
    parser.add_argument(
        "--trainer-kind",
        type=str,
        choices=["ce", "rb", "gae"],
        default="ce",
        help="The trainer algorithm used during training to log properly and load auxiliary states.",
    )

    eval_group = parser.add_argument_group("Evaluation argument overrides")
    eval_group.add_argument("--per-device-eval-batch-size", type=_positive_int, default=None)
    eval_group.add_argument("--dataloader-num-workers", type=int, default=None)
    eval_group.add_argument("--dataloader-pin-memory", action="store_true")
    eval_group.add_argument("--no-dataloader-pin-memory", action="store_false", dest="dataloader_pin_memory")
    eval_group.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision if supported")
    eval_group.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision if supported")
    eval_group.add_argument("--report-to", nargs="*", default=None, help="Backends to report metrics to (e.g. tensorboard)")
    eval_group.set_defaults(dataloader_pin_memory=True)

    callback_group = parser.add_argument_group("Semantic evaluation callback")
    callback_group.add_argument("--disable-semantic-callback", action="store_true")
    callback_group.add_argument("--semantic-eval-batch-size", type=_positive_int, default=10240)
    callback_group.add_argument(
        "--callback-debug",
        action="store_true",
        help="Enable detailed debug prints for metrics gathering/aggregation in callbacks",
    )

    return parser.parse_args()


def _load_kernel(kernel_dir: str) -> LTLKernel:
    kernel = LTLKernel.load(kernel_dir)
    if kernel.F is None or kernel.traces is None or kernel.m is None:
        raise RuntimeError(
            "Kernel must include traces, anchor formulas, and the feature matrix F. Recreate it via prepare_kernel.py."
        )
    return kernel


def _load_dataset(path: str) -> LTLDataset:
    dataset = LTLDataset.load(path, load_satisfactions=True)
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
        base_kwargs["do_train"] = False
        base_kwargs["do_eval"] = True
        print("Loaded training_args from directory.")
    else:
        base_kwargs = {
            "output_dir": args.output_dir,
            "per_device_eval_batch_size": 32,
            "logging_dir": os.path.join(args.output_dir, "logs"),
            "remove_unused_columns": False,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "report_to": ["all"],
            "ddp_find_unused_parameters": False,
        }
        print("Built training_args from scratch.")

    override_fields = {
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
    }
    for key, value in override_fields.items():
        if value is not None:
            base_kwargs[key] = value

    if args.report_to is not None:
        base_kwargs["report_to"] = args.report_to
    if args.fp16:
        base_kwargs["fp16"] = True
    if args.bf16:
        base_kwargs["bf16"] = True
    if args.seed is not None:
        base_kwargs["seed"] = args.seed

    return TrainingArguments(**base_kwargs)


def _build_model(args: argparse.Namespace, kernel: LTLKernel) -> LTLModel:
    if kernel.m is None:
        raise ValueError("Kernel anchor set size 'm' is unknown")

    if not os.path.isdir(args.model_load_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_load_dir}")

    print(f"Loading model from {args.model_load_dir}")
    model = LTLModel.from_pretrained(args.model_load_dir)
    if model.config.n_embd != kernel.m:
        raise ValueError(
            f"Loaded model embedding dim ({model.config.n_embd}) does not match kernel anchor count ({kernel.m})."
        )
    return model


def main() -> None:
    args = parse_args()

    # Respect DDP local rank if applicable
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    os.makedirs(args.output_dir, exist_ok=True)

    kernel = _load_kernel(args.kernel_dir)
    tokenizer = _load_tokenizer(args.tokenizer_dir)
    eval_dataset = _load_dataset(args.eval_dataset_dir)

    training_args = _load_training_args(args)
    model = _build_model(args, kernel)

    max_length_hint = getattr(model.config, "n_positions", None)
    if isinstance(max_length_hint, int) and max_length_hint > 0:
        tokenizer.model_max_length = max_length_hint

    callbacks = []
    semantic_callback = None
    if not args.disable_semantic_callback:
        semantic_callback = SemanticEvaluationCallback(
            tokenizer=tokenizer,
            enable_train_end_eval=True,
            debug_metrics=args.callback_debug,
        )
        callbacks.append(semantic_callback)

    metrics_logger_callback = UnifiedMetricsLoggerCallback(
        output_dir=args.output_dir,
        stage_name=args.stage_name,
        debug_metrics=args.callback_debug,
    )
    if semantic_callback is not None:
        semantic_callback.attach_metrics_logger(metrics_logger_callback)
    callbacks.append(metrics_logger_callback)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        data_collator=lambda batch: tokenizer.collate_batch(batch, model.config.n_positions, include_metadata=True),
        train_dataset=None,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        tokenizer=tokenizer,
        kernel=kernel,
        semantic_eval_batch_size=args.semantic_eval_batch_size,
    )

    if args.trainer_kind == "ce":
        trainer = CETrainer(processing_class=tokenizer, **trainer_kwargs)
    elif args.trainer_kind == "rb":
        trainer = REINFORCETrainerRB(**trainer_kwargs)
    elif args.trainer_kind == "gae":
        trainer = REINFORCETrainerGAE(**trainer_kwargs)
    else:
        raise ValueError(f"Unknown trainer kind: {args.trainer_kind}")

    if args.trainer_kind in ["rb", "gae"] and hasattr(trainer, "load_trainer_state"):
        trainer.load_trainer_state(args.model_load_dir, load_optimizer=False, strict_critic=False)

    ce_reference_model_dir = args.ce_reference_model_dir
    if ce_reference_model_dir is not None and not os.path.isdir(ce_reference_model_dir):
        raise FileNotFoundError(
            f"CE reference model directory not found: {ce_reference_model_dir}"
        )
    trainer._ce_reference_model_path = ce_reference_model_dir

    metrics_logger_callback.attach_trainer(trainer)
    if semantic_callback is not None:
        semantic_callback.attach_trainer(trainer)

    metrics = trainer.evaluate()
    print(metrics)

    if semantic_callback is not None:
        semantic_callback.on_train_end(
            args=training_args,
            state=trainer.state,
            control=trainer.control,
            model=trainer.model,
        )

    metrics_logger_callback.on_train_end(
        args=training_args,
        state=trainer.state,
        control=trainer.control,
    )

    log_dir = training_args.logging_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "trainer_log_history.log")

    with open(log_path, "w") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
