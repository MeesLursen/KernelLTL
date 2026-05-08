"""End-to-end validation entry point for KernelLTL trained models.

Runs two passes per model on a held-out validation dataset:

  1. Greedy single-sample.
  2. Top-K (K=5 by default, T=1, do_sample=True).

Writes per-sample (and per-token) JSONL records under
``<output-dir>/per_sample/`` plus aggregated summaries under
``<output-dir>/`` for downstream post-hoc statistical analysis.

Usage (single GPU)::

    python scripts/validate_model.py \
        --kernel-dir <kernel> --tokenizer-dir <tokenizer> \
        --eval-dataset-dir <validation_dataset> \
        --model-load-dir <trained_model> \
        --ce-reference-model-dir <ce_base_model> \
        --output-dir <out>

Usage (multi-GPU)::

    torchrun --nproc_per_node=N scripts/validate_model.py ...

Metric-computation logic lives in ``validation_passes.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset_class import LTLDataset
from kernel_class import LTLKernel
from model_class import LTLModel
from tokenizer_pretrained_class import LTLTokenizer

from validation_utils import (
    aggregate_greedy_by_depth,
    aggregate_topk_by_depth,
    run_greedy_pass,
    run_topk_pass,
)


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a saved KernelLTL model on a held-out dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--eval-dataset-dir", required=True)
    parser.add_argument("--model-load-dir", required=True)
    parser.add_argument("--ce-reference-model-dir", default=None,
                        help="Reference (CE base) model directory for KL computation. "
                             "If omitted, KL columns are filled with zeros.")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--per-device-eval-batch-size", type=_positive_int, default=96)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--semantic-eval-batch-size", type=_positive_int, default=10240)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)

    parser.add_argument("--no-greedy", action="store_true",
                        help="Skip the greedy pass.")
    parser.add_argument("--no-topk", action="store_true",
                        help="Skip the top-K pass.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def _load_kernel(kernel_dir: str) -> LTLKernel:
    kernel = LTLKernel.load(kernel_dir)
    if kernel.F is None or kernel.traces is None or kernel.m is None:
        raise RuntimeError(
            "Kernel must include traces, anchor formulas, and the feature matrix F. "
            "Recreate it via prepare_kernel.py."
        )
    return kernel


def _load_dataset(path: str) -> LTLDataset:
    dataset = LTLDataset.load(path, load_satisfactions=True)
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {path} is empty")
    if dataset.satisfactions is None:
        raise ValueError(
            f"Dataset at {path} is missing per-sample satisfaction tensors. "
            "Validation requires them for semantic distance / reward computation."
        )
    return dataset


def _build_dataloader(
    *,
    dataset: LTLDataset,
    tokenizer: LTLTokenizer,
    model_n_positions: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: tokenizer.collate_batch(
            b, max_len=model_n_positions, include_metadata=True
        ),
    )


def _build_model(model_dir: str, kernel: LTLKernel) -> LTLModel:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model = LTLModel.from_pretrained(model_dir)
    if model.config.n_embd != kernel.m and model.semantic_emb_dim != kernel.m:
        raise ValueError(
            f"Loaded model embedding dim ({model.config.n_embd}) does not match "
            f"kernel anchor count ({kernel.m})."
        )
    return model


def main() -> None:
    args = parse_args()

    mixed_precision = "no"
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"

    accelerator = Accelerator(mixed_precision=mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "per_sample"), exist_ok=True)

    kernel = _load_kernel(args.kernel_dir)
    tokenizer = LTLTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = _load_dataset(args.eval_dataset_dir)
    model = _build_model(args.model_load_dir, kernel)

    # Move kernel + model to the accelerator's device
    if hasattr(kernel, "set_device"):
        kernel.set_device(accelerator.device)
    model.to(accelerator.device)

    n_positions = int(model.config.n_positions)
    if isinstance(n_positions, int) and n_positions > 0:
        tokenizer.model_max_length = n_positions

    dataloader = _build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        model_n_positions=n_positions,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    dataloader = accelerator.prepare(dataloader)

    ref_path = args.ce_reference_model_dir
    if ref_path is not None and not os.path.isdir(ref_path):
        raise FileNotFoundError(f"CE reference model directory not found: {ref_path}")

    greedy_jsonl = os.path.join(args.output_dir, "per_sample", "greedy.jsonl")
    topk_flat_jsonl = os.path.join(args.output_dir, "per_sample", "topk_flat.jsonl")
    topk_grouped_jsonl = os.path.join(args.output_dir, "per_sample", "topk_grouped.jsonl")

    summary: dict = {
        "model_load_dir": args.model_load_dir,
        "eval_dataset_dir": args.eval_dataset_dir,
        "ce_reference_model_dir": ref_path,
        "n_dataset_samples": len(dataset),
        "top_k": args.top_k,
    }

    if not args.no_greedy:
        if accelerator.is_main_process:
            print("=" * 60)
            print("Running greedy pass")
            print("=" * 60)
        greedy_summary = run_greedy_pass(
            model=model,
            ref_model_path=ref_path,
            eval_dataloader=dataloader,
            kernel=kernel,
            tokenizer=tokenizer,
            dataset=dataset,
            accelerator=accelerator,
            output_jsonl_path=greedy_jsonl,
            semantic_eval_batch_size=args.semantic_eval_batch_size,
        )
        if accelerator.is_main_process:
            summary["greedy"] = greedy_summary
            print("[greedy summary]", json.dumps(greedy_summary, indent=2))

    accelerator.wait_for_everyone()

    if not args.no_topk:
        if accelerator.is_main_process:
            print("=" * 60)
            print(f"Running top-{args.top_k} pass")
            print("=" * 60)
        topk_summary = run_topk_pass(
            model=model,
            ref_model_path=ref_path,
            eval_dataloader=dataloader,
            kernel=kernel,
            tokenizer=tokenizer,
            dataset=dataset,
            accelerator=accelerator,
            top_k=args.top_k,
            output_flat_path=topk_flat_jsonl,
            output_grouped_path=topk_grouped_jsonl,
            semantic_eval_batch_size=args.semantic_eval_batch_size,
        )
        if accelerator.is_main_process:
            summary["topk"] = topk_summary
            print("[topk summary]", json.dumps(topk_summary, indent=2))

    accelerator.wait_for_everyone()

    # Per-depth aggregates -- main process only, derived from the JSONLs.
    if accelerator.is_main_process:
        by_depth: dict = {}
        if not args.no_greedy and os.path.exists(greedy_jsonl):
            by_depth["greedy"] = aggregate_greedy_by_depth(greedy_jsonl)
        if not args.no_topk and os.path.exists(topk_grouped_jsonl):
            by_depth["topk"] = aggregate_topk_by_depth(topk_grouped_jsonl)
        summary["by_depth"] = by_depth

        summary_path = os.path.join(args.output_dir, "validation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        depth_path = os.path.join(args.output_dir, "validation_metrics_by_depth.json")
        with open(depth_path, "w") as f:
            json.dump(by_depth, f, indent=2)

        print("=" * 60)
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {depth_path}")
        print(f"Per-sample JSONLs under: {os.path.join(args.output_dir, 'per_sample')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
