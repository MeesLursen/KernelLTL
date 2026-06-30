"""Greedy-only validation runner for the embedding-ablation floor (G1b).

The three ablation datasets (``validation_ablation_{zero,mean,shuffle}``) already
carry the CORRUPTED embeddings -- zero / non-trivial-mean / globally-shuffled,
computed over the non-trivial targets only -- while keeping every row (and so every
``formula_id``) aligned with the original validation set. This runner therefore
conditions on them as-is: no in-pipeline ablation, no top-K pass, and no
reference-model / KL comparison. It writes the same per-sample greedy JSONL and
per-depth summary that ``validate_model.py`` does, so the existing feasibility-floor
analysis consumes its output unchanged.

Usage (single GPU)::

    python scripts/validation_ablation.py \
        --kernel-dir <kernel> --tokenizer-dir <tokenizer> \
        --eval-dataset-dir <validation_ablation_zero> \
        --model-load-dir <ce_base_model> \
        --output-dir <out>

Usage (multi-GPU)::

    torchrun --nproc_per_node=N scripts/validation_ablation.py ...
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

from validation_utils import aggregate_greedy_by_depth, run_greedy_pass


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy-only validation over a pre-corrupted ablation dataset (G1b floor).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--eval-dataset-dir", required=True,
                        help="A validation_ablation_* dataset whose embeddings are already corrupted.")
    parser.add_argument("--model-load-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--per-device-eval-batch-size", type=_positive_int, default=96)
    parser.add_argument("--semantic-eval-batch-size", type=_positive_int, default=10240)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def _load_kernel(kernel_dir: str) -> LTLKernel:
    return LTLKernel.load(kernel_dir)


def _load_dataset(path: str) -> LTLDataset:
    dataset = LTLDataset.load(path, load_satisfactions=True)
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {path} is empty")
    if dataset.satisfactions is None:
        raise ValueError(
            f"Dataset at {path} is missing per-sample satisfaction tensors. "
            "Validation requires them for semantic distance / equivalence."
        )
    return dataset


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
        os.makedirs(os.path.join(args.output_dir, "per_sample"), exist_ok=True)

    kernel = _load_kernel(args.kernel_dir)
    tokenizer = LTLTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = _load_dataset(args.eval_dataset_dir)
    model = _build_model(args.model_load_dir, kernel)

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

    greedy_jsonl = os.path.join(args.output_dir, "per_sample", "greedy.jsonl")
    summary: dict = {
        "model_load_dir": args.model_load_dir,
        "eval_dataset_dir": args.eval_dataset_dir,
        "n_dataset_samples": len(dataset),
    }

    if accelerator.is_main_process:
        print("=" * 60)
        print("Running greedy ablation pass (no top-K, no reference)")
        print("=" * 60)

    greedy_summary = run_greedy_pass(
        model=model,
        ref_model_path=None,          # no reference / KL comparison for the floor
        eval_dataloader=dataloader,
        kernel=kernel,
        tokenizer=tokenizer,
        dataset=dataset,
        accelerator=accelerator,
        output_jsonl_path=greedy_jsonl,
        semantic_eval_batch_size=args.semantic_eval_batch_size,
        embedding_ablation="none",    # corruption is already baked into the dataset
        mean_embedding=None,
        ablation_seed=0,
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        summary["greedy"] = greedy_summary

        by_depth: dict = {}
        if os.path.exists(greedy_jsonl):
            by_depth["greedy"] = aggregate_greedy_by_depth(greedy_jsonl)
        summary["by_depth"] = by_depth

        summary_path = os.path.join(args.output_dir, "validation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        depth_path = os.path.join(args.output_dir, "validation_metrics_by_depth.json")
        with open(depth_path, "w") as f:
            json.dump(by_depth, f, indent=2)

        print("[greedy summary]", json.dumps(greedy_summary, indent=2))
        print("=" * 60)
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {depth_path}")
        print(f"Per-sample JSONL under: {os.path.join(args.output_dir, 'per_sample')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
