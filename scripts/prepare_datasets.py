"""Generate and persist train/eval datasets derived from a saved LTL kernel."""

from __future__ import annotations

import argparse
import os
from typing import Callable

from dataset_class import LTLDataset
from kernel_class import LTLKernel


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _ratio(value: str) -> float:
    fval = float(value)
    if not (0.0 < fval < 1.0):
        raise argparse.ArgumentTypeError("Value must be between 0 and 1 (exclusive)")
    return fval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample formula datasets (train/eval) from a persisted kernel and save them for curriculum stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--kernel-dir", required=True, help="Directory containing a previously saved kernel")
    
    # Disjoint split mode
    parser.add_argument(
        "--disjoint-split",
        action="store_true",
        help="Use disjoint splitting: sample once and split into train/eval with no formula overlap (uses kernel's RNG for reproducibility)"
    )
    parser.add_argument(
        "--eval-ratio",
        type=_ratio,
        default=0.05,
        help="Target fraction of formulas for evaluation when using --disjoint-split (default: 0.05 = 5%%)"
    )

    # Train dataset options
    train = parser.add_argument_group("Training dataset")
    train.add_argument("--train-out", required=True, help="Output directory for the training dataset")
    train.add_argument("--train-k", type=_positive_int, required=True, help="Number of formulas to sample for training (or total when using --disjoint-split)")
    train.add_argument("--train-p-leaf-range", nargs=2, type=float, help="Probability that a sampled node becomes a leaf")
    train.add_argument("--train-max-depth", type=_positive_int, help="Maximum tree depth for training formulas")
    train.add_argument("--train-dedupe", action="store_true", help="Deduplicate formulas before computing embeddings")
    train.add_argument("--train-store-formula-str", action="store_true", help="Persist canonical formula strings in the dataset")
    train.add_argument("--train-store-satisfaction", action="store_true", help="Persist satisfaction tensors in the dataset")
    train.add_argument("--train-satisfaction-batch-size", type=_positive_int, default=1024, help="Batch size used when recording satisfactions")
    train.add_argument("--train-satisfaction-time-index", type=int, default=0, help="Trace time index used when recording satisfactions")

    # Eval dataset options (optional)
    eval_group = parser.add_argument_group("Evaluation dataset")
    eval_group.add_argument("--eval-out", help="Output directory for the evaluation dataset")
    eval_group.add_argument("--eval-k", type=_positive_int, help="Number of formulas for eval dataset (ignored when using --disjoint-split)")
    eval_group.add_argument("--eval-p-leaf-range", nargs=2, type=float, help="Probability that a sampled node becomes a leaf (ignored when using --disjoint-split)")
    eval_group.add_argument("--eval-max-depth", type=_positive_int, help="Maximum tree depth for eval formulas (ignored when using --disjoint-split)")
    eval_group.add_argument("--eval-dedupe", action="store_true", help="Deduplicate formulas before computing embeddings")
    eval_group.add_argument("--eval-store-formula-str", dest="eval_store_formula_str", action="store_true", help="Persist canonical formula strings in the eval dataset")
    eval_group.add_argument("--no-eval-store-formula-str", dest="eval_store_formula_str", action="store_false", help="Disable formula string storage for eval dataset")
    eval_group.add_argument("--eval-store-satisfaction", dest="eval_store_satisfaction", action="store_true", help="Persist evaluation satisfactions")
    eval_group.add_argument("--no-eval-store-satisfaction", dest="eval_store_satisfaction", action="store_false", help="Disable satisfaction storage for eval dataset")
    eval_group.add_argument("--eval-satisfaction-batch-size", type=_positive_int, default=1024, help="Batch size used when recording eval satisfactions")
    eval_group.add_argument("--eval-satisfaction-time-index", type=int, default=0, help="Trace time index used when recording eval satisfactions")
    parser.set_defaults(eval_store_formula_str=True, eval_store_satisfaction=True)

    return parser.parse_args()


def _build_dataset(
    name: str,
    kernel: LTLKernel,
    out_dir: str,
    k: int,
    p_leaf_range: tuple[float,float],
    max_depth: int,
    dedupe: bool,
    store_formula_str: bool,
    store_satisfaction: bool,
    satisfaction_batch_size: int,
    satisfaction_time_index: int,
) -> None:
    print(f"\nBuilding {name} dataset with k={k}, p_leaf_range={p_leaf_range}, max_depth={max_depth}, dedupe={dedupe}")
    os.makedirs(out_dir, exist_ok=True)

    dataset = LTLDataset(
        store_formula_str=store_formula_str,
        store_satisfaction=store_satisfaction,
        satisfaction_batch_size=satisfaction_batch_size,
        satisfaction_time_index=satisfaction_time_index,
    )

    construct_fn: Callable[..., None]
    if dedupe:
        construct_fn = dataset.construct_dataset_from_kernel_dedupe
    else:
        construct_fn = dataset.construct_dataset_from_kernel

    construct_fn(
        kernel=kernel,
        k=k,
        p_leaf_range=p_leaf_range,
        max_depth=max_depth,
    )

    dataset.save(out_dir)
    print(f"Saved {name} dataset to {out_dir}")


def main() -> None:
    args = parse_args()
    kernel = LTLKernel.load(args.kernel_dir)
    train_p_leaf_range = tuple(args.train_p_leaf_range)
    eval_p_leaf_range = tuple(args.eval_p_leaf_range)
    if kernel.traces is None or kernel.F is None:
        raise RuntimeError(
            "Loaded kernel is missing traces or feature matrix F. Rerun prepare_kernel.py with --build-f options before generating datasets."
        )

    if args.disjoint_split:
        # Use disjoint splitting mode
        if not args.eval_out:
            raise ValueError("--eval-out is required when using --disjoint-split")
        
        print(f"\nBuilding disjoint train/eval datasets with k={args.train_k}, "
              f"p_leaf_range={args.train_p_leaf_range}, max_depth={args.train_max_depth}, "
              f"eval_ratio={args.eval_ratio}")
        
        os.makedirs(args.train_out, exist_ok=True)
        os.makedirs(args.eval_out, exist_ok=True)
        
        train_dataset, eval_dataset = LTLDataset.construct_disjoint_datasets(
            kernel=kernel,
            k=args.train_k,
            p_leaf_range=train_p_leaf_range,
            max_depth=args.train_max_depth,
            eval_ratio=args.eval_ratio,
            store_formula_str_train=args.train_store_formula_str,
            store_formula_str_eval=args.eval_store_formula_str,
            store_satisfaction_train=args.train_store_satisfaction,
            store_satisfaction_eval=args.eval_store_satisfaction,
            satisfaction_batch_size=args.train_satisfaction_batch_size,
            satisfaction_time_index=args.train_satisfaction_time_index,
            dedupe_eval=args.eval_dedupe,
        )
        
        train_dataset.save(args.train_out)
        print(f"Saved training dataset to {args.train_out}")
        
        eval_dataset.save(args.eval_out)
        print(f"Saved evaluation dataset to {args.eval_out}")
    else:
        # Original independent sampling mode
        _build_dataset(
            name="training",
            kernel=kernel,
            out_dir=args.train_out,
            k=args.train_k,
            p_leaf_range=train_p_leaf_range,
            max_depth=args.train_max_depth,
            dedupe=args.train_dedupe,
            store_formula_str=args.train_store_formula_str,
            store_satisfaction=args.train_store_satisfaction,
            satisfaction_batch_size=args.train_satisfaction_batch_size,
            satisfaction_time_index=args.train_satisfaction_time_index,
        )

        if args.eval_out and args.eval_k:
            _build_dataset(
                name="evaluation",
                kernel=kernel,
                out_dir=args.eval_out,
                k=args.eval_k,
                p_leaf_range=eval_p_leaf_range,
                max_depth=args.eval_max_depth,
                dedupe=args.eval_dedupe,
                store_formula_str=args.eval_store_formula_str,
                store_satisfaction=args.eval_store_satisfaction,
                satisfaction_batch_size=args.eval_satisfaction_batch_size,
                satisfaction_time_index=args.eval_satisfaction_time_index,
            )
        elif args.eval_out or args.eval_k:
            raise ValueError("Both --eval-out and --eval-k must be provided together to build the eval dataset.")


if __name__ == "__main__":
    main()
