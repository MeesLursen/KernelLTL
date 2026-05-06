"""Generate a fine-tuning dataset from cumulative curriculum stages via AST mutations."""

from __future__ import annotations

import argparse

from dataset_class import LTLDataset
from kernel_class import LTLKernel


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _nonnegative_int(value: str) -> int:
    ival = int(value)
    if ival < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer")
    return ival


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample stage-balanced base formulas from cumulative stage datasets and "
            "build a mutated fine-tuning dataset (equivalent rewrites + near-miss negations)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--kernel-dir", required=True, help="Directory containing a saved kernel")
    parser.add_argument(
        "--stage-train-dirs",
        nargs="+",
        required=True,
        help=(
            "Ordered cumulative training dataset directories (e.g. stage0/train ... stage4/train). "
            "Each stage is expected to include all previous stages."
        ),
    )
    parser.add_argument(
        "--exclude-dataset-dirs",
        nargs="*",
        default=[],
        help=(
            "Dataset directories whose formulas are excluded from the final mutated dataset "
            "(typically include stage4 eval)."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for the fine-tuning dataset")
    parser.add_argument("--sample-count", type=_positive_int, default=20000, help="Number of base formulas to sample")
    parser.add_argument(
        "--equivalent-mutations-per-formula",
        type=_nonnegative_int,
        default=2,
        help="Number of semantic-equivalent rewrites to apply per sampled base formula",
    )
    parser.add_argument(
        "--near-miss-mutations-per-formula",
        type=_nonnegative_int,
        default=1,
        help="Number of near-miss negation mutations to generate per sampled base formula",
    )
    parser.add_argument(
        "--satisfaction-batch-size",
        type=_positive_int,
        default=10240,
        help="Batch size used to evaluate satisfactions for mutated formulas",
    )
    parser.add_argument(
        "--satisfaction-time-index",
        type=int,
        default=0,
        help="Trace time index used for satisfaction extraction",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling/mutations")

    return parser.parse_args()


def _load_exclusion_set(dataset_dirs: list[str]) -> set[str]:
    excluded: set[str] = set()
    for path in dataset_dirs:
        ds = LTLDataset.load(path, load_satisfactions=False)
        for phi in ds.formulas:
            excluded.add(str(phi))
    return excluded


def main() -> None:
    args = parse_args()

    if (
        args.equivalent_mutations_per_formula == 0
        and args.near_miss_mutations_per_formula == 0
    ):
        raise ValueError("At least one of equivalent or near-miss mutation counts must be > 0")

    kernel = LTLKernel.load(args.kernel_dir)
    print(f"Loaded Kernel.")
    exclude_formula_strs = _load_exclusion_set(args.exclude_dataset_dirs)

    print(
        f"Building fine-tuning dataset from {len(args.stage_train_dirs)} stages, "
        f"sample_count={args.sample_count}, "
        f"equivalent_mutations_per_formula={args.equivalent_mutations_per_formula}, "
        f"near_miss_mutations_per_formula={args.near_miss_mutations_per_formula}"
    )
    if exclude_formula_strs:
        print(f"Excluding {len(exclude_formula_strs)} formulas from provided exclusion datasets")

    dataset = LTLDataset.construct_finetuning_mutation_dataset(
        kernel=kernel,
        stage_train_dirs=args.stage_train_dirs,
        sample_count=args.sample_count,
        equivalent_mutations_per_formula=args.equivalent_mutations_per_formula,
        near_miss_mutations_per_formula=args.near_miss_mutations_per_formula,
        exclude_formula_strs=exclude_formula_strs,
        store_formula_str=True,
        store_satisfaction=True,
        satisfaction_batch_size=args.satisfaction_batch_size,
        satisfaction_time_index=args.satisfaction_time_index,
        seed=args.seed,
    )

    dataset.save(args.output_dir)
    print(f"Saved fine-tuning dataset to {args.output_dir}")
    print(f"Final dataset size: {len(dataset)}")


if __name__ == "__main__":
    main()
