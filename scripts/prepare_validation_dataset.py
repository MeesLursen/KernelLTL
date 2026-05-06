"""Generate a validation set disjoint from provided datasets/formula lists."""

from __future__ import annotations

import argparse

from dataset_class import LTLDataset
from kernel_class import LTLKernel


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _parse_depth_target(token: str) -> tuple[int, int]:
    try:
        depth_text, count_text = token.split(":", maxsplit=1)
        depth = int(depth_text)
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid depth target '{token}'. Expected format depth:count (e.g. 2:5000)."
        ) from exc

    if depth < 0:
        raise argparse.ArgumentTypeError(f"Depth must be >= 0 in '{token}'")
    if count <= 0:
        raise argparse.ArgumentTypeError(f"Count must be > 0 in '{token}'")
    return depth, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validation dataset from kernel while excluding known formula sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kernel-dir", required=True, help="Path to saved kernel")
    parser.add_argument(
        "--exclude-dataset-dirs",
        nargs="+",
        required=True,
        help=(
            "Dataset directories to exclude from validation sampling "
            "(for example stage4 train, stage4 eval, finetune train)."
        ),
    )
    parser.add_argument(
        "--exclude-formula-files",
        nargs="*",
        default=[],
        help="Optional text files with one canonical formula string per line to exclude.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save the validation set")
    parser.add_argument("--k", type=_positive_int, default=5000, help="Number of formulas to sample")
    parser.add_argument(
        "--depth-targets",
        nargs="*",
        default=None,
        help=(
            "Optional per-depth quotas in depth:count format. "
            "Example: --depth-targets 2:5000 3:5000 4:5000 5:5000"
        ),
    )
    parser.add_argument(
        "--p-leaf-range",
        nargs=2,
        type=float,
        default=[0.1, 0.5],
        help="Leaf-probability range for kernel sampling",
    )
    parser.add_argument("--max-depth", type=_positive_int, default=5, help="Maximum depth to sample")
    parser.add_argument("--min-depth", type=int, default=None, help="Optional minimum depth to keep")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicates within the sampled validation dataset",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=10240, help="Satisfaction batch size")
    parser.add_argument("--time-index", type=int, default=0, help="Satisfaction time index")
    parser.add_argument(
        "--max-sampling-attempts",
        type=_positive_int,
        default=100,
        help="Maximum resampling rounds before failing",
    )
    return parser.parse_args()


def _build_exclusion_set(dataset_dirs: list[str], formula_files: list[str]) -> set[str]:
    excluded: set[str] = set()
    for path in dataset_dirs:
        ds = LTLDataset.load(path, load_satisfactions=False)
        excluded.update(str(phi) for phi in ds.formulas)

    for path in formula_files:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if text:
                    excluded.add(text)

    return excluded


def main() -> None:
    args = parse_args()
    kernel = LTLKernel.load(args.kernel_dir)

    depth_targets: dict[int, int] | None = None
    effective_k = args.k
    if args.depth_targets:
        depth_targets = {}
        for token in args.depth_targets:
            depth, count = _parse_depth_target(token)
            depth_targets[depth] = depth_targets.get(depth, 0) + count
        effective_k = sum(depth_targets.values())
        print(f"Using per-depth targets {depth_targets} (effective k={effective_k})")

    exclude_formula_strs = _build_exclusion_set(args.exclude_dataset_dirs, args.exclude_formula_files)
    print(f"Loaded {len(exclude_formula_strs)} formulas to exclude")

    dataset = LTLDataset.construct_dataset_from_kernel_excluding(
        kernel=kernel,
        k=effective_k,
        p_leaf_range=tuple(args.p_leaf_range),
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        depth_targets=depth_targets,
        exclude_formula_strs=exclude_formula_strs,
        dedupe=not args.allow_duplicates,
        store_formula_str=True,
        store_satisfaction=True,
        satisfaction_batch_size=args.batch_size,
        satisfaction_time_index=args.time_index,
        max_sampling_attempts=args.max_sampling_attempts,
    )

    dataset.save(args.output_dir)
    print(f"Saved validation dataset to {args.output_dir} with {len(dataset)} formulas")


if __name__ == "__main__":
    main()