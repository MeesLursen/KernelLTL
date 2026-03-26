"""Utility for building and persisting an LTL kernel that can be reused across
curriculum-training stages."""

from __future__ import annotations

import argparse
import math
import os

from kernel_class import LTLKernel


def _positive_float(value: str) -> float:
    fval = float(value)
    if fval <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return fval


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct an LTL kernel (traces + anchor formulas) and save it to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir", required=True, help="Directory to store the serialized kernel")
    parser.add_argument("--trace-length", type=_positive_int, default=20, help="Maximum trace horizon T")
    parser.add_argument("--num-atomic-props", type=_positive_int, default=5, help="Number of atomic propositions (AP)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    trace_group = parser.add_argument_group("Trace sampling")
    trace_group.add_argument("--num-traces", type=_positive_int, default=None, help="Explicit number of traces to sample. Overrides eps/delta heuristic if provided.")
    trace_group.add_argument("--epsilon", type=_positive_float, default=0.01, help="Generalization gap tolerance used to compute trace budget when --num-traces is not provided")
    trace_group.add_argument("--delta", type=_positive_float, default=0.01, help="Failure probability used with --epsilon to compute trace budget")
    trace_group.add_argument("--trace-sampler", choices=["iid", "correlated"], default="correlated", help="Distribution used to sample traces")
    trace_group.add_argument("--low-variance-ratio", type=float, default=0.3, help="Ratio of low-variance traces (only used with correlated sampler)")
    trace_group.add_argument("--low-var-switch-prob", type=float, default=0.1, help="Switch probability inside low-variance segments (correlated sampler only)")

    anchor_group = parser.add_argument_group("Anchor formula sampling")
    anchor_group.add_argument("--anchor-count", type=_positive_int, default=1024, help="Number of anchor formulas (m)")
    anchor_group.add_argument("--anchor-sampler", choices=["uniform", "cosine"], default="cosine", help="Strategy used to populate anchor set")
    anchor_group.add_argument("--anchor-p-leaf", type=float, default=0.5, help="Probability that a syntax tree node becomes a leaf")
    anchor_group.add_argument("--anchor-max-depth", type=_positive_int, default=6, help="Maximum syntax tree depth for sampled formulas")
    anchor_group.add_argument("--anchor-force-tree", dest="anchor_force_tree", action="store_true", default=True, help="Ensure sampled formulas start with an operator (default: True)")
    anchor_group.add_argument("--no-anchor-force-tree", dest="anchor_force_tree", action="store_false", help="Allow root-level atomic propositions")
    anchor_group.add_argument("--cosine-batch-size", type=_positive_int, default=1024, help="Batch size used when evaluating cosine-controlled anchors")
    anchor_group.add_argument("--cosine-threshold", type=float, default=0.8, help="Threshold used when evaluating cosine-similarity anchors")
    anchor_group.add_argument("--cosine-max-attempts", type=_positive_int, default=100, help="Max attempts per anchor for cosine-controlled sampling")

    build_group = parser.add_argument_group("Feature matrix (F) construction")
    build_group.add_argument("--build-f-batch-size", type=_positive_int, default=1024, help="Batch size when evaluating anchors across traces")

    return parser.parse_args()


def _compute_trace_budget(anchor_count: int, epsilon: float, delta: float) -> int:
    return math.ceil((2.0 / (epsilon ** 2)) * math.log((2 * anchor_count) / delta))


def _sample_traces(kernel: LTLKernel, args: argparse.Namespace, num_traces: int) -> None:
    if args.trace_sampler == "correlated":
        kernel.sample_traces_kernel_correlated(
            num_traces,
            low_variance_ratio=args.low_variance_ratio,
            low_var_switch_prob=args.low_var_switch_prob,
        )
    else:
        kernel.sample_traces_kernel(num_traces)


def _sample_anchors(kernel: LTLKernel, args: argparse.Namespace) -> None:
    if args.anchor_sampler == "cosine":
        kernel.sample_anchor_formulas_kernel_cosine_controlled(
            m=args.anchor_count,
            p_leaf=args.anchor_p_leaf,
            max_depth=args.anchor_max_depth,
            force_tree=args.anchor_force_tree,
            batch_size=args.cosine_batch_size,
            threshold= args.cosine_threshold,
            max_attempts_per_formula=args.cosine_max_attempts,
        )
    else:
        kernel.sample_anchor_formulas_kernel(
            m=args.anchor_count,
            p_leaf=args.anchor_p_leaf,
            max_depth=args.anchor_max_depth,
            force_tree=args.anchor_force_tree,
        )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kernel = LTLKernel(T=args.trace_length, AP=args.num_atomic_props, seed=args.seed)

    if args.num_traces is not None:
        num_traces = args.num_traces
    else:
        num_traces = _compute_trace_budget(anchor_count=args.anchor_count, epsilon=args.epsilon, delta=args.delta)
    print(f"Sampling {num_traces} traces using '{args.trace_sampler}' strategy...")
    _sample_traces(kernel, args, num_traces)
    if kernel.traces is None:
        raise RuntimeError("Trace sampling failed: kernel.traces is None")
    print(f"Trace tensor shape: {tuple(kernel.traces.shape)}")

    print(f"Sampling {args.anchor_count} anchor formulas using '{args.anchor_sampler}' strategy...")
    _sample_anchors(kernel, args)
    if kernel.m is None or kernel.m == 0:
        raise RuntimeError("Anchor sampling failed: kernel anchor set is empty")

    print("Building feature matrix F...")
    kernel.build_F(batch_size=args.build_f_batch_size)
    if kernel.F is None:
        raise RuntimeError("Kernel feature matrix remained uninitialized")

    kernel.save(args.output_dir)
    print(f"Kernel persisted to {args.output_dir}")


if __name__ == "__main__":
    main()
