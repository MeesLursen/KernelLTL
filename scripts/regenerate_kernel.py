"""Regenerate a kernel's anchor set and feature matrix, reusing an existing kernel's traces.

Motivation
----------
The original anchor set was selected with an upper-bound-only similarity gate, which
admits a formula together with its (near-)negation and thereby wastes anchor dimensions.
This script resamples the anchor set under the *symmetric* Hamming-band gate
(``|sim^pm| <= tau``) with explicit trivial rejection (see
``LTLKernel.sample_anchor_formulas_kernel_cosine_controlled``) and rebuilds the feature
matrix ``F``.

Crucially, it *reuses the traces of the source kernel* (they are never resampled), so every
stored satisfaction vector in the datasets remains valid and only the embeddings change.
Because ``F`` is boolean-derived (0/1) and embeddings are recomputed exactly from ``F`` and
the stored satvecs (see ``LTLKernel._covariance_embeddings``), the resulting embeddings are
bit-reproducible on any IEEE-754 hardware.

Reproducibility note
--------------------
Anchor *selection* is stochastic rejection sampling whose accept/reject decisions use a
float32 signed-dot similarity; the accepted set is therefore reproducible bit-for-bit only
on fixed hardware. We therefore treat the *saved* kernel (anchor_formulas.jsonl + F.pt) as
the canonical artifact. Downstream embedding reproducibility depends only on the saved F and
the stored satvecs, both of which are exact.

Resumability
------------
Resumable per object using the saved-kernel state (``metadata.json`` acts as the sentinel):
  * if the output dir has anchors but no F, anchors are reused and only F is rebuilt;
  * if the output dir has F, the run is already complete.
Delete the output dir to force a clean rerun.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_class import LTLKernel


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _unit_float(value: str) -> float:
    fval = float(value)
    if not (0.0 < fval < 1.0):
        raise argparse.ArgumentTypeError("Value must lie strictly in (0, 1)")
    return fval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate anchors (symmetric Hamming-band gate) + F, reusing source-kernel traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-kernel-dir", required=True, help="Existing kernel whose traces (and RNG seed) are reused")
    parser.add_argument("--output-dir", required=True, help="Directory for the regenerated kernel")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda, cpu)")

    anchor = parser.add_argument_group("Anchor sampling")
    anchor.add_argument("--anchor-count", type=_positive_int, default=1024, help="Number of anchors m")
    anchor.add_argument("--threshold", type=_unit_float, default=0.6, help="Similarity band half-width tau")
    anchor.add_argument("--anchor-max-depth", type=_positive_int, default=6, help="Maximum candidate syntax-tree depth")
    anchor.add_argument("--anchor-p-leaf-range", nargs=2, type=float, default=[0.4, 0.6], help="Leaf-probability range (low high)")
    anchor.add_argument("--no-force-tree", dest="force_tree", action="store_false", default=True, help="Allow root-level atoms")
    anchor.add_argument("--cosine-batch-size", type=_positive_int, default=1024, help="Batch size for candidate satvec evaluation")
    anchor.add_argument("--max-attempts", type=_positive_int, default=500, help="Max attempts per anchor before failing")
    anchor.add_argument("--seed", type=int, default=None, help="RNG seed for anchor sampling (default: source kernel seed)")

    parser.add_argument("--build-f-batch-size", type=_positive_int, default=1024, help="Batch size when evaluating anchors across traces")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    resume_meta = os.path.join(args.output_dir, "metadata.json")
    if os.path.exists(resume_meta):
        kernel = LTLKernel.load(args.output_dir, device=args.device)
        print(
            f"[resume] Loaded in-progress kernel from {args.output_dir}: "
            f"{len(kernel.anchor_formulas)} anchors, F {'present' if kernel.F is not None else 'absent'}."
        )
    else:
        kernel = LTLKernel.load(args.source_kernel_dir, device=args.device)
        if kernel.traces is None:
            raise ValueError(f"Source kernel at {args.source_kernel_dir} has no traces to reuse.")
        # Reset the anchor set / F; keep the source traces untouched.
        kernel.anchor_formulas = []
        kernel.m = None
        kernel.F = None
        seed = args.seed if args.seed is not None else kernel.seed
        if seed is not None:
            kernel.rng.manual_seed(int(seed))
        print(
            f"[start] Source traces reused: N={kernel.traces.size(0)} on {kernel.device}. "
            f"Anchor RNG seeded with {seed}."
        )

    # ---- Object 1: anchor set ----
    if kernel.m is None or len(kernel.anchor_formulas) == 0:
        print(f"[anchors] Sampling {args.anchor_count} anchors at tau={args.threshold}, max_depth={args.anchor_max_depth} ...")
        kernel.sample_anchor_formulas_kernel_cosine_controlled(
            m=args.anchor_count,
            p_leaf_range=tuple(args.anchor_p_leaf_range),
            max_depth=args.anchor_max_depth,
            force_tree=args.force_tree,
            batch_size=args.cosine_batch_size,
            threshold=args.threshold,
            max_attempts_per_formula=args.max_attempts,
        )
        kernel.save(args.output_dir)  # checkpoint: anchors + traces + rng saved (F still None)
        print(f"[anchors] Checkpoint saved ({kernel.m} anchors); F not yet built.")
    else:
        print(f"[anchors] Reusing {len(kernel.anchor_formulas)} anchors from checkpoint.")

    # ---- Object 2: feature matrix F ----
    if kernel.F is None:
        print(f"[F] Building feature matrix over {kernel.traces.size(0)} traces ...")
        kernel.build_F(batch_size=args.build_f_batch_size)
        if kernel.F is None:
            raise RuntimeError("Feature matrix remained uninitialized after build_F().")
        kernel.save(args.output_dir)  # checkpoint: F saved
        print(f"[F] Checkpoint saved: F shape {tuple(kernel.F.shape)}.")
    else:
        print("[F] Reusing F from checkpoint.")

    print(f"[done] Regenerated kernel persisted to {args.output_dir} (m={kernel.m}, N={kernel.traces.size(0)}).")


if __name__ == "__main__":
    main()
