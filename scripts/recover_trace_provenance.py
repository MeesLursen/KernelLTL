"""One-off: recover which process generated each row of the thesis trace sample.

The thesis kernel's ``traces.pt`` is the trace sample drawn by the December 2025 kernel job
(job 17762902, partition gpu_a100) and reused unchanged by the July regeneration of the
anchors. It holds the output of ``formula_utils.sample_traces_correlated``, which ends in
``torch.unique(dim=0)``: the rows are sorted and deduplicated, and no record is kept of the
process that drew each row.

This script replays the December call sequence exactly,

    LTLKernel(T=20, AP=5, seed=1)    ->  torch.Generator(device="cuda").manual_seed(1)
    sample_traces_correlated(500000, n_ap=5, trace_length=20, rng, device="cuda",
                             low_variance_ratio=0.3, low_var_switch_prob=0.1)

(the traces were the first and only consumer of the generator before deduplication), keeps
the tensor from before deduplication, checks that its ``torch.unique`` equals the stored
traces bit for bit, and maps every stored row back to the draw(s) that produced it.

The CUDA random stream depends on the torch version and on the GPU model, so the replay
must run with torch 2.8.0 (pinned since 2025-10-10) on an A100, as in December. The
bit-for-bit comparison is the verdict: labels are written only when it holds.

Outputs, in --output-dir (which must not contain earlier outputs):

  trace_origin.npy        uint8 (N,), aligned with the rows of traces.pt. Bitmask of the
                          draws behind each stored row: 1 = all-false baseline,
                          2 = all-true baseline, 4 = Markov (low-variance) process,
                          8 = i.i.d. Bernoulli(0.5) process. A row drawn more than once
                          carries the OR of its sources.
  trace_multiplicity.npy  uint16 (N,), number of draws behind each stored row.
  report.json             environment, parameters, verdict, row counts per origin, the
                          duplicates, and the agreement with the flip-count rule.

On a mismatch only report.json is written (verdict "mismatch", with diagnostics), and the
script exits with status 2.

Usage (on a gpu_a100 node, from the repository root)::

    python -u scripts/recover_trace_provenance.py \
        --kernel-dir <artifacts>/kernel --output-dir <artifacts>/kernel_provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formula_utils import sample_traces_correlated  # noqa: E402

SOURCES = {1: "baseline_all_false", 2: "baseline_all_true", 4: "markov", 8: "iid"}
DEVICE = "cuda"  # the December draw ran on CUDA; overridden only by tests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kernel-dir", required=True,
                   help="Kernel whose traces.pt is to be explained (the thesis kernel).")
    p.add_argument("--output-dir", required=True,
                   help="Where the labels and the report are written.")
    dec = p.add_argument_group("The December call (job 17762902)")
    dec.add_argument("--seed", type=int, default=1)
    dec.add_argument("--num-traces", type=int, default=500_000)
    dec.add_argument("--num-atomic-props", type=int, default=5)
    dec.add_argument("--trace-length", type=int, default=20)
    dec.add_argument("--low-variance-ratio", type=float, default=0.3)
    dec.add_argument("--low-var-switch-prob", type=float, default=0.1)
    env = p.add_argument_group("Environment guards")
    env.add_argument("--expected-gpu", default="A100",
                     help="Substring the GPU name must contain.")
    env.add_argument("--expected-torch", default="2.8.0",
                     help="Prefix torch.__version__ must have.")
    env.add_argument("--skip-environment-guards", action="store_true",
                     help="Run on another GPU or torch version anyway (diagnostics only: "
                          "a mismatch is then expected).")
    return p.parse_args()


def check_environment(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        sys.exit("CUDA is required: the December traces were drawn with a CUDA generator.")
    props = torch.cuda.get_device_properties(0)
    env = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps(env, indent=2), flush=True)
    problems = []
    if args.expected_gpu not in props.name:
        problems.append(f"GPU is {props.name!r}, expected {args.expected_gpu!r}")
    if not torch.__version__.startswith(args.expected_torch):
        problems.append(f"torch is {torch.__version__}, expected {args.expected_torch}")
    if problems and not args.skip_environment_guards:
        sys.exit("; ".join(problems) + ". The CUDA random stream depends on both; "
                 "use --skip-environment-guards only for diagnostics.")
    return env


def replay_before_dedup(args: argparse.Namespace, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The body of ``formula_utils.sample_traces_correlated`` up to, not including, the
    ``torch.unique``: the same random calls with the same shapes in the same order. The
    Markov block draws its initial states, then one uniform per (trace, proposition) per
    later step; the i.i.d. block draws after it; the two baselines consume no randomness.

    Returns the tensor before deduplication and, per row, the bit of the draw behind it.
    """
    rng = torch.Generator(device=device).manual_seed(args.seed)  # as LTLKernel.__init__ did
    n, n_ap, T = args.num_traces, args.num_atomic_props, args.trace_length
    n_low = int(round(args.low_variance_ratio * n))
    n_high = max(0, n - n_low)

    low = torch.empty((n_low, n_ap, T), dtype=torch.bool, device=device)
    current = torch.randint(0, 2, (n_low, n_ap), generator=rng, dtype=torch.bool, device=device)
    low[:, :, 0] = current
    for t in range(1, T):
        flip = torch.rand((n_low, n_ap), generator=rng, device=device) < args.low_var_switch_prob
        current = torch.where(flip, torch.logical_not(current), current)
        low[:, :, t] = current
    high = torch.randint(0, 2, (n_high, n_ap, T), generator=rng, dtype=torch.bool, device=device)

    zeros = torch.zeros((1, n_ap, T), dtype=torch.bool, device=device)
    ones = torch.ones((1, n_ap, T), dtype=torch.bool, device=device)
    before = torch.cat((zeros, ones, low, high), dim=0)
    source = torch.cat((torch.tensor([1, 2], dtype=torch.uint8),
                        torch.full((n_low,), 4, dtype=torch.uint8),
                        torch.full((n_high,), 8, dtype=torch.uint8))).to(device)
    return before, source


def replay_via_repository(args: argparse.Namespace, device: str) -> torch.Tensor:
    """The repository function itself, on a fresh generator with the same seed."""
    rng = torch.Generator(device=device).manual_seed(args.seed)
    return sample_traces_correlated(args.num_traces,
                                    n_ap=args.num_atomic_props,
                                    trace_length=args.trace_length,
                                    rng=rng,
                                    device=device,
                                    low_variance_ratio=args.low_variance_ratio,
                                    low_var_switch_prob=args.low_var_switch_prob)


def flip_counts(traces: torch.Tensor) -> torch.Tensor:
    return (traces[:, :, 1:] != traces[:, :, :-1]).sum(dim=(1, 2))


def flip_rule(flips: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, int, float]:
    """Bayes rule Markov vs i.i.d. from the flip count alone, with the mixing share as prior.

    Under the i.i.d. process every trace has probability 2^-(AP*T); under the Markov process
    a trace with f flips has 2^-AP * p^f * (1-p)^(AP*(T-1)-f). Returns the per-row decision
    (True = Markov), the largest flip count classified as Markov, and the expected number of
    misclassified rows (the sum of the smaller posterior).
    """
    n_ap, T, p, lam = (args.num_atomic_props, args.trace_length,
                       args.low_var_switch_prob, args.low_variance_ratio)
    steps = n_ap * (T - 1)
    f = flips.astype(np.float64)
    log_markov = math.log(lam) - n_ap * math.log(2) + f * math.log(p) + (steps - f) * math.log(1 - p)
    log_iid = math.log(1 - lam) - n_ap * T * math.log(2)
    posterior_markov = 1.0 / (1.0 + np.exp(log_iid - log_markov))
    decision = posterior_markov > 0.5
    threshold = int(f[decision].max()) if decision.any() else -1
    return decision, threshold, float(np.minimum(posterior_markov, 1 - posterior_markov).sum())


def describe(bits: int) -> str:
    return "+".join(name for bit, name in SOURCES.items() if bits & bit)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mismatch_diagnostics(replayed: torch.Tensor, stored: torch.Tensor) -> dict:
    out = {"replayed_shape": list(replayed.shape), "stored_shape": list(stored.shape)}
    if replayed.shape[1:] == stored.shape[1:]:
        rep = {bytes(r) for r in replayed.reshape(replayed.size(0), -1).numpy().astype(np.uint8)}
        sto = [bytes(r) for r in stored.reshape(stored.size(0), -1).numpy().astype(np.uint8)]
        out["stored_rows_found_in_replay"] = int(sum(r in rep for r in sto))
    return out


def main() -> None:
    args = parse_args()
    env = check_environment(args)
    device = DEVICE

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    targets = ["trace_origin.npy", "trace_multiplicity.npy", "report.json"]
    existing = [t for t in targets if os.path.exists(os.path.join(out_dir, t))]
    if existing:
        sys.exit(f"{out_dir} already holds {existing}; refusing to overwrite.")

    stored = torch.load(os.path.join(args.kernel_dir, "traces.pt"), map_location="cpu")
    print(f"stored traces: {tuple(stored.shape)} {stored.dtype}", flush=True)

    before, source = replay_before_dedup(args, device)
    flat = before.reshape(before.size(0), -1)
    uniq_flat, inverse = torch.unique(flat, dim=0, return_inverse=True)
    uniq = uniq_flat.reshape(-1, args.num_atomic_props, args.trace_length).cpu()
    via_repo = replay_via_repository(args, device).cpu()

    checks = {
        "replay_equals_stored": bool(uniq.shape == stored.shape and torch.equal(uniq, stored)),
        "repository_function_equals_stored": bool(via_repo.shape == stored.shape
                                                  and torch.equal(via_repo, stored)),
        "replay_equals_repository_function": bool(uniq.shape == via_repo.shape
                                                  and torch.equal(uniq, via_repo)),
    }
    print(json.dumps(checks, indent=2), flush=True)
    params = {k: v for k, v in vars(args).items() if k not in ("kernel_dir", "output_dir")}
    report = {"environment": env, "parameters": params,
              "kernel_dir": os.path.abspath(args.kernel_dir), "checks": checks}

    if not all(checks.values()):
        report["verdict"] = "mismatch"
        report["diagnostics"] = mismatch_diagnostics(uniq, stored)
        with open(os.path.join(out_dir, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps(report["diagnostics"], indent=2))
        print("MISMATCH: no labels written.")
        sys.exit(2)

    n_unique = uniq.size(0)
    origin = torch.zeros(n_unique, dtype=torch.uint8, device=device)
    for bit in SOURCES:
        hit = torch.zeros(n_unique, dtype=torch.bool, device=device)
        hit[inverse[source == bit]] = True
        origin |= hit.to(torch.uint8) * bit
    multiplicity = torch.bincount(inverse, minlength=n_unique)
    origin, multiplicity = origin.cpu().numpy(), multiplicity.cpu().numpy()
    if multiplicity.max() > np.iinfo(np.uint16).max:
        sys.exit("multiplicity does not fit in uint16")

    flips = flip_counts(stored).numpy()
    rule_markov, threshold, expected_errors = flip_rule(flips, args)
    drawn_markov_only = origin == 4
    drawn_iid_only = origin == 8
    disagree = np.flatnonzero((rule_markov & drawn_iid_only) | (~rule_markov & drawn_markov_only))

    patterns, counts = np.unique(origin, return_counts=True)
    duplicates = np.flatnonzero(multiplicity > 1)
    report.update({
        "verdict": "reproduced",
        "rows_before_dedup": int(before.size(0)),
        "rows_after_dedup": int(n_unique),
        "rows_by_origin": {describe(int(b)): int(c) for b, c in zip(patterns, counts)},
        "duplicates": [{"row": int(i), "multiplicity": int(multiplicity[i]),
                        "origin": describe(int(origin[i])), "flips": int(flips[i])}
                       for i in duplicates],
        "flip_rule": {
            "markov_iff_flips_at_most": threshold,
            "expected_misclassified_rows": expected_errors,
            "rows_with_single_process_origin_misclassified": [
                {"row": int(i), "origin": describe(int(origin[i])), "flips": int(flips[i])}
                for i in disagree],
        },
    })

    np.save(os.path.join(out_dir, "trace_origin.npy"), origin.astype(np.uint8))
    np.save(os.path.join(out_dir, "trace_multiplicity.npy"), multiplicity.astype(np.uint16))
    report["files"] = {name: sha256(os.path.join(out_dir, name))
                       for name in ("trace_origin.npy", "trace_multiplicity.npy")}
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: report[k] for k in ("verdict", "rows_before_dedup", "rows_after_dedup",
                                             "rows_by_origin", "duplicates", "flip_rule")},
                     indent=2))


if __name__ == "__main__":
    main()
