"""Experiment 3 analysis driver: does the decoder read the embedding norm?

Test 1 (observational) needs the conditioned greedy run, the Experiment 2
features, and the kernel's trace sample; Test 2 (interventional) needs the
rescaling sweep. Run either or both.

Outputs (under --output-dir):

  test1_per_target.csv     formula_id, vbin, u, variance, delta_variance, flags
  test1_per_bin.csv        per (subset, bin): n, mean_variance, slope [CI], spearman
  test1_pooled.csv         per subset: pooled within-bin slope [CI], unadjusted and adjusted
                           for depth + operators; interaction fit and its crossover
  test1_by_range.csv       per (subset, target-variance quartile): pooled slope, both arms
  test1_prior.json         V_0 from the zero-ablation run (if given)
  test2_per_scale.csv      per scale: the six curves with CIs
  test2_per_target.csv     per target: rho(c, V_gen), invariance flag
  test2_monotonicity.csv   mean rho, shares increasing / decreasing / invariant [CIs]
  test2_by_tercile.csv     per (target-variance tercile, scale) curves
  checks.csv               Experiment 1 consistency checks on every run read
  manifest.json            inputs, sizes, bootstrap settings, trace-gate result
  figures/                 see plot_exp3.py

Usage::

    python scripts/analysis_exp3/run_exp3.py \
        --test both \
        --features-dir <artifacts>/analysis/exp2/features \
        --dataset-dir <artifacts>/datasets/validation \
        --run-dir <validation_root>/ce_base \
        --zero-run-dir <validation_root>/ce_base_ablation_zero \
        --kernel-dir <artifacts>/kernel \
        --rescaling-dir <validation_root>/ce_base_rescaling \
        --output-dir <artifacts>/analysis/exp3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "analysis_exp1"))
sys.path.insert(0, str(_HERE.parent / "analysis_exp2"))
from bootstrap import DEFAULT_B, DEFAULT_SEED, index_matrix                      # noqa: E402
from load import load_greedy, read_dataset_size                                  # noqa: E402
from frame import DEFAULT_N_BINS                                                 # noqa: E402
import gen_variance                                                              # noqa: E402
import plot_exp3                                                                 # noqa: E402
import test1                                                                     # noqa: E402
import test2                                                                     # noqa: E402
from records import load_features, load_rescaling, read_manifest, summarise_checks  # noqa: E402


def _log(msg: str) -> None:
    print(f"[exp3 {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--test", choices=("1", "2", "both"), default="both")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Validation dataset dir (metadata.json gives the expected target count "
                        "and the satisfaction time index).")
    # Test 1
    p.add_argument("--features-dir", type=Path, default=None, help="Dir with exp2_features.csv")
    p.add_argument("--run-dir", type=Path, default=None, help="Conditioned run (ce_base)")
    p.add_argument("--zero-run-dir", type=Path, default=None, help="Zero-ablation run, for V_0")
    p.add_argument("--kernel-dir", type=Path, default=None,
                   help="Kernel dir holding traces.pt (or the traces.pt path itself)")
    p.add_argument("--time-index", type=int, default=None,
                   help="Evaluation time index; default from the dataset metadata, else 0")
    p.add_argument("--eval-batch-size", type=int, default=16384)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verify-sample", type=int, default=32,
                   help="Targets whose base rate is recomputed to gate the trace sample (0 disables)")
    p.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    # Test 2
    p.add_argument("--rescaling-dir", type=Path, default=None,
                   help="Output dir of scripts/validation_variance_rescaling.py")
    # Bootstrap
    p.add_argument("--bootstrap-samples", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def _require(args: argparse.Namespace, names: list[str], test: str) -> None:
    missing = [n for n in names if getattr(args, n.replace("-", "_")) is None]
    if missing:
        raise SystemExit(f"test {test} needs --{' --'.join(missing)}")


def _time_index(args: argparse.Namespace) -> int:
    if args.time_index is not None:
        return int(args.time_index)
    if args.dataset_dir is not None:
        meta = json.loads((args.dataset_dir / "metadata.json").read_text())
        return int(meta.get("satisfaction_time_index", 0))
    return 0


def run_test1(args: argparse.Namespace, out: Path, manifest: dict, checks: list[dict]) -> None:
    _require(args, ["features-dir", "run-dir", "kernel-dir"], "1")
    expected_n = read_dataset_size(args.dataset_dir)
    time_index = _time_index(args)

    features = load_features(args.features_dir)
    greedy, c = load_greedy(args.run_dir, expected_n=expected_n)
    checks.extend(c)
    if set(features["formula_id"]) != set(greedy["formula_id"]):
        raise ValueError("formula_id sets differ between features and the greedy run")

    _log(f"loading traces from {args.kernel_dir} onto {args.device}")
    traces = gen_variance.load_traces(args.kernel_dir, device=args.device)
    gate = gen_variance.verify_trace_sample(
        features, greedy, traces, time_index=time_index, batch_size=args.eval_batch_size,
        n_sample=args.verify_sample, seed=args.seed)
    _log(f"trace-sample gate: {gate}")

    df = gen_variance.add_generated_variance(
        greedy, features, traces, time_index=time_index, batch_size=args.eval_batch_size, log=_log)
    df = df.merge(features[["formula_id", "emb_norm", "relational_faithfulness", "variance"]],
                  on="formula_id", validate="one_to_one")

    v0_info: dict | None = None
    if args.zero_run_dir is not None:
        zero, cz = load_greedy(args.zero_run_dir, expected_n=expected_n)
        checks.extend(cz)
        v0_info = gen_variance.prior_variance(
            zero, traces, time_index=time_index, batch_size=args.eval_batch_size)
        _log(f"prior under the zero embedding: {v0_info}")
        (out / "test1_prior.json").write_text(json.dumps(v0_info, indent=2))

    idx = index_matrix(len(df), b=args.bootstrap_samples, seed=args.seed)
    res = test1.run_test1(df, idx=idx, n_bins=args.n_bins,
                          v0=(v0_info["V0"] if v0_info else None), log=_log)
    res["per_target"].to_csv(out / "test1_per_target.csv", index=False)
    res["per_bin"].to_csv(out / "test1_per_bin.csv", index=False)
    res["pooled"].to_csv(out / "test1_pooled.csv", index=False)
    res["by_range"].to_csv(out / "test1_by_range.csv", index=False)

    manifest["test1"] = {
        "features_dir": str(args.features_dir), "run_dir": str(args.run_dir),
        "zero_run_dir": (str(args.zero_run_dir) if args.zero_run_dir else None),
        "kernel_dir": str(args.kernel_dir), "time_index": time_index,
        "n_targets": int(len(df)), "n_bins": args.n_bins,
        "n_misses": int(test1.subset_mask(df, "misses").sum()),
        "n_valid": int(test1.subset_mask(df, "valid").sum()),
        "trace_gate": gate, "prior": v0_info, "adjustment": res["adjustment"],
    }
    show = ["subset", "n", "slope", "slope_lo", "slope_hi", "slope_adj", "slope_adj_lo", "slope_adj_hi",
            "crossover", "crossover_adj"]
    _log("test1 pooled within-bin slope of delta on u:\n" + res["pooled"][show].to_string(index=False))
    _log("test1 by target-variance range:\n" + res["by_range"].to_string(index=False))


def run_test2(args: argparse.Namespace, out: Path, manifest: dict, checks: list[dict]) -> None:
    _require(args, ["rescaling-dir"], "2")
    expected_n = read_dataset_size(args.dataset_dir)
    long, c = load_rescaling(args.rescaling_dir, expected_n=expected_n)
    checks.extend(c)
    n_targets = int(long["formula_id"].nunique())
    idx = index_matrix(n_targets, b=args.bootstrap_samples, seed=args.seed)
    res = test2.run_test2(long, idx=idx, log=_log)
    res["per_scale"].to_csv(out / "test2_per_scale.csv", index=False)
    res["per_target"].to_csv(out / "test2_per_target.csv", index=False)
    res["monotonicity"].to_csv(out / "test2_monotonicity.csv", index=False)
    res["by_tercile"].to_csv(out / "test2_by_tercile.csv", index=False)
    manifest["test2"] = {
        "rescaling_dir": str(args.rescaling_dir),
        "rescaling_manifest": read_manifest(args.rescaling_dir),
        "n_targets": n_targets, "scales": sorted(float(s) for s in long["scale"].unique()),
    }
    _log("test2 monotonicity:\n" + res["monotonicity"].to_string(index=False))
    _log("test2 per scale:\n" + res["per_scale"][
        ["scale", "validity", "equivalence", "generated_variance", "delta_variance",
         "direction_cosine", "entropy"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "dataset_dir": (str(args.dataset_dir) if args.dataset_dir else None),
    }
    checks: list[dict] = []

    if args.test in ("1", "both"):
        run_test1(args, out, manifest, checks)
    if args.test in ("2", "both"):
        run_test2(args, out, manifest, checks)

    summarise_checks(checks).to_csv(out / "checks.csv", index=False)
    bad = [c for c in checks if c["violations"]]
    if bad:
        _log(f"WARNING: {len(bad)} consistency checks with violations -- see checks.csv")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if not args.no_figures:
        figs = out / "figures"
        figs.mkdir(exist_ok=True)
        if (out / "test1_per_bin.csv").exists():
            plot_exp3.fig_test1_slopes(pd.read_csv(out / "test1_per_bin.csv"),
                                       pd.read_csv(out / "test1_pooled.csv"), figs)
            plot_exp3.fig_test1_scatter(pd.read_csv(out / "test1_per_target.csv"),
                                        pd.read_csv(out / "test1_pooled.csv"), figs)
            plot_exp3.fig_test1_by_range(pd.read_csv(out / "test1_by_range.csv"), figs)
        if (out / "test2_per_scale.csv").exists():
            plot_exp3.fig_test2_curves(pd.read_csv(out / "test2_per_scale.csv"), figs)
            plot_exp3.fig_test2_rho(pd.read_csv(out / "test2_per_target.csv"),
                                    pd.read_csv(out / "test2_monotonicity.csv"), figs)
        _log(f"figures under {figs}")
    _log(f"done -> {out}")


if __name__ == "__main__":
    main()
