"""Experiment 1 analysis driver: raw validation JSONLs -> statistical tables.

Consumes the per-generation records of one conditioned validation run (greedy
+ top-K passes) and, optionally, the three embedding-ablation runs (greedy
only). Emits tidy CSV tables carrying point estimates with 95% percentile-
bootstrap intervals -- everything the local data-viz layer and the thesis
tables need, and nothing else. No plotting here.

Outputs (under --output-dir):
    overall.csv     pass x metric                (thesis tab:exp1_results_overall)
    by_depth.csv    pass x metric x target_depth (tab:exp1_results_by_depth and
                                                  the by-depth figures)
    pass_at_k.csv   k x estimate                 (fig:exp1_pass_at_k)
    ablation.csv    condition x metric           (tab:exp1_ablation)
    checks.csv      consistency-check violation counts
    manifest.json   inputs, settings, thesis-object -> (file, filter) map

Usage::

    python scripts/analysis_exp1/run_exp1.py \
        --run-dir <validation_root>/ce_base \
        --ablation-dir zero=<validation_root>/ce_base_ablation_zero \
        --ablation-dir mean=<validation_root>/ce_base_ablation_mean \
        --ablation-dir shuffle=<validation_root>/ce_base_ablation_shuffle \
        --dataset-dir <artifacts>/datasets/validation \
        --output-dir <artifacts>/analysis/exp1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import DEFAULT_B, DEFAULT_SEED, index_matrix, mean_ci          # noqa: E402
from load import (infer_special_ids, load_greedy, load_topk,                  # noqa: E402
                  read_dataset_size)
from metrics import (GREEDY_METRICS, TOPK_METRICS, per_target_greedy,         # noqa: E402
                     per_target_topk)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Conditioned validation run dir (contains per_sample/).")
    p.add_argument("--ablation-dir", action="append", default=[],
                   metavar="NAME=PATH",
                   help="Ablation run as name=path (greedy-only). Repeatable.")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Validation dataset dir; metadata.json supplies the expected "
                        "target count for the completeness check.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--allow-check-failures", action="store_true",
                   help="Report consistency violations without aborting "
                        "(intended for inspecting ablation runs).")
    return p.parse_args()


def slice_rows(
    pt: pd.DataFrame,
    metric_specs: list[tuple[str, str]],
    *,
    pass_name: str,
    seed: int,
    b: int,
    depth: int | None = None,
) -> list[dict]:
    """Estimate + CI rows for one (pass, depth-slice); one index matrix reused
    across every metric of the slice so intervals share their resamples."""
    sub = pt if depth is None else pt[pt["target_depth"] == depth]
    slice_seed = seed if depth is None else seed + 100 * depth
    idx = index_matrix(len(sub), b=b, seed=slice_seed)
    rows = []
    for metric, col in metric_specs:
        est, lo, hi, n_eff = mean_ci(sub[col].to_numpy(), idx)
        rows.append({
            "pass": pass_name,
            "metric": metric,
            "target_depth": "all" if depth is None else depth,
            "estimate": est, "ci_lo": lo, "ci_hi": hi,
            "n_targets": len(sub), "n_effective": n_eff,
        })
    return rows


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    expected_n = read_dataset_size(args.dataset_dir)
    all_checks: list[dict] = []

    # ---- load conditioned run -------------------------------------------- #
    df_g, checks = load_greedy(args.run_dir, expected_n=expected_n)
    all_checks += checks
    df_t, checks = load_topk(args.run_dir, expected_n=expected_n, k=args.top_k)
    all_checks += checks

    bos, eos = infer_special_ids(df_t["token_ids"])
    pt_g = per_target_greedy(df_g)
    pt_t = per_target_topk(df_t, k_max=args.top_k, bos=bos, eos=eos)
    depths = sorted(pt_g["target_depth"].unique().tolist())

    topk_specs = TOPK_METRICS + [(f"pass_at_{k}", f"pass_at_{k}")
                                 for k in range(1, args.top_k + 1)]

    # ---- overall + by-depth ---------------------------------------------- #
    overall = (slice_rows(pt_g, GREEDY_METRICS, pass_name="greedy",
                          seed=args.seed, b=args.bootstrap_samples)
               + slice_rows(pt_t, topk_specs, pass_name="sampling",
                            seed=args.seed, b=args.bootstrap_samples))
    by_depth = []
    for d in depths:
        by_depth += slice_rows(pt_g, GREEDY_METRICS, pass_name="greedy",
                               seed=args.seed, b=args.bootstrap_samples, depth=d)
        by_depth += slice_rows(pt_t, topk_specs, pass_name="sampling",
                               seed=args.seed, b=args.bootstrap_samples, depth=d)

    pd.DataFrame(overall).to_csv(out / "overall.csv", index=False)
    pd.DataFrame(by_depth).to_csv(out / "by_depth.csv", index=False)

    pass_k = [{"k": k, **{key: r[key] for key in
                          ("estimate", "ci_lo", "ci_hi", "n_targets")}}
              for k in range(1, args.top_k + 1)
              for r in overall
              if r["pass"] == "sampling" and r["metric"] == f"pass_at_{k}"]
    pd.DataFrame(pass_k).to_csv(out / "pass_at_k.csv", index=False)

    # ---- ablation floors -------------------------------------------------- #
    ablation_rows = []
    floor_specs = [(m, c) for m, c in GREEDY_METRICS
                   if m in ("semantic_equivalent_rate", "invalid_rate",
                            "semantic_distance")]
    for r in overall:
        if r["pass"] == "greedy" and r["metric"] in dict(floor_specs):
            ablation_rows.append({"condition": "conditioned", **{k: r[k] for k in
                                  ("metric", "estimate", "ci_lo", "ci_hi",
                                   "n_targets", "n_effective")}})
    for spec in args.ablation_dir:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--ablation-dir expects NAME=PATH, got {spec!r}")
        df_a, checks = load_greedy(Path(path), expected_n=expected_n)
        all_checks += checks
        pt_a = per_target_greedy(df_a)
        for r in slice_rows(pt_a, floor_specs, pass_name="greedy",
                            seed=args.seed, b=args.bootstrap_samples):
            ablation_rows.append({"condition": name, **{k: r[k] for k in
                                  ("metric", "estimate", "ci_lo", "ci_hi",
                                   "n_targets", "n_effective")}})
    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(out / "ablation.csv", index=False)

    # ---- checks + manifest ------------------------------------------------ #
    checks_df = pd.DataFrame(all_checks)
    checks_df.to_csv(out / "checks.csv", index=False)
    n_violations = int(checks_df["violations"].sum())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "ablation_dirs": args.ablation_dir,
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir else None,
        "expected_n_targets": expected_n,
        "top_k": args.top_k,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed,
                      "slice_seed_rule": "seed + 100 * target_depth",
                      "interval": "percentile 2.5/97.5"},
        "special_ids": {"bos": bos, "eos": eos},
        "check_violations": n_violations,
        "thesis_objects": {
            "tab:exp1_results_overall": {"file": "overall.csv"},
            "tab:exp1_results_by_depth": {"file": "by_depth.csv"},
            "fig:exp1_greedy_by_depth": {
                "file": "by_depth.csv",
                "filter": "pass=='greedy' & metric in ['semantic_equivalent_rate',"
                          "'semantic_distance','invalid_rate']"},
            "fig:exp1_generated_depth": {
                "file": "by_depth.csv",
                "filter": "metric=='generated_depth_valid'"},
            "fig:exp1_pass_at_k": {"file": "pass_at_k.csv"},
            "fig:exp1_discovery": {
                "file": "by_depth.csv",
                "filter": "pass=='sampling' & metric in ['distinct_correct_all',"
                          "'distinct_correct_solved','self_bleu']"},
            "tab:exp1_ablation": {"file": "ablation.csv"},
            "incorrect_valid_distance": {
                "file": "by_depth.csv (+ overall.csv)",
                "filter": "metric=='wrong_valid_distance'"},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[exp1] wrote tables to {out}  (check violations: {n_violations})")
    if n_violations and not args.allow_check_failures:
        raise SystemExit(
            f"{n_violations} consistency violations (see checks.csv); "
            "rerun with --allow-check-failures to keep the tables anyway.")


if __name__ == "__main__":
    main()
