"""Driver: embedding-geometry vs. correctness analysis on the validation runs.

Light step (consumes the cached geometry_features.csv from compute_geometry_features.py).
Mirrors visualize_validation_extra.py: writes stats/extra/geometry_*.csv and
figures/extra/geometry_* into the existing _analysis tree.

Example:
  python scripts/visualize_validation_geometry.py \
      --validation-root /home/mees/Documents/KernelLTL/artifacts/validation \
      --geometry-features /home/mees/Documents/KernelLTL/artifacts/validation/_analysis/geometry_features.csv \
      --output-dir /home/mees/Documents/KernelLTL/artifacts/validation/_analysis \
      --runs ce_base ce_finetune rb_momentum_09 gae_lambda_09 gae_lambda_1 \
      --reference-run ce_base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/visualize_validation_geometry.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._validation_analysis import geometry_analysis as ga
from scripts._validation_analysis import geometry_plots as gp


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation-root", required=True,
                   help="Dir holding the per-run folders (with per_sample/*.jsonl)")
    p.add_argument("--geometry-features", required=True, help="geometry_features.csv")
    p.add_argument("--output-dir", required=True, help="Analysis output dir (_analysis)")
    p.add_argument("--runs", nargs="+", required=True, help="Short run labels")
    p.add_argument("--reference-run", default="ce_base")
    p.add_argument("--bootstrap-n", type=int, default=10000)
    p.add_argument("--n-sim", type=int, default=2000, help="Parametric-sim draws for AME CIs")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=12, help="Marginal-curve bins")
    p.add_argument("--grid-bins", type=int, default=8, help="2-D heatmap bins per axis")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--regularized-fallback", action="store_true")
    return p.parse_args()


def _resolve_run_dir(root: Path, label: str) -> Path:
    """Map a short run label to its folder (exact, or `label_*` prefix)."""
    exact = root / label
    if exact.is_dir():
        return exact
    cands = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d.name == label or d.name.startswith(label + "_")))
    if not cands:
        raise FileNotFoundError(f"no run folder for label '{label}' under {root}")
    return cands[0]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_correctness(root: Path, runs: list[str]) -> dict[str, pd.DataFrame]:
    """Per-run long frames for two outcomes, joined later to geometry features.

    Returns {'greedy': df, 'topk_any': df} each with columns
    run, formula_id, correct, target_depth.
    """
    greedy_frames, topk_frames = [], []
    for label in runs:
        d = _resolve_run_dir(root, label)
        _log(f"[geom] {label} -> {d.name}")
        g = pd.DataFrame(_read_jsonl(d / "per_sample" / "greedy.jsonl"))
        g_out = g[["formula_id", "target_depth"]].copy()
        g_out["correct"] = g["is_semantic_equivalent"].astype(int)
        g_out["run"] = label
        greedy_frames.append(g_out)

        tf = pd.DataFrame(_read_jsonl(d / "per_sample" / "topk_flat.jsonl"))
        tf["is_equiv"] = (tf["reward"].astype(float) == 1.0)
        any_correct = tf.groupby("formula_id")["is_equiv"].any().astype(int).reset_index(name="correct")
        t_out = any_correct.merge(g[["formula_id", "target_depth"]], on="formula_id", how="left")
        t_out["run"] = label
        topk_frames.append(t_out)

    return {
        "greedy": pd.concat(greedy_frames, ignore_index=True),
        "topk_any": pd.concat(topk_frames, ignore_index=True),
    }


def run_outcome(tag: str, corr: pd.DataFrame, features: pd.DataFrame, args, stats_dir: Path,
                fig_dir: Path) -> dict:
    runs = args.runs
    df = ga.build_frame(features, corr)

    # per-model logistic (CI headline; no BH)
    coef = ga.per_model_logistic(df, runs=runs, alpha=args.alpha,
                                 use_regularized=args.regularized_fallback)
    coef.to_csv(stats_dir / f"geometry_logistic_coef_{tag}.csv", index=False)
    gp.plot_logistic_forest(coef, runs=runs, stem=fig_dir / f"geometry_logistic_forest_{tag}", dpi=args.dpi)

    # marginal binned curves
    for feat in ga.GEOMETRY_PREDICTORS:
        binned = ga.marginal_binned(df, runs=runs, feature=feat, n_bins=args.bins,
                                    n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed)
        binned.to_csv(stats_dir / f"geometry_marginal_{feat}_{tag}.csv", index=False)
        gp.plot_marginal(binned, runs=runs, feature=feat, stem=fig_dir / f"geometry_marginal_{feat}_{tag}", dpi=args.dpi)

    # 2-D std x alignment grid + per-run heatmaps
    grid = ga.two_d_grid(df, runs=runs, fx="std", fy="alignment", nbins=args.grid_bins)
    grid.to_csv(stats_dir / f"geometry_2d_grid_{tag}.csv", index=False)
    for r in runs:
        gp.plot_2d_heatmap(grid, run=r, stem=fig_dir / f"geometry_2d_heatmap_{r}_{tag}", dpi=args.dpi)

    # cross-model interaction + AME (BH-FDR family)
    res = ga.fit_pooled_interaction(df, runs=runs, reference_run=args.reference_run,
                                    outcome_col="correct", alpha=args.alpha,
                                    n_sim=args.n_sim, rng_seed=args.rng_seed)
    res["interactions"].to_csv(stats_dir / f"geometry_crossmodel_interaction_{tag}.csv", index=False)
    res["interactions_pairwise"].to_csv(stats_dir / f"geometry_crossmodel_interaction_pairwise_{tag}.csv", index=False)
    res["ame"].to_csv(stats_dir / f"geometry_crossmodel_ame_{tag}.csv", index=False)
    res["ame_pairwise"].to_csv(stats_dir / f"geometry_crossmodel_ame_pairwise_{tag}.csv", index=False)
    gp.plot_crossmodel_interaction(res["interactions"], reference_run=args.reference_run,
                                   stem=fig_dir / f"geometry_crossmodel_interaction_{tag}", dpi=args.dpi)
    return {"coef": coef, "interactions": res["interactions"], "ame": res["ame"],
            "n_obs": res["n_obs"], "n_targets": res["n_targets"]}


def main() -> None:
    args = parse_args()
    root = Path(args.validation_root)
    out = Path(args.output_dir)
    stats_dir = out / "stats" / "extra"
    fig_dir = out / "figures" / "extra"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.geometry_features)
    _log(f"[geom] features: {len(features)} targets")
    gp.plot_basrate_hist(features, stem=fig_dir / "geometry_basrate_hist", dpi=args.dpi)

    corr = load_correctness(root, args.runs)
    summary = {"reference_run": args.reference_run, "runs": args.runs,
               "bootstrap_n": args.bootstrap_n, "alpha": args.alpha, "outcomes": {}}
    for tag in ("greedy", "topk_any"):
        _log(f"[geom] outcome = {tag}")
        r = run_outcome(tag, corr[tag], features, args, stats_dir, fig_dir)
        summary["outcomes"][tag] = {"n_obs": r["n_obs"], "n_targets": r["n_targets"]}

    with open(out / "geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"[geom] done. stats -> {stats_dir}, figures -> {fig_dir}")


if __name__ == "__main__":
    main()
