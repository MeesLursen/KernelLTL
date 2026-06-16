"""Driver: embedding-geometry vs. correctness (Q1 magnitude / Q2 orthogonality).

Light step — consumes the cached geometry_features.csv (from compute_geometry_features.py).
Drops trivial (tautology/contradiction, std==0) targets via the is_trivial flag, then runs:
  Q1  correct ~ emb_norm                                  (marginal magnitude)
  Q2  variance-stratified norm slopes                     (primary orthogonality test)
  Q2  correct ~ variance + norm_resid + C(depth)          (FWL residual summary)
  bonus cross-model interaction (BH-FDR) + AME
Outcome = binary correct; semantic_distance kept only as a flagged descriptive curve.

Writes stats/extra/geometry_*.csv and figures/extra/geometry_* into the _analysis tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._validation_analysis import geometry_analysis as ga
from scripts._validation_analysis import geometry_plots as gp


def _log(m):
    print(m, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation-root", required=True)
    p.add_argument("--geometry-features", required=True)
    p.add_argument("--faithfulness-features", default=None,
                   help="Optional faithfulness_features.csv (compute_faithfulness_features.py): "
                        "enables RQ1a representation-faithfulness descriptives, the mediation "
                        "(does relational faithfulness screen off norm_resid), and the "
                        "faithfulness-conditioned cross-model interaction (clean H1).")
    p.add_argument("--faith-quantile", type=float, default=0.5,
                   help="Faithfulness threshold (quantile) for the faithful-but-weak subset (I2).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--reference-run", default="ce_base")
    p.add_argument("--bootstrap-n", type=int, default=10000)
    p.add_argument("--n-sim", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--grid-bins", type=int, default=8)
    p.add_argument("--n-strata", type=int, default=3)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _resolve_run_dir(root: Path, label: str) -> Path:
    if (root / label).is_dir():
        return root / label
    cands = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d.name == label or d.name.startswith(label + "_")))
    if not cands:
        raise FileNotFoundError(f"no run folder for '{label}' under {root}")
    return cands[0]


def load_greedy_correctness(root: Path, runs) -> pd.DataFrame:
    frames = []
    for label in runs:
        d = _resolve_run_dir(root, label)
        rows = [json.loads(l) for l in open(d / "per_sample" / "greedy.jsonl")]
        g = pd.DataFrame(rows)[["formula_id", "is_semantic_equivalent", "semantic_distance", "target_depth"]]
        g["correct"] = g["is_semantic_equivalent"].astype(int)
        g["run"] = label
        frames.append(g[["run", "formula_id", "correct", "semantic_distance", "target_depth"]])
        _log(f"[geom] {label} -> {d.name} ({len(g)} rows)")
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    stats_dir = out / "stats" / "extra"
    fig_dir = out / "figures" / "extra"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs, ref = args.runs, args.reference_run

    features = pd.read_csv(args.geometry_features)
    n_triv = int(features.get("is_trivial", pd.Series(0)).sum())
    _log(f"[geom] features: {len(features)} targets ({n_triv} trivial dropped)")

    corr = load_greedy_correctness(Path(args.validation_root), runs)
    df = ga.build_frame(features, corr)
    _log(f"[geom] analysis frame: {len(df)} rows, {df.formula_id.nunique()} non-trivial targets")

    # --- Q1: marginal magnitude ---
    q1 = ga.q1_marginal(df, runs=runs, alpha=args.alpha)
    q1.to_csv(stats_dir / "geometry_q1_marginal.csv", index=False)

    # --- Q2: variance-stratified norm slopes (primary) + FWL residual (summary) ---
    strat = ga.variance_stratified_slopes(df, runs=runs, n_strata=args.n_strata, alpha=args.alpha)
    strat.to_csv(stats_dir / "geometry_stratified.csv", index=False)
    gp.plot_stratified(strat, runs=runs, stem=fig_dir / "geometry_stratified", dpi=args.dpi)

    resid = ga.q2_residual(df, runs=runs, alpha=args.alpha)
    resid.to_csv(stats_dir / "geometry_q2_residual.csv", index=False)
    gp.plot_residual_forest(resid, runs=runs, stem=fig_dir / "geometry_q2_residual_forest", dpi=args.dpi)

    # --- descriptive marginal curves (binary + flagged distance) ---
    for feat in ["emb_norm", "variance"]:
        for outcome in ["correct", "semantic_distance"]:
            mb = ga.marginal_binned(df, runs=runs, feature=feat, outcome=outcome, n_bins=args.bins,
                                    n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed)
            mb.to_csv(stats_dir / f"geometry_marginal_{feat}_{outcome}.csv", index=False)
            gp.plot_marginal(mb, runs=runs, feature=feat, outcome=outcome,
                             stem=fig_dir / f"geometry_marginal_{feat}_{outcome}", dpi=args.dpi)

    # --- 2-D variance x norm_resid grid (fills, since decorrelated) + scatter w/ ceiling ---
    grid = ga.two_d_grid(df, runs=runs, fx="variance", fy="norm_resid", nbins=args.grid_bins)
    grid.to_csv(stats_dir / "geometry_2d_grid.csv", index=False)
    for r in runs:
        gp.plot_2d_heatmap(grid, run=r, stem=fig_dir / f"geometry_2d_heatmap_{r}", dpi=args.dpi)
        gp.plot_scatter_ceiling(df, run=r, color_by="correct", stem=fig_dir / f"geometry_scatter_{r}", dpi=args.dpi)

    # --- bonus: cross-model residual interaction (BH-FDR) + AME ---
    res = ga.cross_model_interaction(df, runs=runs, reference_run=ref, alpha=args.alpha,
                                     n_sim=args.n_sim, rng_seed=args.rng_seed)
    res["interactions"].to_csv(stats_dir / "geometry_crossmodel_interaction.csv", index=False)
    res["ame"].to_csv(stats_dir / "geometry_crossmodel_ame.csv", index=False)
    gp.plot_crossmodel_interaction(res["interactions"], reference_run=ref,
                                   stem=fig_dir / "geometry_crossmodel_interaction", dpi=args.dpi)

    # --- RQ1a representation faithfulness (optional; needs faithfulness_features.csv) ---
    faith_summary = {}
    if args.faithfulness_features:
        _log("[geom] representation-faithfulness analyses ...")
        faith = pd.read_csv(args.faithfulness_features)
        dff = df.merge(faith[["formula_id", "relational_faithfulness"]],
                       on="formula_id", how="inner")
        one = dff[dff["run"] == ref]   # faithfulness is model-independent -> one per-target curve

        # descriptive: faithfulness over the variance x norm_resid cell (does DIRECTION collapse?)
        fgrid = ga.two_d_grid(one, runs=[ref], fx="variance", fy="norm_resid",
                              nbins=args.grid_bins, outcome="relational_faithfulness")
        fgrid.to_csv(stats_dir / "geometry_faith_2d_grid.csv", index=False)
        gp.plot_2d_heatmap(fgrid, run=ref, stem=fig_dir / "geometry_faith_2d_heatmap", dpi=args.dpi)
        for feat in ["variance", "norm_resid", "emb_norm"]:
            mb = ga.marginal_binned(one, runs=[ref], feature=feat,
                                    outcome="relational_faithfulness", n_bins=args.bins,
                                    n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed)
            mb.to_csv(stats_dir / f"geometry_faith_marginal_{feat}.csv", index=False)
            gp.plot_marginal(mb, runs=[ref], feature=feat, outcome="relational_faithfulness",
                             stem=fig_dir / f"geometry_faith_marginal_{feat}", dpi=args.dpi)

        # mediation: does relational faithfulness screen off norm_resid? (sec:geometry_bridge)
        med = ga.faithfulness_mediation(df, faith, runs=runs, alpha=args.alpha)
        med.to_csv(stats_dir / "geometry_faith_mediation.csv", index=False)

        # I2: faithfulness-conditioned cross-model interaction (clean H1) on population A
        res2 = ga.cross_model_interaction_on_faithful(
            df, faith, runs=runs, reference_run=ref, faith_quantile=args.faith_quantile,
            alpha=args.alpha, n_sim=args.n_sim, rng_seed=args.rng_seed)
        res2["interactions"].to_csv(stats_dir / "geometry_faith_interaction.csv", index=False)
        res2["ame"].to_csv(stats_dir / "geometry_faith_ame.csv", index=False)
        if not res2["interactions"].empty:
            gp.plot_crossmodel_interaction(res2["interactions"], reference_run=ref,
                                           stem=fig_dir / "geometry_faith_interaction", dpi=args.dpi)
        faith_summary = {
            "faith_threshold": res2.get("faith_threshold"),
            "faith_n_dropped_unfaithful": res2.get("n_dropped_unfaithful"),
            "faith_interaction_n_targets": res2.get("n_targets"),
            "faith_mediation_runs": int(len(med)),
        }

    summary = {"reference_run": ref, "runs": runs, "n_targets_nontrivial": int(df.formula_id.nunique()),
               "n_trivial_dropped": n_triv, "bootstrap_n": args.bootstrap_n,
               "crossmodel_n_obs": res["n_obs"], "crossmodel_n_targets": res["n_targets"],
               **faith_summary}
    with open(out / "geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"[geom] done -> {stats_dir}, {fig_dir}")


if __name__ == "__main__":
    main()
