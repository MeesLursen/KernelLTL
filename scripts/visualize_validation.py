"""Cross-model validation analysis CLI.

Loads per-sample JSONLs from every run under ``--validation-root``,
derives the top-K per-(target, k_idx) booleans + depth + length and the
per-target K-aggregates, then produces:

* ~30 figures under ``<output-dir>/figures/``
* paired stats CSVs under ``<output-dir>/stats/``
* a Wilcoxon-assumption diagnostics PDF under ``<output-dir>/diagnostics/``
* a narrative ``summary.md`` and ``run_metadata.json``

Usage::

    python scripts/visualize_validation.py \\
        --validation-root artifacts/validation \\
        --runs ce_base ce_finetune rb_momentum_09_lr_5e-8 \\
               gae_lambda_09_lr_5e-8_crlr_1e-3 gae_lambda_1_lr_5e-8_crlr_5e-3 \\
        --reference-run ce_base \\
        --tokenizer-dir artifacts/tokenizer \\
        --output-dir artifacts/validation/_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._validation_analysis import loaders, plots, report, stats
from scripts._validation_analysis.stats import MetricSpec


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


GREEDY_SPECS: list[MetricSpec] = [
    MetricSpec("semantic_distance", "continuous_bounded", "semantic_distance", "greedy"),
    MetricSpec("is_invalid", "binary", "is_invalid", "greedy"),
    MetricSpec("is_exact_match", "binary", "is_exact_match", "greedy"),
    MetricSpec("is_semantic_equivalent", "binary", "is_semantic_equivalent", "greedy"),
    MetricSpec("generated_depth", "continuous_unbounded", "generated_depth", "greedy"),
    MetricSpec("generated_length_tokens", "continuous_unbounded", "generated_length_tokens", "greedy"),
    MetricSpec("policy_entropy_seq_mean", "continuous_unbounded", "seq_entropy_mean", "greedy"),
    MetricSpec("kl_from_base_seq_mean", "continuous_unbounded", "seq_kl_mean", "greedy"),
]

TOPK_AGG_SPECS: list[MetricSpec] = [
    MetricSpec("semantic_distance_mean_topk", "continuous_bounded", "semantic_distance_mean_topk", "topk_aggregates"),
    MetricSpec("semantic_distance_variance_topk", "continuous_unbounded", "semantic_distance_variance_topk", "topk_aggregates"),
    MetricSpec("invalid_rate_topk", "rate", "invalid_rate_topk", "topk_aggregates"),
    MetricSpec("exact_match_rate_topk", "rate", "exact_match_rate_topk", "topk_aggregates"),
    MetricSpec("semantic_equiv_rate_topk", "rate", "semantic_equiv_rate_topk", "topk_aggregates"),
    MetricSpec("syntax_semantics_gap_topk", "continuous_bounded", "syntax_semantics_gap_topk", "topk_aggregates"),
    MetricSpec("generated_depth_mean_topk", "continuous_unbounded", "generated_depth_mean_topk", "topk_aggregates"),
    MetricSpec("generated_length_tokens_mean_topk", "continuous_unbounded", "generated_length_tokens_mean_topk", "topk_aggregates"),
]

TOPK_GROUPED_SPECS: list[MetricSpec] = [
    MetricSpec("self_bleu_mean", "continuous_bounded", "self_bleu", "topk_grouped"),
    MetricSpec("policy_entropy_target_seq_mean", "continuous_unbounded", "policy_entropy_target_seq_mean", "topk_grouped"),
    MetricSpec("kl_from_base_target_seq_mean", "continuous_unbounded", "kl_from_base_target_seq_mean", "topk_grouped"),
]

ALL_SPECS = GREEDY_SPECS + TOPK_AGG_SPECS + TOPK_GROUPED_SPECS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--validation-root", default="artifacts/validation")
    p.add_argument("--dataset-dir", default="artifacts/datasets/validation")
    p.add_argument("--output-dir", default="artifacts/validation/_analysis")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--reference-run", default="ce_base")
    p.add_argument("--tokenizer-dir", default=None,
                   help="Tokenizer directory; used to determine pad_token_id. "
                        "If omitted, pad_token_id defaults to 0.")
    p.add_argument("--bootstrap-n", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mc-correction", choices=["bh", "holm"], default="bh")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-stats", action="store_true")
    p.add_argument("--keep-token-arrays", action="store_true",
                   help="Retain per-token float arrays in df_topk_flat (memory-heavy).")
    return p.parse_args()


def _resolve_pad_token_id(tokenizer_dir: str | None) -> int:
    if tokenizer_dir is None:
        return 0
    try:
        from tokenizer_pretrained_class import LTLTokenizer  # type: ignore
        tok = LTLTokenizer.from_pretrained(tokenizer_dir)
        return int(tok.pad_token_id)
    except Exception as e:  # pragma: no cover
        print(f"[visualize_validation] WARNING: could not load tokenizer ({e}); pad_token_id=0", file=sys.stderr)
        return 0


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-run summary table for radar / pareto plots
# ---------------------------------------------------------------------------


def build_per_run_summary(sources: dict[str, pd.DataFrame], runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        g = sources["greedy"][sources["greedy"]["run"] == run]
        agg = sources["topk_aggregates"][sources["topk_aggregates"]["run"] == run]
        gp = sources["topk_grouped"][sources["topk_grouped"]["run"] == run]
        rows.append({
            "run": run,
            # Monotone-quality axes (high = better)
            "validity_greedy": 1.0 - float(g["is_invalid"].astype(bool).mean()) if len(g) else float("nan"),
            "correctness_greedy": 1.0 - float(g["semantic_distance"].astype(float).mean()) if len(g) else float("nan"),
            "semantic_equiv_rate": float(g["is_semantic_equivalent"].astype(bool).mean()) if len(g) else float("nan"),
            "validity_topk": 1.0 - float(agg["invalid_rate_topk"].astype(float).mean()) if len(agg) else float("nan"),
            "correctness_topk": 1.0 - float(agg["semantic_distance_mean_topk"].astype(float).mean()) if len(agg) else float("nan"),
            "semantic_equiv_rate_topk": float(agg["semantic_equiv_rate_topk"].astype(float).mean()) if len(agg) else float("nan"),
            "diversity_topk": 1.0 - float(gp["self_bleu"].astype(float).mean(skipna=True)) if "self_bleu" in gp.columns else float("nan"),
            # Other quantities for the descriptive plots
            "syntax_semantics_gap_greedy": float(
                g["is_semantic_equivalent"].astype(bool).mean() - g["is_exact_match"].astype(bool).mean()
            ) if len(g) else float("nan"),
            "self_bleu": float(gp["self_bleu"].astype(float).mean(skipna=True)) if "self_bleu" in gp.columns else float("nan"),
            "kl_from_base_seq_mean": float(g["seq_kl_mean"].astype(float).mean(skipna=True)) if "seq_kl_mean" in g.columns else float("nan"),
            "semantic_distance_greedy_mean": float(g["semantic_distance"].astype(float).mean()) if len(g) else float("nan"),
            "semantic_distance_topk_mean": float(agg["semantic_distance_mean_topk"].astype(float).mean()) if len(agg) else float("nan"),
            "depth_degradation_slope": _depth_slope(g, "semantic_distance"),
        })
    return pd.DataFrame(rows)


def _depth_slope(df: pd.DataFrame, col: str) -> float:
    if "target_depth" not in df.columns or len(df) == 0 or col not in df.columns:
        return float("nan")
    by = df.groupby("target_depth")[col].mean().reset_index().sort_values("target_depth")
    if len(by) < 2:
        return float("nan")
    x = by["target_depth"].astype(float).to_numpy()
    y = by[col].astype(float).to_numpy()
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# Plot orchestration
# ---------------------------------------------------------------------------


def _render_cross_model_figures(
    sources: dict[str, pd.DataFrame],
    runs: list[str],
    fig_dir: Path,
    rng_seed: int,
) -> None:
    df_greedy = sources["greedy"]
    df_agg = sources["topk_aggregates"]
    df_grp = sources["topk_grouped"]

    # Greedy bars (rate-style)
    plots.plot_metric_rate_bar(df_greedy, "is_invalid", runs=runs,
                               title="Greedy invalid rate", ylabel="rate",
                               stem=fig_dir / "greedy_invalid_rate", rng_seed=rng_seed)
    plots.plot_metric_rate_bar(df_greedy, "is_exact_match", runs=runs,
                               title="Greedy exact-match rate", ylabel="rate",
                               stem=fig_dir / "greedy_exact_match_rate", rng_seed=rng_seed)
    plots.plot_metric_rate_bar(df_greedy, "is_semantic_equivalent", runs=runs,
                               title="Greedy semantic-equiv rate", ylabel="rate",
                               stem=fig_dir / "greedy_semantic_equiv_rate", rng_seed=rng_seed)

    # Greedy boxes
    plots.plot_metric_box(df_greedy, "semantic_distance", runs=runs,
                          title="Greedy semantic distance", ylabel="distance",
                          stem=fig_dir / "greedy_semantic_distance_box")
    plots.plot_metric_box(df_greedy, "generated_length_tokens", runs=runs,
                          title="Greedy generated length (tokens)", ylabel="tokens",
                          stem=fig_dir / "greedy_length_box")
    plots.plot_metric_box(df_greedy, "generated_depth", runs=runs,
                          title="Greedy generated depth", ylabel="depth",
                          stem=fig_dir / "greedy_gen_depth_box")
    plots.plot_metric_box(df_greedy, "seq_entropy_mean", runs=runs,
                          title="Greedy policy entropy (seq mean)", ylabel="nats",
                          stem=fig_dir / "greedy_entropy_seq_mean_box")
    plots.plot_metric_box(df_greedy, "seq_kl_mean", runs=runs,
                          title="Greedy KL from base (seq mean)", ylabel="nats",
                          stem=fig_dir / "greedy_kl_seq_mean_box",
                          grey_runs={"ce_base"})

    # Top-K rate-style metrics
    plots.plot_metric_rate_bar(df_agg, "invalid_rate_topk", runs=runs,
                               title="Top-K invalid rate", ylabel="rate",
                               stem=fig_dir / "topk_invalid_rate", rng_seed=rng_seed)
    plots.plot_metric_rate_bar(df_agg, "exact_match_rate_topk", runs=runs,
                               title="Top-K exact-match rate", ylabel="rate",
                               stem=fig_dir / "topk_exact_match_rate", rng_seed=rng_seed)
    plots.plot_metric_rate_bar(df_agg, "semantic_equiv_rate_topk", runs=runs,
                               title="Top-K semantic-equiv rate", ylabel="rate",
                               stem=fig_dir / "topk_semantic_equiv_rate", rng_seed=rng_seed)

    # Top-K boxes
    plots.plot_metric_box(df_agg, "semantic_distance_mean_topk", runs=runs,
                          title="Top-K semantic distance (mean over K)", ylabel="distance",
                          stem=fig_dir / "topk_semantic_distance_mean_box")
    plots.plot_metric_box(df_agg, "semantic_distance_variance_topk", runs=runs,
                          title="Top-K within-target semantic-distance variance", ylabel="variance",
                          stem=fig_dir / "topk_semantic_distance_variance_box")
    plots.plot_metric_box(df_agg, "syntax_semantics_gap_topk", runs=runs,
                          title="Top-K syntax–semantics gap", ylabel="equiv − exact-match",
                          stem=fig_dir / "topk_syntax_semantics_gap_box")
    plots.plot_metric_box(df_agg, "generated_depth_mean_topk", runs=runs,
                          title="Top-K mean generated depth", ylabel="depth",
                          stem=fig_dir / "topk_gen_depth_mean_box")
    plots.plot_metric_box(df_agg, "generated_length_tokens_mean_topk", runs=runs,
                          title="Top-K mean generated length", ylabel="tokens",
                          stem=fig_dir / "topk_gen_length_mean_box")
    plots.plot_metric_box(df_grp, "self_bleu", runs=runs,
                          title="Top-K self-BLEU (lower = more diverse)", ylabel="self-BLEU",
                          stem=fig_dir / "topk_self_bleu_box")
    plots.plot_metric_box(df_grp, "policy_entropy_target_seq_mean", runs=runs,
                          title="Top-K policy entropy (per-target seq mean)", ylabel="nats",
                          stem=fig_dir / "topk_entropy_seq_mean_box")
    plots.plot_metric_box(df_grp, "kl_from_base_target_seq_mean", runs=runs,
                          title="Top-K KL from base (per-target seq mean)", ylabel="nats",
                          stem=fig_dir / "topk_kl_seq_mean_box",
                          grey_runs={"ce_base"})


def _render_ecdfs(sources, runs, fig_dir, rng_seed):
    df_g = sources["greedy"]
    df_flat = sources["topk_flat"]
    df_grp = sources["topk_grouped"]

    plots.plot_ecdf(df_g, "semantic_distance", runs=runs,
                    title="ECDF of greedy semantic distance",
                    xlabel="semantic distance",
                    stem=fig_dir / "ecdf_semantic_distance_greedy", rng_seed=rng_seed)
    plots.plot_ecdf(df_g[df_g["is_invalid"] == False], "semantic_distance", runs=runs,  # noqa: E712
                    title="ECDF of greedy semantic distance (valid only)",
                    xlabel="semantic distance",
                    stem=fig_dir / "ecdf_semantic_distance_greedy_valid_only", rng_seed=rng_seed)
    plots.plot_ecdf(df_flat, "semantic_distance", runs=runs,
                    title="ECDF of top-K semantic distance (per (target, k))",
                    xlabel="semantic distance",
                    stem=fig_dir / "ecdf_semantic_distance_topk", rng_seed=rng_seed)
    plots.plot_ecdf(df_g, "seq_entropy_mean", runs=runs,
                    title="ECDF of greedy policy entropy (seq mean)",
                    xlabel="nats",
                    stem=fig_dir / "ecdf_entropy_seq_mean_greedy", rng_seed=rng_seed)
    plots.plot_ecdf(df_grp, "policy_entropy_target_seq_mean", runs=runs,
                    title="ECDF of top-K policy entropy (per-target seq mean)",
                    xlabel="nats",
                    stem=fig_dir / "ecdf_entropy_seq_mean_topk", rng_seed=rng_seed)
    plots.plot_ecdf(df_g, "seq_kl_mean", runs=runs, exclude_runs={"ce_base"},
                    title="ECDF of greedy KL from base",
                    xlabel="nats",
                    stem=fig_dir / "ecdf_kl_seq_mean_greedy", rng_seed=rng_seed)
    plots.plot_ecdf(df_grp, "kl_from_base_target_seq_mean", runs=runs, exclude_runs={"ce_base"},
                    title="ECDF of top-K KL from base (per-target seq mean)",
                    xlabel="nats",
                    stem=fig_dir / "ecdf_kl_seq_mean_topk", rng_seed=rng_seed)


def _render_bydepth(sources, runs, fig_dir, rng_seed):
    df_g = sources["greedy"]
    df_agg = sources["topk_aggregates"]
    df_grp = sources["topk_grouped"]

    bydepth_specs = [
        (df_g, "is_invalid", "Invalid rate (greedy)", "rate", "bydepth_invalid_rate", set()),
        (df_g, "is_exact_match", "Exact-match rate (greedy)", "rate", "bydepth_exact_match_rate", set()),
        (df_g, "is_semantic_equivalent", "Semantic-equiv rate (greedy)", "rate", "bydepth_semantic_equiv_rate", set()),
        (df_g, "semantic_distance", "Semantic distance (greedy)", "distance", "bydepth_semantic_distance", set()),
        (df_g, "generated_depth", "Generated depth (greedy)", "depth", "bydepth_gen_depth", set()),
        (df_g, "generated_length_tokens", "Generated length tokens (greedy)", "tokens", "bydepth_gen_length", set()),
        (df_g, "seq_entropy_mean", "Policy entropy (greedy, seq mean)", "nats", "bydepth_entropy_seq_mean", set()),
        (df_g, "seq_kl_mean", "KL from base (greedy, seq mean)", "nats", "bydepth_kl_seq_mean", {"ce_base"}),
        # Top-K aggregates
        (df_agg, "invalid_rate_topk", "Invalid rate (top-K)", "rate", "bydepth_invalid_rate_topk", set()),
        (df_agg, "exact_match_rate_topk", "Exact-match rate (top-K)", "rate", "bydepth_exact_match_rate_topk", set()),
        (df_agg, "semantic_equiv_rate_topk", "Semantic-equiv rate (top-K)", "rate", "bydepth_semantic_equiv_rate_topk", set()),
        (df_agg, "semantic_distance_mean_topk", "Semantic distance (top-K mean)", "distance", "bydepth_semantic_distance_topk", set()),
        (df_agg, "syntax_semantics_gap_topk", "Syntax-semantics gap (top-K)", "equiv − exact-match", "bydepth_syntax_semantics_gap_topk", set()),
        (df_agg, "generated_depth_mean_topk", "Mean generated depth (top-K)", "depth", "bydepth_gen_depth_topk", set()),
        (df_agg, "generated_length_tokens_mean_topk", "Mean generated length (top-K)", "tokens", "bydepth_gen_length_topk", set()),
        # Top-K grouped
        (df_grp, "self_bleu", "Self-BLEU (top-K)", "self-BLEU", "bydepth_self_bleu", set()),
        (df_grp, "policy_entropy_target_seq_mean", "Policy entropy (top-K)", "nats", "bydepth_entropy_seq_mean_topk", set()),
        (df_grp, "kl_from_base_target_seq_mean", "KL from base (top-K)", "nats", "bydepth_kl_seq_mean_topk", {"ce_base"}),
    ]

    for df, col, title, ylabel, stem_name, grey in bydepth_specs:
        plots.plot_bydepth_line(
            df, col, runs=runs, title=title, ylabel=ylabel,
            stem=fig_dir / stem_name, exclude_runs=grey, rng_seed=rng_seed,
        )


def _render_paired_diff(sources, reference_run, variants, fig_dir, rng_seed):
    df_g = sources["greedy"]
    df_agg = sources["topk_aggregates"]
    df_grp = sources["topk_grouped"]

    for df, col, title, ylabel, stem_name in [
        (df_g, "semantic_distance", "Greedy semantic distance: paired diff vs ce_base", "Δ semantic_distance", "paired_diff_semantic_distance_greedy"),
        (df_g, "is_semantic_equivalent", "Greedy semantic-equiv rate: paired diff", "Δ p(equiv)", "paired_diff_semantic_equiv_greedy"),
        (df_g, "is_invalid", "Greedy invalid rate: paired diff", "Δ p(invalid)", "paired_diff_invalid_greedy"),
        (df_agg, "semantic_distance_mean_topk", "Top-K mean semantic distance: paired diff", "Δ d̄_topk", "paired_diff_semantic_distance_topk"),
        (df_agg, "semantic_equiv_rate_topk", "Top-K equiv rate: paired diff", "Δ rate_topk", "paired_diff_semantic_equiv_topk"),
        (df_grp, "self_bleu", "Top-K self-BLEU: paired diff", "Δ self_bleu", "paired_diff_self_bleu"),
        (df_grp, "policy_entropy_target_seq_mean", "Top-K policy entropy: paired diff", "Δ entropy", "paired_diff_entropy_seq_mean"),
    ]:
        plots.plot_paired_diff_multiples(
            df, col, reference_run=reference_run, variants=variants,
            title=title, ylabel=ylabel, stem=fig_dir / stem_name, rng_seed=rng_seed,
        )


def _render_holistic(summary: pd.DataFrame, runs: list[str], fig_dir: Path) -> None:
    radar_axes = [
        ("validity_greedy",         "validity (greedy)"),
        ("correctness_greedy",      "correctness (greedy)"),
        ("semantic_equiv_rate",     "equiv-rate (greedy)"),
        ("validity_topk",           "validity (top-K)"),
        ("correctness_topk",        "correctness (top-K)"),
        ("semantic_equiv_rate_topk","equiv-rate (top-K)"),
        ("diversity_topk",          "diversity (1 − self-bleu)"),
    ]
    plots.plot_radar(
        summary, runs=runs, axes_metrics=radar_axes,
        title="Cross-model radar (monotone-quality axes)",
        stem=fig_dir / "radar_overall",
    )
    plots.plot_radar(
        summary, runs=runs, axes_metrics=radar_axes,
        title="Cross-model radar (monotone-quality axes)",
        stem=fig_dir / "radar_overall_normalized",
        normalize=True,
    )
    plots.plot_pareto(
        summary, x_col="diversity_topk", y_col="correctness_topk",
        runs=runs,
        x_label="diversity (1 − self-bleu)",
        y_label="correctness (top-K)",
        title="Quality ↔ diversity",
        higher_is_better=(True, True),
        stem=fig_dir / "pareto_quality_diversity",
    )
    plots.plot_pareto(
        summary, x_col="syntax_semantics_gap_greedy", y_col="semantic_equiv_rate",
        runs=runs,
        x_label="syntax-semantics gap (greedy)",
        y_label="semantic-equiv rate (greedy)",
        title="Quality ↔ syntactic creativity",
        higher_is_better=(True, True),
        stem=fig_dir / "pareto_quality_creativity",
    )
    plots.plot_pareto(
        summary, x_col="depth_degradation_slope", y_col="semantic_distance_greedy_mean",
        runs=runs,
        x_label="depth-degradation slope (semantic_distance vs depth)",
        y_label="overall greedy semantic distance",
        title="Quality ↔ depth-degradation (lower-left = best)",
        higher_is_better=(False, False),
        stem=fig_dir / "pareto_quality_depth_degradation",
    )
    plots.plot_kl_drift_vs_quality(summary, runs=runs, stem=fig_dir / "kl_drift_vs_quality")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    stats_dir = out / "stats"
    diag_dir = out / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    pad_id = _resolve_pad_token_id(args.tokenizer_dir)
    print(f"[visualize_validation] pad_token_id={pad_id}", file=sys.stderr)

    loaded = loaders.load_runs(
        validation_root=Path(args.validation_root),
        runs=args.runs,
        pad_token_id=pad_id,
        drop_token_arrays=not args.keep_token_arrays,
        dataset_dir=args.dataset_dir,
    )
    sources = {
        "greedy": loaded["df_greedy"],
        "topk_flat": loaded["df_topk_flat"],
        "topk_grouped": loaded["df_topk_grouped"],
        "topk_aggregates": loaded["df_topk_aggregates"],
    }
    runs = list(args.runs)
    variants = [r for r in runs if r != args.reference_run]

    summary_per_run = build_per_run_summary(sources, runs)
    summary_per_run.to_csv(out / "per_run_summary.csv", index=False)

    descriptives = stats.descriptives_by_run(
        sources["greedy"],
        columns=[
            "semantic_distance", "is_invalid", "is_exact_match", "is_semantic_equivalent",
            "generated_depth", "generated_length_tokens",
            "seq_entropy_mean", "seq_kl_mean",
        ],
    )
    descriptives_topk = stats.descriptives_by_run(
        sources["topk_aggregates"],
        columns=[
            "semantic_distance_mean_topk", "semantic_distance_variance_topk",
            "invalid_rate_topk", "exact_match_rate_topk", "semantic_equiv_rate_topk",
            "syntax_semantics_gap_topk",
            "generated_depth_mean_topk", "generated_length_tokens_mean_topk",
            "mc_se_topk",
        ],
    )
    descriptives_grouped = stats.descriptives_by_run(
        sources["topk_grouped"],
        columns=[
            "self_bleu",
            "policy_entropy_target_seq_mean", "policy_entropy_target_token_mean",
            "kl_from_base_target_seq_mean", "kl_from_base_target_token_mean",
        ],
    )
    descriptives_all = pd.concat([descriptives, descriptives_topk, descriptives_grouped], ignore_index=True)
    descriptives_all.to_csv(stats_dir / "descriptives_by_run.csv", index=False)

    pairwise = pd.DataFrame()
    pairwise_by_depth = pd.DataFrame()
    diagnostics = pd.DataFrame()
    if not args.no_stats:
        rng = np.random.default_rng(args.rng_seed)
        print("[visualize_validation] running pairwise stats (overall)...", file=sys.stderr)
        pairwise = stats.run_pairwise_grid(
            sources=sources, specs=ALL_SPECS, variants=runs,
            reference_run=args.reference_run,
            n_resamples=args.bootstrap_n, alpha=args.alpha,
            rng=rng, by_depth=False,
        )
        if not pairwise.empty:
            pairwise.to_csv(stats_dir / "pairwise_vs_reference.csv", index=False)

        print("[visualize_validation] running pairwise stats (by depth)...", file=sys.stderr)
        pairwise_by_depth = stats.run_pairwise_grid(
            sources=sources, specs=ALL_SPECS, variants=runs,
            reference_run=args.reference_run,
            n_resamples=args.bootstrap_n, alpha=args.alpha,
            rng=rng, by_depth=True,
        )
        if not pairwise_by_depth.empty:
            pairwise_by_depth.to_csv(stats_dir / "per_depth_pairwise_vs_reference.csv", index=False)

        diagnostics = stats.paired_diff_diagnostics(
            sources, ALL_SPECS, runs, args.reference_run,
        )
        if not diagnostics.empty:
            diagnostics.to_csv(stats_dir / "paired_diff_diagnostics.csv", index=False)

    if not args.no_figures:
        print("[visualize_validation] rendering cross-model figures...", file=sys.stderr)
        _render_cross_model_figures(sources, runs, fig_dir, args.rng_seed)
        _render_ecdfs(sources, runs, fig_dir, args.rng_seed)
        _render_bydepth(sources, runs, fig_dir, args.rng_seed)
        _render_paired_diff(sources, args.reference_run, variants, fig_dir, args.rng_seed)
        _render_holistic(summary_per_run, runs, fig_dir)

        if not diagnostics.empty:
            plots.plot_paired_diff_diagnostics(
                None, ALL_SPECS,
                reference_run=args.reference_run, variants=variants,
                sources=sources,
                stem=diag_dir / "wilcoxon_assumption_diagnostics",
            )

    metadata = {
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "reference_run": args.reference_run,
        "bootstrap_n": args.bootstrap_n,
        "alpha": args.alpha,
        "mc_correction": args.mc_correction,
        "rng_seed": args.rng_seed,
        "pad_token_id": pad_id,
        "n_dataset_samples": int(len(sources["greedy"]["formula_id"].unique())),
        "validation_summary": loaded.get("validation_summary", {}),
    }
    with open(out / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    md_path = report.write_summary(
        output_dir=out,
        runs=runs,
        reference_run=args.reference_run,
        descriptives=descriptives_all,
        pairwise=pairwise,
        pairwise_by_depth=pairwise_by_depth,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    print(f"[visualize_validation] wrote {md_path}", file=sys.stderr)
    print(f"[visualize_validation] done. all artifacts under {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
