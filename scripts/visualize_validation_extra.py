"""Extra cross-model validation analyses, complementary to ``visualize_validation.py``.

Adds conditional descriptive metrics (depth/length gaps and semantic distance
sliced by correctness), top-K diagnostics (pass@k', distinct-correct counts),
and target-side operator analysis (KL, decomposition, log-odds ratios,
logistic regression).

Usage::

    python scripts/visualize_validation_extra.py \\
        --validation-root artifacts/validation \\
        --runs ce_base ce_finetune rb_momentum_09 gae_lambda_09 gae_lambda_1 \\
        --reference-run ce_base \\
        --tokenizer-dir artifacts/tokenizer \\
        --output-dir artifacts/validation/_analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._validation_analysis import (
    extra_contrast, extra_metrics, extra_plots, flexibility_metrics, loaders,
    operator_analysis, operator_crossmodel,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--validation-root", default="artifacts/validation")
    p.add_argument("--dataset-dir", default="artifacts/datasets/validation",
                   help="Validation dataset dir; its trivial_ids.csv is auto-used to drop "
                        "tautology/contradiction targets from all results.")
    p.add_argument("--output-dir", default="artifacts/validation/_analysis")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--reference-run", default="ce_base")
    p.add_argument("--tokenizer-dir", default=None,
                   help="Tokenizer directory for pad_token_id resolution.")
    p.add_argument("--bootstrap-n", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--keep-token-arrays", action="store_true")
    p.add_argument("--logistic-regularized-fallback", action="store_true",
                   help="If unregularized Logit fails to converge, fall back to L2-regularized fit.")
    return p.parse_args()


def _resolve_pad_token_id(tokenizer_dir: str | None) -> int:
    if tokenizer_dir is None:
        return 0
    try:
        from tokenizer_pretrained_class import LTLTokenizer  # type: ignore
        tok = LTLTokenizer.from_pretrained(tokenizer_dir)
        return int(tok.pad_token_id)
    except Exception as e:  # pragma: no cover
        print(f"[visualize_validation_extra] WARNING: tokenizer load failed ({e}); pad_token_id=0",
              file=sys.stderr)
        return 0


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    fig_dir = out / "figures" / "extra"
    stats_dir = out / "stats" / "extra"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    pad_id = _resolve_pad_token_id(args.tokenizer_dir)
    loaded = loaders.load_runs(
        validation_root=Path(args.validation_root),
        runs=args.runs,
        pad_token_id=pad_id,
        drop_token_arrays=not args.keep_token_arrays,
        dataset_dir=args.dataset_dir,
    )
    df_greedy = loaded["df_greedy"]
    df_topk_flat = loaded["df_topk_flat"]
    df_topk_grouped = loaded["df_topk_grouped"]
    df_topk_aggregates = loaded["df_topk_aggregates"]
    runs = list(args.runs)

    # target_length_tokens lives only on the greedy frame in the loader output.
    # Merge it into the top-K flat frame so the conditional length-diff metrics
    # work on both sources.
    df_topk_flat = df_topk_flat.merge(
        df_greedy[["run", "formula_id", "target_length_tokens"]],
        on=["run", "formula_id"], how="left",
    )

    print("[extra] computing conditional metrics...", file=sys.stderr)
    _run_conditional_metrics(df_greedy, df_topk_flat, runs, fig_dir, stats_dir, args)

    print("[extra] computing top-K diagnostics (pass@k', distinct-correct)...", file=sys.stderr)
    _run_topk_diagnostics(df_topk_flat, runs, fig_dir, stats_dir, args)

    print("[extra] computing RQ2 flexibility + graceful-degradation (I4/I5/I6)...", file=sys.stderr)
    _run_flexibility_metrics(df_topk_flat, df_greedy, runs, fig_dir, stats_dir, args)

    print("[extra] computing target-side operator analysis...", file=sys.stderr)
    _run_operator_analysis(df_greedy, df_topk_flat, runs, fig_dir, stats_dir, args)

    print("[extra] computing contrast studies (paired-diff, agreement, output similarity)...", file=sys.stderr)
    _run_contrast_studies(df_greedy, df_topk_flat, runs, fig_dir, stats_dir, args)

    print("[extra] computing unconditional paired-diff coverage (overall + by-depth)...", file=sys.stderr)
    _run_unconditional_paired_diffs(
        sources={
            "greedy": df_greedy,
            "topk_aggregates": df_topk_aggregates,
            "topk_grouped": df_topk_grouped,
        },
        runs=runs,
        fig_dir=fig_dir,
        stats_dir=stats_dir,
        args=args,
    )

    print("[extra] computing fair cross-model operator comparison "
          "(pooled interaction + AME + stratified McNemar)...", file=sys.stderr)
    _run_crossmodel_operator_comparison(
        df_greedy, df_topk_flat, runs, fig_dir, stats_dir, args,
    )

    metadata = {
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "reference_run": args.reference_run,
        "bootstrap_n": args.bootstrap_n,
        "alpha": args.alpha,
        "rng_seed": args.rng_seed,
        "pad_token_id": pad_id,
    }
    with open(out / "run_metadata_extra.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[extra] done. figures → {fig_dir}, stats → {stats_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Conditional descriptive metrics
# ---------------------------------------------------------------------------


def _run_conditional_metrics(
    df_greedy: pd.DataFrame,
    df_topk_flat: pd.DataFrame,
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    # Spec: (source_name, df, condition_fn, value_or_diff, cols, ylabel, suffix)
    sources = [("greedy", df_greedy), ("topk", df_topk_flat)]

    # 1. gen_depth - tgt_depth | correct
    # 2. gen_length - tgt_length | correct
    # 3. gen_depth - tgt_depth | wrong & valid
    # 4. gen_length - tgt_length | wrong & valid
    # 5. semantic_distance | wrong & valid
    diff_specs = [
        ("depth_correct",       "generated_depth",          "target_depth",
         extra_metrics.cond_correct,         "Δ depth (gen − target) | correct"),
        ("length_correct",      "generated_length_tokens",  "target_length_tokens",
         extra_metrics.cond_correct,         "Δ length (gen − target) | correct"),
        ("depth_wrong_valid",   "generated_depth",          "target_depth",
         extra_metrics.cond_wrong_and_valid, "Δ depth (gen − target) | wrong & valid"),
        ("length_wrong_valid",  "generated_length_tokens",  "target_length_tokens",
         extra_metrics.cond_wrong_and_valid, "Δ length (gen − target) | wrong & valid"),
    ]

    for src_name, df in sources:
        for name, gen_col, tgt_col, cond, ylabel in diff_specs:
            overall = extra_metrics.conditional_diff_stats(
                df, gen_col=gen_col, tgt_col=tgt_col, condition_fn=cond,
                runs=runs, by_depth=False,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
            )
            bydepth = extra_metrics.conditional_diff_stats(
                df, gen_col=gen_col, tgt_col=tgt_col, condition_fn=cond,
                runs=runs, by_depth=True,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
            )
            overall.to_csv(stats_dir / f"cond_{name}_{src_name}_overall.csv", index=False)
            bydepth.to_csv(stats_dir / f"cond_{name}_{src_name}_bydepth.csv", index=False)
            extra_plots.plot_conditional_bar(
                overall, runs=runs, title=f"{ylabel} ({src_name})",
                ylabel=ylabel,
                stem=fig_dir / f"cond_{name}_{src_name}_bar",
            )
            extra_plots.plot_conditional_bydepth(
                bydepth, runs=runs, title=f"{ylabel} by target_depth ({src_name})",
                ylabel=ylabel,
                stem=fig_dir / f"cond_{name}_{src_name}_bydepth",
            )

        # semantic_distance | wrong & valid
        overall_sd = extra_metrics.conditional_value_stats(
            df, value_col="semantic_distance",
            condition_fn=extra_metrics.cond_wrong_and_valid,
            runs=runs, by_depth=False,
            n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
        )
        bydepth_sd = extra_metrics.conditional_value_stats(
            df, value_col="semantic_distance",
            condition_fn=extra_metrics.cond_wrong_and_valid,
            runs=runs, by_depth=True,
            n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
        )
        overall_sd.to_csv(stats_dir / f"cond_semdist_wrong_valid_{src_name}_overall.csv", index=False)
        bydepth_sd.to_csv(stats_dir / f"cond_semdist_wrong_valid_{src_name}_bydepth.csv", index=False)
        extra_plots.plot_conditional_bar(
            overall_sd, runs=runs,
            title=f"semantic_distance | wrong & valid ({src_name})",
            ylabel="semantic_distance",
            stem=fig_dir / f"cond_semdist_wrong_valid_{src_name}_bar",
            hline=None,
        )
        extra_plots.plot_conditional_bydepth(
            bydepth_sd, runs=runs,
            title=f"semantic_distance | wrong & valid, by target_depth ({src_name})",
            ylabel="semantic_distance",
            stem=fig_dir / f"cond_semdist_wrong_valid_{src_name}_bydepth",
            hline=None,
        )


# ---------------------------------------------------------------------------
# pass@k' and distinct-correct
# ---------------------------------------------------------------------------


def _run_topk_diagnostics(
    df_topk_flat: pd.DataFrame,
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    pak = extra_metrics.compute_pass_at_k_curve(
        df_topk_flat, runs=runs,
        n_bootstrap=max(1000, args.bootstrap_n // 2),
        alpha=args.alpha, rng_seed=args.rng_seed,
    )
    pak.to_csv(stats_dir / "pass_at_k.csv", index=False)
    extra_plots.plot_pass_at_k_curve(pak, runs=runs, stem=fig_dir / "pass_at_k_curve")

    distinct_all = extra_metrics.compute_distinct_correct_stats(
        df_topk_flat, runs=runs, conditional_on_any_correct=False,
        n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
    )
    distinct_any = extra_metrics.compute_distinct_correct_stats(
        df_topk_flat, runs=runs, conditional_on_any_correct=True,
        n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
    )
    distinct_all.to_csv(stats_dir / "distinct_correct_all_targets.csv", index=False)
    distinct_any.to_csv(stats_dir / "distinct_correct_any_correct.csv", index=False)
    extra_plots.plot_distinct_correct_bar(
        distinct_all, runs=runs,
        title="Distinct correct generations per target (top-K)",
        ylabel="# distinct correct (mean)",
        stem=fig_dir / "distinct_correct_all_targets",
    )
    extra_plots.plot_distinct_correct_bar(
        distinct_any, runs=runs,
        title="Distinct correct generations per target | ≥1 correct (top-K)",
        ylabel="# distinct correct (mean | ≥1)",
        stem=fig_dir / "distinct_correct_any_correct",
    )


# ---------------------------------------------------------------------------
# RQ2 flexibility (I4 spread / I5 distinct-correct) + graceful degradation (I6)
# ---------------------------------------------------------------------------


def _run_flexibility_metrics(
    df_topk_flat: pd.DataFrame,
    df_greedy: pd.DataFrame,
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    ref = args.reference_run
    variants = [r for r in runs if r != ref]
    bn, al, seed = args.bootstrap_n, args.alpha, args.rng_seed

    # --- I5: distinct-correct as a paired contrast (conditional on >=1 correct) ---
    dc = flexibility_metrics.distinct_correct_per_target(
        df_topk_flat, runs=runs, conditional_on_any_correct=True)
    if not dc.empty:
        dc_sum, dc_pt = extra_contrast.compute_paired_diff_summary(
            dc, reference_run=ref, variants=variants, n_bootstrap=bn, alpha=al, rng_seed=seed)
        dc_sum.to_csv(stats_dir / "flex_distinct_correct_paired.csv", index=False)
        extra_plots.plot_paired_diff_conditional(
            dc_pt, dc_sum, variants=variants, runs=runs, reference_run=ref,
            title=f"Paired Δ vs {ref} — # distinct correct | ≥1 correct",
            ylabel="Δ # distinct correct", stem=fig_dir / "flex_distinct_correct_paired",
            rng_seed=seed)

    # --- I4: correct-only equivalence-class spread (tree-edit distance, target-weighted) ---
    sp = flexibility_metrics.correct_only_spread_per_target(df_topk_flat)
    if not sp.empty:
        sp.to_csv(stats_dir / "flex_spread_per_target.csv", index=False)
        sp_desc, _ = flexibility_metrics.correct_only_spread_contrast(
            sp, runs=runs, reference_run=ref, n_bootstrap=bn, alpha=al, rng_seed=seed)
        sp_desc.to_csv(stats_dir / "flex_spread_descriptive.csv", index=False)
        extra_plots.plot_conditional_bar(
            sp_desc.rename(columns={"mean_spread": "mean"}), runs=runs,
            title="Correct-only equivalence-class spread (target-weighted TED, ≥2 correct)",
            ylabel="mean pairwise tree-edit distance",
            stem=fig_dir / "flex_spread_descriptive", hline=None)
        sp_sum, sp_pt = extra_contrast.compute_paired_diff_summary(
            sp[["run", "formula_id", "target_depth", "value"]],
            reference_run=ref, variants=variants, n_bootstrap=bn, alpha=al, rng_seed=seed)
        sp_sum.to_csv(stats_dir / "flex_spread_paired.csv", index=False)
        extra_plots.plot_paired_diff_conditional(
            sp_pt, sp_sum, variants=variants, runs=runs, reference_run=ref,
            title=f"Paired Δ vs {ref} — correct-only spread (≥2 correct in both)",
            ylabel="Δ mean pairwise TED", stem=fig_dir / "flex_spread_paired", rng_seed=seed)

    # --- I6: wrong-and-valid set overlap (Jaccard) + paired graceful-degradation (greedy) ---
    overlap = flexibility_metrics.wrong_valid_overlap(df_greedy, runs=runs)
    if not overlap.empty:
        overlap.to_csv(stats_dir / "flex_wrong_valid_overlap.csv", index=False)
        extra_plots.plot_output_similarity_heatmap(
            overlap, runs=runs, metric_col="jaccard",
            title="Wrong-and-valid set overlap (Jaccard, greedy)",
            stem=fig_dir / "flex_wrong_valid_overlap", vmin=0.0, vmax=1.0, cmap="magma")
    wv = flexibility_metrics.wrong_valid_distance_per_target(df_greedy, runs=runs)
    if not wv.empty:
        wv_sum, wv_pt = extra_contrast.compute_paired_diff_summary(
            wv, reference_run=ref, variants=variants, n_bootstrap=bn, alpha=al, rng_seed=seed)
        wv_sum.to_csv(stats_dir / "flex_graceful_degradation_paired.csv", index=False)
        extra_plots.plot_paired_diff_conditional(
            wv_pt, wv_sum, variants=variants, runs=runs, reference_run=ref,
            title=f"Paired Δ vs {ref} — semantic_distance | both wrong-and-valid (greedy)",
            ylabel="Δ semantic_distance", stem=fig_dir / "flex_graceful_degradation_paired",
            rng_seed=seed)


# ---------------------------------------------------------------------------
# Operator analysis (target-side)
# ---------------------------------------------------------------------------


def _run_operator_analysis(
    df_greedy: pd.DataFrame,
    df_topk_flat: pd.DataFrame,
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    # Two outcome sources: greedy (one row per target) and topk_any (any-of-K correct)
    df_op_greedy = operator_analysis.build_target_operator_frame(df_greedy)
    df_op_topk = operator_analysis.build_target_operator_frame_topk(df_topk_flat, df_greedy)

    for src_name, df_op in [("greedy", df_op_greedy), ("topk_any", df_op_topk)]:
        # KL + per-op contributions (Case B: summed per-operator Bernoulli KL)
        kl_df = operator_analysis.compute_kl_per_run(df_op, runs=runs)
        kl_df.to_csv(stats_dir / f"op_kl_{src_name}.csv", index=False)
        extra_plots.plot_kl_per_run(kl_df, runs=runs, stem=fig_dir / f"op_kl_{src_name}")
        extra_plots.plot_kl_contribution_per_run(
            kl_df, runs=runs, stem=fig_dir / f"op_kl_contribution_{src_name}",
        )

        # Case-B independence diagnostic: per-subset 8x8 phi co-occurrence
        cooc_df = operator_analysis.compute_operator_cooccurrence(df_op, runs=runs)
        cooc_df.to_csv(stats_dir / f"op_cooccurrence_{src_name}.csv", index=False)
        extra_plots.plot_operator_cooccurrence(
            cooc_df, runs=runs, stem=fig_dir / f"op_cooccurrence_{src_name}",
        )

        # Decomposition: P(op | correct/wrong) with bootstrap CIs + base-rate tick
        decomp_df = operator_analysis.compute_op_decomposition(
            df_op, runs=runs,
            n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
        )
        decomp_df.to_csv(stats_dir / f"op_decomposition_{src_name}.csv", index=False)
        extra_plots.plot_op_decomposition(
            decomp_df, runs=runs, stem=fig_dir / f"op_decomposition_{src_name}",
        )

        # Marginal log-odds-ratio forest
        lor_df = operator_analysis.compute_log_odds_ratios(
            df_op, runs=runs,
            n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
        )
        lor_df.to_csv(stats_dir / f"op_log_odds_{src_name}.csv", index=False)
        extra_plots.plot_log_odds_forest(
            lor_df, runs=runs, stem=fig_dir / f"op_log_odds_forest_{src_name}",
        )

        # Adjusted logistic regression — BH-FDR over its (source, adjusted_op_effects) family
        coef_df = operator_analysis.compute_logistic_regression(
            df_op, runs=runs, alpha=args.alpha,
            use_regularized=args.logistic_regularized_fallback,
        )
        coef_df = _bh_fdr_operator_families(coef_df, alpha=args.alpha)
        coef_df.to_csv(stats_dir / f"op_logistic_coef_{src_name}.csv", index=False)
        extra_plots.plot_logistic_coefficient_forest(
            coef_df, runs=runs, stem=fig_dir / f"op_logistic_forest_{src_name}",
        )
        # Overlay marginal vs adjusted
        if not lor_df.empty and not coef_df.empty:
            extra_plots.plot_log_odds_vs_regression_overlay(
                lor_df, coef_df, runs=runs,
                stem=fig_dir / f"op_marginal_vs_adjusted_{src_name}",
            )


# ---------------------------------------------------------------------------
# Contrast studies (paired-diff, agreement, output similarity) + BH-FDR
# ---------------------------------------------------------------------------


# Conditional metric specs reused across sources.
# (name, source-applicable, value_col-or-(gen,tgt), condition_fn, ylabel)
CONDITIONAL_PAIRED_SPECS: list[tuple] = [
    ("depth_correct",      None, ("generated_depth", "target_depth"),
     extra_metrics.cond_correct,         "Δ depth (gen − target) | correct"),
    ("length_correct",     None, ("generated_length_tokens", "target_length_tokens"),
     extra_metrics.cond_correct,         "Δ length (gen − target) | correct"),
    ("depth_wrong_valid",  None, ("generated_depth", "target_depth"),
     extra_metrics.cond_wrong_and_valid, "Δ depth (gen − target) | wrong & valid"),
    ("length_wrong_valid", None, ("generated_length_tokens", "target_length_tokens"),
     extra_metrics.cond_wrong_and_valid, "Δ length (gen − target) | wrong & valid"),
    ("semdist_wrong_valid", "semantic_distance", None,
     extra_metrics.cond_wrong_and_valid, "semantic_distance | wrong & valid"),
]


def _run_contrast_studies(
    df_greedy: pd.DataFrame,
    df_topk_flat: pd.DataFrame,
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    reference_run = args.reference_run
    variants = [r for r in runs if r != reference_run]

    sources = [("greedy", df_greedy), ("topk", df_topk_flat)]

    # --- 1. Conditional paired-diffs --------------------------------------------
    # Per (source) we accumulate one BH-FDR family across all metrics × variants.
    for src_name, df_src in sources:
        family_rows = []  # for BH-FDR adjustment across this (source, conditional) family
        family_pvals = []
        family_keys = []  # parallel: (metric_name, variant) to re-attach adjusted p

        for metric_name, value_col, diff_cols, condition_fn, ylabel in CONDITIONAL_PAIRED_SPECS:
            per_target = extra_contrast.compute_per_target_conditional_value(
                df_src,
                value_col=value_col,
                diff_cols=diff_cols,
                condition_fn=condition_fn,
            )
            if per_target.empty:
                continue
            summary, per_target_diffs = extra_contrast.compute_paired_diff_summary(
                per_target, reference_run=reference_run, variants=variants,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
            )
            bydepth = extra_contrast.compute_paired_diff_by_depth(
                per_target, reference_run=reference_run, variants=variants,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
            )

            # Attach metric label and stash p-values for BH-FDR.
            summary = summary.assign(metric=metric_name, source=src_name)
            bydepth = (bydepth.assign(metric=metric_name, source=src_name)
                       if not bydepth.empty else bydepth)
            family_rows.append(summary)
            for _, row in summary.iterrows():
                family_pvals.append(row["wilcoxon_p"])
                family_keys.append((metric_name, row["variant"]))

            # Per-metric figures + per-target CSV
            extra_plots.plot_paired_diff_conditional(
                per_target_diffs, summary,
                variants=variants, runs=runs, reference_run=reference_run,
                title=f"Paired Δ vs {reference_run} — {ylabel} ({src_name})",
                ylabel=ylabel,
                stem=fig_dir / f"contrast_paired_diff_{metric_name}_{src_name}",
                rng_seed=args.rng_seed,
            )
            if not bydepth.empty:
                extra_plots.plot_paired_diff_bydepth(
                    bydepth, variants=variants, runs=runs, reference_run=reference_run,
                    title=f"Paired Δ vs {reference_run} by target_depth — {ylabel} ({src_name})",
                    ylabel=ylabel,
                    stem=fig_dir / f"contrast_paired_diff_{metric_name}_{src_name}_bydepth",
                )
            per_target_diffs.assign(metric=metric_name, source=src_name).to_csv(
                stats_dir / f"contrast_paired_diff_{metric_name}_{src_name}_per_target.csv",
                index=False,
            )
            bydepth.to_csv(
                stats_dir / f"contrast_paired_diff_{metric_name}_{src_name}_bydepth.csv",
                index=False,
            ) if not bydepth.empty else None

        # BH-FDR across the (source, conditional_paired_diff) family
        if family_rows:
            family_df = pd.concat(family_rows, ignore_index=True)
            adj_p, reject = extra_contrast.apply_bh_fdr(family_pvals, alpha=args.alpha)
            # Re-attach via the parallel key list.
            adj_lookup = {k: (p, r) for k, p, r in zip(family_keys, adj_p, reject)}
            family_df["wilcoxon_p_adj_bh"] = [
                adj_lookup.get((m, v), (float("nan"), False))[0]
                for m, v in zip(family_df["metric"], family_df["variant"])
            ]
            family_df["reject_bh"] = [
                adj_lookup.get((m, v), (float("nan"), False))[1]
                for m, v in zip(family_df["metric"], family_df["variant"])
            ]
            family_df.to_csv(
                stats_dir / f"contrast_paired_diff_summary_{src_name}_bhfdr.csv",
                index=False,
            )

    # --- 2. Pairwise correctness agreement (Cohen's κ + McNemar) -----------------
    corr_sources = [
        ("greedy",   extra_contrast.correctness_long_greedy(df_greedy)),
        ("topk_any", extra_contrast.correctness_long_topk_any(df_topk_flat)),
    ]
    for src_name, corr_long in corr_sources:
        agreement = extra_contrast.compute_pairwise_agreement(corr_long, runs=runs)
        if agreement.empty:
            continue
        # BH-FDR within this (source, agreement_mcnemar) family — only one
        # p-value per *unordered* pair to avoid double counting.
        unordered_mask = agreement["run_a"] < agreement["run_b"]
        unordered_pairs = agreement[unordered_mask]
        adj_p, reject = extra_contrast.apply_bh_fdr(
            unordered_pairs["mcnemar_p"].tolist(), alpha=args.alpha,
        )
        adj_lookup = {
            (a, b): (p, r) for (a, b), p, r in zip(
                zip(unordered_pairs["run_a"], unordered_pairs["run_b"]), adj_p, reject,
            )
        }
        # Symmetrise: map (a,b) and (b,a) to the same adjusted p.
        def _adj_for(row):
            a, b = row["run_a"], row["run_b"]
            if (a, b) in adj_lookup:
                return adj_lookup[(a, b)]
            if (b, a) in adj_lookup:
                return adj_lookup[(b, a)]
            return float("nan"), False

        agreement["mcnemar_p_adj"] = [_adj_for(r)[0] for _, r in agreement.iterrows()]
        agreement["mcnemar_reject_bh"] = [_adj_for(r)[1] for _, r in agreement.iterrows()]
        agreement.to_csv(
            stats_dir / f"contrast_agreement_{src_name}.csv", index=False,
        )
        extra_plots.plot_agreement_kappa_heatmap(
            agreement, runs=runs,
            stem=fig_dir / f"contrast_agreement_kappa_{src_name}",
            title=f"Pairwise Cohen's κ — correctness agreement ({src_name})",
        )
        extra_plots.plot_agreement_mcnemar_heatmap(
            agreement, runs=runs,
            stem=fig_dir / f"contrast_agreement_mcnemar_{src_name}",
            significance_col="mcnemar_p_adj", alpha=args.alpha,
            title=f"McNemar signed advantage of row over column ({src_name})",
        )

    # --- 3. Pairwise output similarity (greedy generations) ----------------------
    sim_df = extra_contrast.compute_pairwise_output_similarity(df_greedy, runs=runs)
    if not sim_df.empty:
        sim_df.to_csv(stats_dir / "contrast_output_similarity_greedy.csv", index=False)
        extra_plots.plot_output_similarity_heatmap(
            sim_df, runs=runs, metric_col="mean_bleu",
            title="Pairwise output similarity — greedy BLEU-4",
            stem=fig_dir / "contrast_output_similarity_bleu_greedy",
            vmin=0.0, vmax=1.0, cmap="viridis",
        )
        extra_plots.plot_output_similarity_heatmap(
            sim_df, runs=runs, metric_col="exact_match_rate",
            title="Pairwise greedy exact-match rate",
            stem=fig_dir / "contrast_output_similarity_exact_greedy",
            vmin=0.0, vmax=1.0, cmap="Blues",
        )


# ---------------------------------------------------------------------------
# Unconditional paired-diff coverage (overall + by-depth) for all metrics
# that the main analysis visualises as boxes/bars but never gave paired-diff
# treatment, plus by-depth versions of the original 7 paired-diff metrics.
# Family structure for BH-FDR: one family per source (greedy / topk).
# ---------------------------------------------------------------------------

# (name, source_key, value_col, ylabel, require_valid)
UNCONDITIONAL_PAIRED_SPECS: list[tuple] = [
    # --- Greedy: original paired-diff metrics (by-depth was missing) ---
    ("semantic_distance",             "greedy",          "semantic_distance",                  "Δ semantic_distance",              False),
    ("is_invalid",                    "greedy",          "is_invalid",                         "Δ p(invalid)",                     False),
    ("is_semantic_equivalent",        "greedy",          "is_semantic_equivalent",             "Δ p(equiv)",                       False),
    # --- Greedy: paired-diff entirely missing ---
    ("is_exact_match",                "greedy",          "is_exact_match",                     "Δ p(exact match)",                 False),
    ("generated_depth",               "greedy",          "generated_depth",                    "Δ generated_depth",                True),
    ("generated_length_tokens",       "greedy",          "generated_length_tokens",            "Δ generated_length_tokens",        True),
    ("seq_entropy_mean",              "greedy",          "seq_entropy_mean",                   "Δ policy entropy (seq mean)",      False),
    # --- Top-K aggregates: original paired-diff metrics (by-depth was missing) ---
    ("semantic_distance_mean_topk",   "topk_aggregates", "semantic_distance_mean_topk",        "Δ SD_topk (mean over K)",          False),
    ("semantic_equiv_rate_topk",      "topk_aggregates", "semantic_equiv_rate_topk",           "Δ equiv-rate (top-K)",             False),
    # --- Top-K aggregates: paired-diff entirely missing ---
    ("semantic_distance_variance_topk","topk_aggregates","semantic_distance_variance_topk",    "Δ within-target SD variance",      False),
    ("invalid_rate_topk",             "topk_aggregates", "invalid_rate_topk",                  "Δ invalid_rate (top-K)",           False),
    ("exact_match_rate_topk",         "topk_aggregates", "exact_match_rate_topk",              "Δ exact-match rate (top-K)",       False),
    ("syntax_semantics_gap_topk",     "topk_aggregates", "syntax_semantics_gap_topk",          "Δ syntax-semantics gap (top-K)",   False),
    ("generated_depth_mean_topk",     "topk_aggregates", "generated_depth_mean_topk",          "Δ generated_depth (top-K mean)",   False),
    ("generated_length_tokens_mean_topk","topk_aggregates","generated_length_tokens_mean_topk","Δ generated_length (top-K mean)",  False),
    # --- Top-K grouped: original paired-diff metrics (by-depth was missing) ---
    ("self_bleu",                     "topk_grouped",    "self_bleu",                          "Δ self-BLEU",                      False),
    ("policy_entropy_target_seq_mean","topk_grouped",    "policy_entropy_target_seq_mean",     "Δ policy entropy (top-K)",         False),
]


def _run_unconditional_paired_diffs(
    sources: dict[str, pd.DataFrame],
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    reference_run = args.reference_run
    variants = [r for r in runs if r != reference_run]

    # Group specs by source for one BH-FDR family per (source, unconditional).
    by_source: dict[str, list[tuple]] = {}
    for spec in UNCONDITIONAL_PAIRED_SPECS:
        by_source.setdefault(spec[1], []).append(spec)
    # Greedy is its own family; topk_aggregates + topk_grouped pool into "topk".
    family_groups = {
        "greedy": by_source.get("greedy", []),
        "topk":   by_source.get("topk_aggregates", []) + by_source.get("topk_grouped", []),
    }

    for family_name, specs in family_groups.items():
        family_summaries: list[pd.DataFrame] = []
        family_pvals: list[float] = []
        family_keys: list[tuple[str, str]] = []

        for name, src_key, value_col, ylabel, require_valid in specs:
            df_src = sources.get(src_key)
            if df_src is None or value_col not in df_src.columns:
                continue
            cond_fn = extra_metrics.cond_valid if require_valid else extra_metrics.cond_always
            per_target = extra_contrast.compute_per_target_conditional_value(
                df_src,
                value_col=value_col,
                condition_fn=cond_fn,
            )
            if per_target.empty:
                continue
            summary, per_target_diffs = extra_contrast.compute_paired_diff_summary(
                per_target, reference_run=reference_run, variants=variants,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
            )
            bydepth = extra_contrast.compute_paired_diff_by_depth(
                per_target, reference_run=reference_run, variants=variants,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, rng_seed=args.rng_seed,
            )

            # Avoid a redundant "_topk_topk" when the metric name already
            # encodes the source (e.g. "semantic_distance_mean_topk").
            stem_label = (
                name if name.endswith(f"_{family_name}")
                else f"{name}_{family_name}"
            )

            summary = summary.assign(metric=name, source=family_name,
                                     require_valid=require_valid)
            if not bydepth.empty:
                bydepth = bydepth.assign(metric=name, source=family_name)
            family_summaries.append(summary)
            for _, row in summary.iterrows():
                family_pvals.append(row["wilcoxon_p"])
                family_keys.append((name, row["variant"]))

            extra_plots.plot_paired_diff_conditional(
                per_target_diffs, summary,
                variants=variants, runs=runs, reference_run=reference_run,
                title=f"Paired Δ vs {reference_run} — {ylabel} ({family_name})",
                ylabel=ylabel,
                stem=fig_dir / f"paired_diff_uncond_{stem_label}",
                rng_seed=args.rng_seed,
            )
            if not bydepth.empty:
                extra_plots.plot_paired_diff_bydepth(
                    bydepth, variants=variants, runs=runs, reference_run=reference_run,
                    title=f"Paired Δ vs {reference_run} by target_depth — {ylabel} ({family_name})",
                    ylabel=ylabel,
                    stem=fig_dir / f"paired_diff_uncond_{stem_label}_bydepth",
                )

            per_target_diffs.assign(metric=name, source=family_name).to_csv(
                stats_dir / f"paired_diff_uncond_{stem_label}_per_target.csv",
                index=False,
            )
            if not bydepth.empty:
                bydepth.to_csv(
                    stats_dir / f"paired_diff_uncond_{stem_label}_bydepth.csv",
                    index=False,
                )

        # BH-FDR over this (source, unconditional_paired_diffs) family.
        if family_summaries:
            family_df = pd.concat(family_summaries, ignore_index=True)
            adj_p, reject = extra_contrast.apply_bh_fdr(family_pvals, alpha=args.alpha)
            adj_lookup = {k: (p, r) for k, p, r in zip(family_keys, adj_p, reject)}
            family_df["wilcoxon_p_adj_bh"] = [
                adj_lookup.get((m, v), (float("nan"), False))[0]
                for m, v in zip(family_df["metric"], family_df["variant"])
            ]
            family_df["reject_bh"] = [
                adj_lookup.get((m, v), (float("nan"), False))[1]
                for m, v in zip(family_df["metric"], family_df["variant"])
            ]
            family_df.to_csv(
                stats_dir / f"paired_diff_uncond_summary_{family_name}_bhfdr.csv",
                index=False,
            )


# ---------------------------------------------------------------------------
# BH-FDR pass over operator-analysis tests (kept inline with operator step)
# ---------------------------------------------------------------------------


def _bh_fdr_operator_families(
    coef_df: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Compute BH-FDR p_adj over the regression coefficient z-tests in this
    (source, adjusted_op_effects) family."""
    if coef_df.empty or "p_value" not in coef_df.columns:
        return coef_df
    adj_p, reject = extra_contrast.apply_bh_fdr(coef_df["p_value"].tolist(), alpha=alpha)
    out = coef_df.copy()
    out["p_value_adj_bh"] = adj_p
    out["reject_bh"] = reject
    return out


# ---------------------------------------------------------------------------
# Fair cross-model operator comparison
# ---------------------------------------------------------------------------


def _run_crossmodel_operator_comparison(
    df_greedy: pd.DataFrame,
    df_topk_flat: pd.DataFrame,
    runs: list[str],
    fig_dir: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Pooled model x has_op interaction (cluster-robust by formula_id) + AME
    differences + operator-stratified McNemar, on the shared common-target set.

    BH-FDR family convention: one family per (source, outcome) for the
    interaction p-values, and one per (source, outcome) for the stratified
    McNemar p-values.
    """
    reference_run = args.reference_run

    df_op_greedy = operator_analysis.build_target_operator_frame(df_greedy)
    df_op_topk = operator_analysis.build_target_operator_frame_topk(
        df_topk_flat, df_greedy,
    )

    # (tag, df_op, outcome_col, outcome_label)
    cases = [
        ("greedy_correct", df_op_greedy, "correct",
         "greedy correctness"),
        ("greedy_invalid", df_op_greedy, "invalid",
         "greedy invalidity (brittleness attribution)"),
        ("topk_any_correct", df_op_topk, "correct",
         "top-K any-correct"),
    ]

    for tag, df_op, outcome_col, outcome_label in cases:
        if outcome_col not in df_op.columns:
            continue
        out = operator_crossmodel.fit_pooled_interaction(
            df_op, runs=runs, reference_run=reference_run,
            outcome_col=outcome_col, alpha=args.alpha,
            n_sim=max(1000, args.bootstrap_n // 2), rng_seed=args.rng_seed,
        )
        interactions = out["interactions"]
        ame = out["ame"]

        if not interactions.empty:
            # BH-FDR over the model x predictor interaction family for this
            # (source, outcome) — only the has_OP interactions (operator tests).
            op_mask = interactions["predictor"].str.startswith("has_")
            op_inter = interactions[op_mask].copy()
            adj_p, reject = extra_contrast.apply_bh_fdr(
                op_inter["p_value"].tolist(), alpha=args.alpha,
            )
            op_inter["p_value_adj_bh"] = adj_p
            op_inter["reject_bh"] = reject
            # Keep covariate interactions unadjusted but in the CSV.
            cov_inter = interactions[~op_mask].copy()
            cov_inter["p_value_adj_bh"] = float("nan")
            cov_inter["reject_bh"] = False
            interactions_out = pd.concat([op_inter, cov_inter],
                                         ignore_index=True)
            interactions_out.insert(0, "source_outcome", tag)
            interactions_out.to_csv(
                stats_dir / f"crossmodel_interaction_{tag}.csv", index=False,
            )
            extra_plots.plot_crossmodel_interaction_forest(
                op_inter, runs=runs, reference_run=reference_run,
                alpha=args.alpha,
                stem=fig_dir / f"crossmodel_interaction_{tag}",
                outcome_label=outcome_label,
            )

        # All-pairs interaction contrasts (own BH family per source,outcome).
        inter_pw = out.get("interactions_pairwise", pd.DataFrame())
        if not inter_pw.empty:
            adj_p, reject = extra_contrast.apply_bh_fdr(
                inter_pw["p_value"].tolist(), alpha=args.alpha,
            )
            inter_pw = inter_pw.copy()
            inter_pw["p_value_adj_bh"] = adj_p
            inter_pw["reject_bh"] = reject
            inter_pw.insert(0, "source_outcome", tag)
            inter_pw.to_csv(
                stats_dir / f"crossmodel_interaction_pairwise_{tag}.csv",
                index=False,
            )
            extra_plots.plot_crossmodel_pairwise_heatmaps(
                inter_pw, value_col="coef", sig_col="p_value_adj_bh",
                runs=runs, alpha=args.alpha,
                title=f"Pairwise interaction Δ log-odds slope ({outcome_label})",
                stem=fig_dir / f"crossmodel_interaction_pairwise_{tag}",
            )

        if not ame.empty:
            ame_out = ame.copy()
            ame_out.insert(0, "source_outcome", tag)
            ame_out["n_obs"] = out["n_obs"]
            ame_out["n_targets"] = out["n_targets"]
            ame_out.to_csv(
                stats_dir / f"crossmodel_ame_{tag}.csv", index=False,
            )
            extra_plots.plot_crossmodel_ame_forest(
                ame, runs=runs, reference_run=reference_run,
                alpha=args.alpha,
                stem=fig_dir / f"crossmodel_ame_{tag}",
                outcome_label=outcome_label,
            )

        # All-pairs AME differences (CI-based; no p-value family).
        ame_pw = out.get("ame_pairwise", pd.DataFrame())
        if not ame_pw.empty:
            ame_pw = ame_pw.copy()
            ame_pw.insert(0, "source_outcome", tag)
            ame_pw.to_csv(
                stats_dir / f"crossmodel_ame_pairwise_{tag}.csv", index=False,
            )
            extra_plots.plot_crossmodel_pairwise_heatmaps(
                ame_pw, value_col="ame_diff", sig_col=None,
                runs=runs, alpha=args.alpha,
                title=f"Pairwise AME difference on P(outcome) ({outcome_label})",
                stem=fig_dir / f"crossmodel_ame_pairwise_{tag}",
                cmap="PuOr_r",
            )

        # Operator-stratified McNemar cross-check (assumption-light).
        strat = operator_crossmodel.operator_stratified_mcnemar(
            df_op, runs=runs, reference_run=reference_run,
            outcome_col=outcome_col,
        )
        if not strat.empty:
            adj_p, reject = extra_contrast.apply_bh_fdr(
                strat["mcnemar_p"].tolist(), alpha=args.alpha,
            )
            strat["mcnemar_p_adj_bh"] = adj_p
            strat["mcnemar_reject_bh"] = reject
            strat.insert(0, "source_outcome", tag)
            strat.to_csv(
                stats_dir / f"crossmodel_stratified_mcnemar_{tag}.csv",
                index=False,
            )
            extra_plots.plot_stratified_mcnemar(
                strat, runs=runs, reference_run=reference_run,
                alpha=args.alpha,
                stem=fig_dir / f"crossmodel_stratified_mcnemar_{tag}",
                outcome_label=outcome_label,
            )

        # All-pairs operator-stratified McNemar (own BH family).
        strat_pw = operator_crossmodel.operator_stratified_mcnemar_pairwise(
            df_op, runs=runs, outcome_col=outcome_col,
        )
        if not strat_pw.empty:
            adj_p, reject = extra_contrast.apply_bh_fdr(
                strat_pw["mcnemar_p"].tolist(), alpha=args.alpha,
            )
            strat_pw["mcnemar_p_adj_bh"] = adj_p
            strat_pw["mcnemar_reject_bh"] = reject
            strat_pw.insert(0, "source_outcome", tag)
            strat_pw.to_csv(
                stats_dir / f"crossmodel_stratified_mcnemar_pairwise_{tag}.csv",
                index=False,
            )
            extra_plots.plot_crossmodel_pairwise_heatmaps(
                strat_pw, value_col="mcnemar_effect",
                sig_col="mcnemar_p_adj_bh", runs=runs, alpha=args.alpha,
                title=f"Pairwise operator-stratified McNemar effect ({outcome_label})",
                stem=fig_dir / f"crossmodel_stratified_mcnemar_pairwise_{tag}",
            )


if __name__ == "__main__":
    main()
