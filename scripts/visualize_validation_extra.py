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
    extra_contrast, extra_metrics, extra_plots, loaders, operator_analysis,
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
    )
    df_greedy = loaded["df_greedy"]
    df_topk_flat = loaded["df_topk_flat"]
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

    print("[extra] computing target-side operator analysis...", file=sys.stderr)
    _run_operator_analysis(df_greedy, df_topk_flat, runs, fig_dir, stats_dir, args)

    print("[extra] computing contrast studies (paired-diff, agreement, output similarity)...", file=sys.stderr)
    _run_contrast_studies(df_greedy, df_topk_flat, runs, fig_dir, stats_dir, args)

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
        # KL + per-op contributions
        kl_df = operator_analysis.compute_kl_per_run(df_op, runs=runs)
        kl_df.to_csv(stats_dir / f"op_kl_{src_name}.csv", index=False)
        extra_plots.plot_kl_per_run(kl_df, runs=runs, stem=fig_dir / f"op_kl_{src_name}")
        extra_plots.plot_kl_contribution_per_run(
            kl_df, runs=runs, stem=fig_dir / f"op_kl_contribution_{src_name}",
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
                variants=variants, reference_run=reference_run,
                title=f"Paired Δ vs {reference_run} — {ylabel} ({src_name})",
                ylabel=ylabel,
                stem=fig_dir / f"contrast_paired_diff_{metric_name}_{src_name}",
                rng_seed=args.rng_seed,
            )
            if not bydepth.empty:
                extra_plots.plot_paired_diff_bydepth(
                    bydepth, variants=variants, reference_run=reference_run,
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


if __name__ == "__main__":
    main()
