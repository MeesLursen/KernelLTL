"""Generate a narrative summary.md from the analysis outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summary(
    *,
    output_dir: Path,
    runs: list[str],
    reference_run: str,
    descriptives: pd.DataFrame,
    pairwise: pd.DataFrame,
    pairwise_by_depth: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    metadata: dict,
) -> Path:
    output_dir = Path(output_dir)
    md_path = output_dir / "summary.md"

    lines: list[str] = []
    lines.append("# Validation Analysis — Summary")
    lines.append("")
    lines.append("## Run inventory")
    lines.append("")
    lines.append(f"- **Reference run**: `{reference_run}`")
    lines.append(f"- **Compared runs**: {', '.join(f'`{r}`' for r in runs)}")
    lines.append(f"- **Bootstrap resamples**: {metadata.get('bootstrap_n')}")
    lines.append(f"- **α**: {metadata.get('alpha')}")
    lines.append(f"- **MC correction**: {metadata.get('mc_correction')}")
    lines.append(f"- **RNG seed**: {metadata.get('rng_seed')}")
    lines.append("")

    lines.append("## How to read this report")
    lines.append("")
    lines.append(
        "Effect size + 95% bootstrap CI is the headline. P-values are reported as a "
        "backstop and BH-FDR adjusted across all (metric × variant) cells. With the "
        "validation set's large n, almost any non-zero population-level difference "
        "will register as 'statistically significant'; what matters in interpretation "
        "is the magnitude of the effect and the tightness of its CI. We deliberately "
        "do **not** declare fixed practical-significance thresholds — those are left "
        "to the thesis discussion. The `tests_agree` column reports whether Wilcoxon "
        "and the paired permutation test agree at α on continuous metrics; "
        "disagreements warrant a closer look at the diagnostics figure."
    )
    lines.append("")

    if not descriptives.empty:
        lines.append("## Descriptive statistics by run")
        lines.append("")
        lines.append(descriptives.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if not pairwise.empty:
        lines.append(f"## Pairwise vs `{reference_run}` (overall)")
        lines.append("")
        cols = [
            "metric", "metric_class", "variant", "n_pairs",
            "mean_ref", "mean_var", "effect_size", "effect_size_name",
            "ci_low", "ci_high", "wilcoxon_p", "perm_p", "mcnemar_p",
            "p_adj_bh", "tests_agree",
        ]
        cols = [c for c in cols if c in pairwise.columns]
        lines.append(pairwise[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if pairwise_by_depth is not None and not pairwise_by_depth.empty:
        lines.append(f"## Pairwise vs `{reference_run}` (by depth, headline metrics)")
        lines.append("")
        focus_metrics = [
            "semantic_distance", "semantic_distance_mean_topk",
            "semantic_equiv_rate", "semantic_equiv_rate_topk",
            "self_bleu_mean",
        ]
        sub = pairwise_by_depth[pairwise_by_depth["metric"].isin(focus_metrics)]
        if not sub.empty:
            cols = [
                "metric", "variant", "target_depth", "n_pairs",
                "mean_ref", "mean_var", "effect_size", "ci_low", "ci_high",
                "p_adj_bh",
            ]
            cols = [c for c in cols if c in sub.columns]
            lines.append(sub[cols].to_markdown(index=False, floatfmt=".4f"))
            lines.append("")

    if diagnostics is not None and not diagnostics.empty:
        lines.append("## Wilcoxon-assumption diagnostics")
        lines.append("")
        lines.append(
            "Skewness (of paired differences) and tie-rate per (metric, variant). "
            "Large |skewness| or high tie-rate is a flag to prefer the permutation "
            "p over the Wilcoxon p; review `diagnostics/wilcoxon_assumption_diagnostics.pdf`."
        )
        lines.append("")
        lines.append(diagnostics.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append("All figures live under `figures/` (PDF + PNG). Highlights:")
    lines.append("")
    lines.append("- `radar_overall.pdf` — 7-axis comparison on monotone-quality metrics.")
    lines.append("- `pareto_quality_diversity.pdf` — quality ↔ diversity trade-off.")
    lines.append("- `pareto_quality_creativity.pdf` — accuracy ↔ syntactic-creativity.")
    lines.append("- `pareto_quality_depth_degradation.pdf` — robustness across depths.")
    lines.append("- `kl_drift_vs_quality.pdf` — drift vs equivalence-rate scatter.")
    lines.append("- `bydepth_*.pdf` — every metric sliced by `target_depth`.")
    lines.append("- `paired_diff_*.pdf` — paired (variant − reference) per target.")
    lines.append("- `ecdf_*.pdf` — distributional comparisons.")
    lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path
