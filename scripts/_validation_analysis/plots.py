"""Cross-model figures for validation analysis.

Most figures are produced by a small number of generic helpers
(``plot_metric_box``, ``plot_metric_rate_bar``, ``plot_ecdf``,
``plot_bydepth_line``, ``plot_paired_diff_multiples``); a few have their
own functions (``plot_radar``, ``plot_pareto``,
``plot_kl_drift_vs_quality``).

Style: seaborn ``whitegrid`` with the ``colorblind`` palette. ``ce_base``
is always grey on KL-related panels via ``_run_palette`` overriding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

CE_BASE = "ce_base"
CE_BASE_GREY = "#9e9e9e"


def _save(fig: plt.Figure, stem: Path, dpi: int = 200) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _run_palette(runs: list[str], grey_runs: Iterable[str] = ()) -> dict[str, tuple]:
    base = sns.color_palette("colorblind", n_colors=max(len(runs), 1))
    pal = {}
    color_idx = 0
    for r in runs:
        if r in grey_runs:
            pal[r] = CE_BASE_GREY
        else:
            pal[r] = base[color_idx % len(base)]
            color_idx += 1
    return pal


# ---------------------------------------------------------------------------
# Generic per-metric figures
# ---------------------------------------------------------------------------


def plot_metric_box(
    df: pd.DataFrame,
    metric: str,
    *,
    runs: list[str],
    title: str,
    ylabel: str,
    stem: Path,
    grey_runs: Iterable[str] = (),
) -> None:
    sub = df[df["run"].isin(runs)].copy()
    sub = sub[~sub[metric].isna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pal = _run_palette(runs, grey_runs=grey_runs)
    sns.boxplot(data=sub, x="run", y=metric, order=runs, hue="run", palette=pal, legend=False, ax=ax, fliersize=2)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, stem)


def plot_metric_rate_bar(
    df: pd.DataFrame,
    metric: str,
    *,
    runs: list[str],
    title: str,
    ylabel: str,
    stem: Path,
    n_bootstrap: int = 2000,
    rng_seed: int = 0,
    grey_runs: Iterable[str] = (),
) -> None:
    """Mean of a rate / binary column per run with bootstrap 95% CI."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        vals = df.loc[df["run"] == r, metric].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            means[i] = vals[rng.integers(0, len(vals), len(vals))].mean()
        rows.append({
            "run": r,
            "mean": float(vals.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)),
        })
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pal = _run_palette(runs, grey_runs=grey_runs)
    colors = [pal[r] for r in plot_df["run"]]
    yerr = np.stack([
        plot_df["mean"] - plot_df["ci_low"],
        plot_df["ci_high"] - plot_df["mean"],
    ])
    ax.bar(plot_df["run"], plot_df["mean"], yerr=yerr, color=colors, capsize=4)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, stem)


def plot_ecdf(
    df: pd.DataFrame,
    metric: str,
    *,
    runs: list[str],
    title: str,
    xlabel: str,
    stem: Path,
    exclude_runs: Iterable[str] = (),
    subsample: int | None = None,
    rng_seed: int = 0,
) -> None:
    plot_runs = [r for r in runs if r not in exclude_runs]
    if not plot_runs:
        return
    rng = np.random.default_rng(rng_seed)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pal = _run_palette(plot_runs)
    for r in plot_runs:
        vals = df.loc[df["run"] == r, metric].astype(float).dropna().to_numpy()
        if subsample is not None and len(vals) > subsample:
            vals = rng.choice(vals, size=subsample, replace=False)
        if len(vals) == 0:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=r, color=pal[r])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


def plot_bydepth_line(
    df: pd.DataFrame,
    metric: str,
    *,
    runs: list[str],
    title: str,
    ylabel: str,
    stem: Path,
    exclude_runs: Iterable[str] = (),
    n_bootstrap: int = 1000,
    rng_seed: int = 0,
) -> None:
    plot_runs = [r for r in runs if r not in exclude_runs]
    if not plot_runs:
        return
    sub = df[df["run"].isin(plot_runs)].copy()
    if sub[metric].isna().all():
        return
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in plot_runs:
        for d, depth_sub in sub[sub["run"] == r].groupby("target_depth"):
            vals = depth_sub[metric].astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            means = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                means[i] = vals[rng.integers(0, len(vals), len(vals))].mean()
            rows.append({
                "run": r,
                "target_depth": int(d),
                "mean": float(vals.mean()),
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
            })
    if not rows:
        return
    plot_df = pd.DataFrame(rows).sort_values(["run", "target_depth"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pal = _run_palette(plot_runs)
    for r in plot_runs:
        rsub = plot_df[plot_df["run"] == r]
        ax.plot(rsub["target_depth"], rsub["mean"], "-o", label=r, color=pal[r])
        ax.fill_between(rsub["target_depth"], rsub["ci_low"], rsub["ci_high"], alpha=0.2, color=pal[r])
    ax.set_xlabel("target_depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


def plot_paired_diff_multiples(
    df: pd.DataFrame,
    metric: str,
    *,
    reference_run: str,
    variants: list[str],
    title: str,
    ylabel: str,
    stem: Path,
    n_bootstrap: int = 5000,
    rng_seed: int = 0,
    scatter_max: int = 2000,
) -> None:
    """1×N small-multiples grid, paired (variant − reference) per target."""
    rng = np.random.default_rng(rng_seed)
    fig, axes = plt.subplots(1, len(variants), figsize=(3.5 * len(variants), 4.5), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        ref_df = df[df["run"] == reference_run][["formula_id", metric]].rename(columns={metric: "ref"})
        var_df = df[df["run"] == variant][["formula_id", metric]].rename(columns={metric: "var"})
        merged = ref_df.merge(var_df, on="formula_id", how="inner")
        merged = merged.dropna()
        diffs = (merged["var"].astype(float) - merged["ref"].astype(float)).to_numpy()
        if len(diffs) == 0:
            ax.set_title(f"{variant} (n=0)")
            continue
        n_show = min(scatter_max, len(diffs))
        idx_show = rng.choice(len(diffs), size=n_show, replace=False) if n_show < len(diffs) else np.arange(len(diffs))
        ax.scatter(np.zeros(n_show) + rng.uniform(-0.15, 0.15, n_show), diffs[idx_show], alpha=0.15, s=6, color="grey")
        means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            means[i] = diffs[rng.integers(0, len(diffs), len(diffs))].mean()
        m = float(diffs.mean())
        lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
        ax.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt="o", color="black", capsize=4, markersize=6)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.set_title(f"{variant}\n(n={len(diffs)})", fontsize=10)
    axes[0].set_ylabel(f"({metric} variant − {reference_run})")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Holistic plots
# ---------------------------------------------------------------------------


def plot_radar(
    summary: pd.DataFrame,
    *,
    runs: list[str],
    axes_metrics: list[tuple[str, str]],  # (column_name_in_summary, axis_label)
    title: str,
    stem: Path,
    normalize: bool = False,
) -> None:
    """Radar with one polygon per run. ``summary`` has columns: run, <metrics...>.

    By default plots raw values on a fixed [0, 1] radial scale — assumes all
    axes are bounded rates. Set ``normalize=True`` for min-max normalization
    across runs (useful when axes have heterogeneous scales).
    """
    runs = [r for r in runs if r in summary["run"].tolist()]
    if not runs or not axes_metrics:
        return
    cols = [c for c, _ in axes_metrics]
    labels = [lab for _, lab in axes_metrics]

    sub = summary.set_index("run").loc[runs, cols].astype(float)
    if normalize:
        plot_vals = sub.copy()
        for c in cols:
            v = plot_vals[c]
            if v.max() == v.min():
                plot_vals[c] = 0.5
            else:
                plot_vals[c] = (v - v.min()) / (v.max() - v.min())
        subtitle = "  (min-max normalised across runs per axis)"
    else:
        plot_vals = sub
        subtitle = ""

    n = len(cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    pal = _run_palette(runs)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for r in runs:
        vals = plot_vals.loc[r].tolist()
        vals_closed = vals + vals[:1]
        ax.plot(angles_closed, vals_closed, label=r, color=pal[r], linewidth=2)
        ax.fill(angles_closed, vals_closed, color=pal[r], alpha=0.10)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    ax.set_title(title + subtitle, pad=20)
    _save(fig, stem)


def plot_pareto(
    summary: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    runs: list[str],
    x_label: str,
    y_label: str,
    title: str,
    stem: Path,
    higher_is_better: tuple[bool, bool] = (True, True),
) -> None:
    sub = summary.set_index("run").loc[[r for r in runs if r in summary["run"].tolist()]]
    sub = sub.dropna(subset=[x_col, y_col])
    if sub.empty:
        return
    pts = sub[[x_col, y_col]].astype(float).copy()

    def is_pareto(idx) -> bool:
        x_i = pts.loc[idx, x_col]
        y_i = pts.loc[idx, y_col]
        for j in pts.index:
            if j == idx:
                continue
            xj = pts.loc[j, x_col]
            yj = pts.loc[j, y_col]
            x_better = (xj > x_i) if higher_is_better[0] else (xj < x_i)
            y_better = (yj > y_i) if higher_is_better[1] else (yj < y_i)
            x_eq = xj == x_i
            y_eq = yj == y_i
            x_dom = x_better or x_eq
            y_dom = y_better or y_eq
            if x_dom and y_dom and (x_better or y_better):
                return False
        return True

    pareto_mask = {idx: is_pareto(idx) for idx in pts.index}
    pal = _run_palette(list(pts.index))
    fig, ax = plt.subplots(figsize=(7, 6))
    for run, row in pts.iterrows():
        ax.scatter(row[x_col], row[y_col], color=pal[run],
                   s=200 if pareto_mask[run] else 100,
                   edgecolor="black" if pareto_mask[run] else "none",
                   linewidth=2 if pareto_mask[run] else 0,
                   label=run + (" *" if pareto_mask[run] else ""))
        ax.annotate(run, (row[x_col], row[y_col]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title + "\n(* = Pareto-optimal)")
    fig.tight_layout()
    _save(fig, stem)


def plot_kl_drift_vs_quality(
    summary: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
) -> None:
    sub = summary.set_index("run").loc[[r for r in runs if r in summary["run"].tolist()]]
    sub = sub.dropna(subset=["kl_from_base_seq_mean", "semantic_equiv_rate"])
    if sub.empty:
        return
    pal = _run_palette(list(sub.index))
    fig, ax = plt.subplots(figsize=(7, 5))
    for run, row in sub.iterrows():
        ax.scatter(row["kl_from_base_seq_mean"], row["semantic_equiv_rate"], s=120, color=pal[run])
        ax.annotate(run, (row["kl_from_base_seq_mean"], row["semantic_equiv_rate"]),
                    fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("KL from base (greedy, seq-mean)")
    ax.set_ylabel("Semantic equivalence rate (greedy)")
    ax.set_title("KL drift vs greedy quality")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def plot_paired_diff_diagnostics(
    df: pd.DataFrame,
    metric_specs: list,  # list of (column_name, source_df) tuples
    *,
    reference_run: str,
    variants: list[str],
    sources: dict[str, pd.DataFrame],
    stem: Path,
) -> None:
    rows = []
    for spec in metric_specs:
        df_src = sources[spec.source]
        for variant in variants:
            ref_df = df_src[df_src["run"] == reference_run][["formula_id", spec.column]].rename(columns={spec.column: "ref"})
            var_df = df_src[df_src["run"] == variant][["formula_id", spec.column]].rename(columns={spec.column: "var"})
            merged = ref_df.merge(var_df, on="formula_id", how="inner").dropna()
            if len(merged) < 2:
                continue
            diffs = (merged["var"].astype(float) - merged["ref"].astype(float)).to_numpy()
            rows.append({"metric": spec.name, "variant": variant, "diffs": diffs})

    if not rows:
        return
    n = len(rows)
    cols = 4
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for ax_row, ax_col in [(i // cols, i % cols) for i in range(nrows * cols)]:
        axes[ax_row, ax_col].axis("off")
    for i, row in enumerate(rows):
        ax = axes[i // cols, i % cols]
        ax.axis("on")
        diffs = row["diffs"]
        mean = float(diffs.mean())
        std = float(diffs.std() or 1e-12)
        skew = float(np.mean(((diffs - mean) / std) ** 3))
        tie = float(np.mean(diffs == 0))
        ax.hist(diffs, bins=40, color=sns.color_palette()[0])
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(
            f"{row['metric']} | {row['variant']}\nskew={skew:.2f}, tie={tie:.2f}, n={len(diffs)}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
    fig.suptitle("Wilcoxon-assumption diagnostics: paired-difference distributions")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, stem)
