"""Figures for the extra (conditional / operator) validation analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

from scripts._validation_analysis.operator_analysis import OPERATORS, COVARIATES


CE_BASE_GREY = "#9e9e9e"


def _save(fig: plt.Figure, stem: Path, dpi: int = 200) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _run_palette(runs: list[str], grey_runs=()) -> dict[str, tuple]:
    base = sns.color_palette("colorblind", n_colors=max(len(runs), 1))
    pal = {}
    idx = 0
    for r in runs:
        if r in grey_runs:
            pal[r] = CE_BASE_GREY
        else:
            pal[r] = base[idx % len(base)]
            idx += 1
    return pal


# ---------------------------------------------------------------------------
# Conditional value / diff plots
# ---------------------------------------------------------------------------


def plot_conditional_bar(
    stats_df: pd.DataFrame,
    *,
    runs: list[str],
    title: str,
    ylabel: str,
    stem: Path,
    hline: float | None = 0.0,
) -> None:
    """Bar of (mean, CI) per run from a conditional stats frame (overall)."""
    sub = stats_df[stats_df["run"].isin(runs)].copy()
    sub["run"] = pd.Categorical(sub["run"], categories=runs, ordered=True)
    sub = sub.sort_values("run")
    if sub.empty:
        return
    pal = _run_palette(runs)
    colors = [pal[r] for r in sub["run"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = np.stack([
        sub["mean"] - sub["ci_low"],
        sub["ci_high"] - sub["mean"],
    ])
    ax.bar(sub["run"].astype(str), sub["mean"], yerr=yerr, color=colors, capsize=4)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    # annotate n
    for x, (_, row) in enumerate(sub.iterrows()):
        if "n" in row and not np.isnan(row.get("n", np.nan)):
            ax.annotate(f"n={int(row['n'])}", (x, row["mean"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    _save(fig, stem)


def plot_conditional_bydepth(
    stats_df: pd.DataFrame,
    *,
    runs: list[str],
    title: str,
    ylabel: str,
    stem: Path,
    hline: float | None = 0.0,
) -> None:
    """Per-depth line plot with shaded CI per run."""
    sub = stats_df[stats_df["run"].isin(runs)].copy()
    if sub.empty:
        return
    pal = _run_palette(runs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        rsub = sub[sub["run"] == r].sort_values("target_depth")
        if rsub.empty:
            continue
        ax.plot(rsub["target_depth"], rsub["mean"], "-o", label=r, color=pal[r])
        ax.fill_between(rsub["target_depth"], rsub["ci_low"], rsub["ci_high"],
                        alpha=0.2, color=pal[r])
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("target_depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# pass@k' curve
# ---------------------------------------------------------------------------


def plot_pass_at_k_curve(
    pak_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
) -> None:
    sub = pak_df[pak_df["run"].isin(runs)].copy()
    if sub.empty:
        return
    pal = _run_palette(runs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        rsub = sub[sub["run"] == r].sort_values("k_prime")
        if rsub.empty:
            continue
        ax.plot(rsub["k_prime"], rsub["mean"], "-o", label=r, color=pal[r])
        ax.fill_between(rsub["k_prime"], rsub["ci_low"], rsub["ci_high"],
                        alpha=0.2, color=pal[r])
    ax.set_xlabel("k' (number of samples drawn)")
    ax.set_ylabel("pass@k' (≥1 correct in k' draws)")
    ax.set_title("Top-K pass@k' curve")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Distinct-correct bar
# ---------------------------------------------------------------------------


def plot_distinct_correct_bar(
    stats_df: pd.DataFrame,
    *,
    runs: list[str],
    title: str,
    ylabel: str,
    stem: Path,
) -> None:
    plot_conditional_bar(stats_df, runs=runs, title=title, ylabel=ylabel,
                         stem=stem, hline=None)


# ---------------------------------------------------------------------------
# Operator analysis plots
# ---------------------------------------------------------------------------


def plot_kl_per_run(
    kl_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
) -> None:
    """Bar of KL(P_op|correct ‖ P_op|wrong) per run."""
    sub = kl_df[kl_df["run"].isin(runs)].copy()
    sub["run"] = pd.Categorical(sub["run"], categories=runs, ordered=True)
    sub = sub.sort_values("run")
    if sub.empty:
        return
    pal = _run_palette(runs)
    colors = [pal[r] for r in sub["run"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sub["run"].astype(str), sub["kl_correct_to_wrong"], color=colors)
    ax.set_ylabel("KL(P_op | correct ‖ P_op | wrong)")
    ax.set_title("Operator-distribution divergence between correct & wrong targets")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, stem)


def plot_kl_contribution_per_run(
    kl_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
) -> None:
    """Per-operator contribution to the per-run KL: P(op|correct) · log(P(op|correct)/P(op|wrong))."""
    sub = kl_df[kl_df["run"].isin(runs)].copy()
    if sub.empty:
        return
    long_rows = []
    for _, row in sub.iterrows():
        for op in OPERATORS:
            long_rows.append({"run": row["run"], "op": op,
                              "contrib": row.get(f"contrib_{op}", float("nan"))})
    long_df = pd.DataFrame(long_rows)
    long_df["run"] = pd.Categorical(long_df["run"], categories=runs, ordered=True)
    long_df["op"] = pd.Categorical(long_df["op"], categories=OPERATORS, ordered=True)
    long_df = long_df.sort_values(["run", "op"])
    pal = _run_palette(runs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(len(runs), 1)
    x_pos = np.arange(len(OPERATORS))
    for i, r in enumerate(runs):
        rsub = long_df[long_df["run"] == r]
        if rsub.empty:
            continue
        ax.bar(x_pos + (i - len(runs) / 2) * width + width / 2,
               rsub["contrib"], width=width, color=pal[r], label=r)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(OPERATORS)
    ax.set_ylabel("P(op|correct) · log( P(op|correct) / P(op|wrong) )")
    ax.set_title("Per-operator contribution to KL(correct ‖ wrong)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


def plot_op_decomposition(
    decomp_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
) -> None:
    """Side-by-side bars of P(op | correct) vs P(op | wrong) per operator, base
    rate annotated as a horizontal tick.

    Produces one panel per run in a single figure (facetted)."""
    sub = decomp_df[decomp_df["run"].isin(runs)].copy()
    if sub.empty:
        return
    n_runs = len(runs)
    ncols = min(n_runs, 3)
    nrows = (n_runs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows),
                             squeeze=False, sharey=True)
    x_pos = np.arange(len(OPERATORS))
    width = 0.38

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols, idx % ncols]
        rsub = sub[sub["run"] == r].set_index("op").reindex(OPERATORS)
        if rsub.empty:
            ax.set_title(f"{r} (no data)")
            continue
        # Error bars: (mean - low, high - mean), NaN -> 0 so errorbar still draws.
        if "p_op_given_correct_ci_low" in rsub.columns:
            c_err = np.stack([
                (rsub["p_op_given_correct"] - rsub["p_op_given_correct_ci_low"]).to_numpy(),
                (rsub["p_op_given_correct_ci_high"] - rsub["p_op_given_correct"]).to_numpy(),
            ])
            w_err = np.stack([
                (rsub["p_op_given_wrong"] - rsub["p_op_given_wrong_ci_low"]).to_numpy(),
                (rsub["p_op_given_wrong_ci_high"] - rsub["p_op_given_wrong"]).to_numpy(),
            ])
            c_err = np.where(np.isnan(c_err), 0.0, c_err)
            w_err = np.where(np.isnan(w_err), 0.0, w_err)
        else:
            c_err = None
            w_err = None
        ax.bar(x_pos - width / 2, rsub["p_op_given_correct"],
               width=width, color=sns.color_palette()[2], label="correct",
               yerr=c_err, capsize=2, ecolor="grey", error_kw={"linewidth": 0.8})
        ax.bar(x_pos + width / 2, rsub["p_op_given_wrong"],
               width=width, color=sns.color_palette()[3], label="wrong",
               yerr=w_err, capsize=2, ecolor="grey", error_kw={"linewidth": 0.8})
        # Base rate tick
        for xi, op in enumerate(OPERATORS):
            br = rsub.loc[op, "p_op_base"] if op in rsub.index else float("nan")
            if not np.isnan(br):
                ax.hlines(br, xi - width, xi + width, colors="black",
                          linestyles="--", linewidths=1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(OPERATORS, fontsize=8)
        ax.set_title(r, fontsize=10)
        ax.set_ylim(0, 1)
        if idx % ncols == 0:
            ax.set_ylabel("P(op present in target)")
    # Hide unused axes
    for j in range(n_runs, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=sns.color_palette()[2], label="correct"),
        plt.Rectangle((0, 0), 1, 1, color=sns.color_palette()[3], label="wrong"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="base rate"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9)
    fig.suptitle("Operator presence in targets, by model correctness", y=1.05)
    fig.tight_layout()
    _save(fig, stem)


def plot_log_odds_forest(
    lor_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
    title: str = "Log-odds ratio: P(correct | has_op) vs P(correct | not has_op)",
) -> None:
    """One forest plot per run, faceted. y = operators, x = log-odds-ratio with CI."""
    sub = lor_df[lor_df["run"].isin(runs)].copy()
    if sub.empty:
        return
    n_runs = len(runs)
    ncols = min(n_runs, 3)
    nrows = (n_runs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True)
    y_pos = np.arange(len(OPERATORS))
    pal = _run_palette(runs)

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols, idx % ncols]
        rsub = sub[sub["run"] == r].set_index("op").reindex(OPERATORS)
        if rsub.empty:
            ax.set_title(f"{r} (no data)")
            continue
        means = rsub["log_odds_ratio"].to_numpy()
        lo = rsub["ci_low"].to_numpy()
        hi = rsub["ci_high"].to_numpy()
        xerr = np.stack([means - lo, hi - means])
        ax.errorbar(means, y_pos, xerr=xerr, fmt="o",
                    color=pal[r], capsize=3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(OPERATORS, fontsize=8)
        ax.set_title(r, fontsize=10)
        ax.set_xlabel("log-odds ratio")
        ax.invert_yaxis()
    for j in range(n_runs, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, stem)


def plot_logistic_coefficient_forest(
    coef_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
    include_covariates: bool = True,
    title: str = "Logistic regression coefficients (target_depth + target_length_tokens covariates)",
) -> None:
    """Forest plot of logistic-regression coefficients per (run, operator).
    Coefficients are on the log-odds scale, controlling for ``target_depth``,
    ``target_length_tokens`` and the other operator indicators simultaneously.
    """
    sub = coef_df[coef_df["run"].isin(runs)].copy()
    if sub.empty:
        return
    # Operator rows by default; covariate rows optionally appended.
    preds = [op for op in OPERATORS]
    if include_covariates:
        preds = preds + list(COVARIATES)
    n_runs = len(runs)
    ncols = min(n_runs, 3)
    nrows = (n_runs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows),
                             squeeze=False, sharex=True)
    y_pos = np.arange(len(preds))
    pal = _run_palette(runs)

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols, idx % ncols]
        rsub = sub[sub["run"] == r].set_index("op").reindex(preds)
        if rsub.empty:
            ax.set_title(f"{r} (no data)")
            continue
        means = rsub["coef"].to_numpy()
        lo = rsub["ci_low"].to_numpy()
        hi = rsub["ci_high"].to_numpy()
        xerr = np.stack([means - lo, hi - means])
        # Replace NaNs in xerr (from non-converged regressions) with 0 so errorbar still plots
        xerr = np.where(np.isnan(xerr), 0.0, xerr)
        ax.errorbar(means, y_pos, xerr=xerr, fmt="o",
                    color=pal[r], capsize=3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(preds, fontsize=8)
        ax.set_title(r, fontsize=10)
        ax.set_xlabel("coefficient (log-odds)")
        ax.invert_yaxis()
    for j in range(n_runs, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Paired-diff small multiples for conditional metrics
# ---------------------------------------------------------------------------


def plot_paired_diff_conditional(
    per_target_diffs: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    variants: list[str],
    reference_run: str,
    title: str,
    ylabel: str,
    stem: Path,
    rng_seed: int = 0,
    scatter_max: int = 2000,
) -> None:
    """One sub-axis per variant. Jittered scatter of per-target paired
    differences + mean with bootstrap CI errorbar from ``summary_df``."""
    if per_target_diffs.empty or summary_df.empty:
        return
    rng = np.random.default_rng(rng_seed)
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, variant in zip(axes, variants):
        d = per_target_diffs[per_target_diffs["variant"] == variant]["diff"].astype(float).to_numpy()
        s = summary_df[summary_df["variant"] == variant]
        if len(d) == 0 or s.empty:
            ax.set_title(f"{variant}\n(n=0)", fontsize=10)
            ax.set_xticks([])
            continue
        n_show = min(scatter_max, len(d))
        idx = rng.choice(len(d), size=n_show, replace=False) if n_show < len(d) else np.arange(len(d))
        ax.scatter(rng.uniform(-0.15, 0.15, n_show), d[idx], alpha=0.15, s=6, color="grey")
        m = float(s["mean_diff"].iloc[0])
        lo = float(s["ci_low"].iloc[0])
        hi = float(s["ci_high"].iloc[0])
        ax.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt="o",
                    color="black", capsize=4, markersize=6)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        n_pairs = int(s["n_pairs"].iloc[0])
        ax.set_title(f"{variant}\n(n={n_pairs})", fontsize=10)
    axes[0].set_ylabel(f"{ylabel}  (variant − {reference_run})")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, stem)


def plot_paired_diff_bydepth(
    bydepth_df: pd.DataFrame,
    *,
    variants: list[str],
    reference_run: str,
    title: str,
    ylabel: str,
    stem: Path,
) -> None:
    """Stratified paired diffs by target_depth — line + shaded CI per variant."""
    if bydepth_df.empty:
        return
    pal = _run_palette(variants)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        sub = bydepth_df[bydepth_df["variant"] == v].sort_values("target_depth")
        if sub.empty:
            continue
        ax.plot(sub["target_depth"], sub["mean_diff"], "-o", label=v, color=pal[v])
        ax.fill_between(sub["target_depth"], sub["ci_low"], sub["ci_high"],
                        alpha=0.2, color=pal[v])
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("target_depth")
    ax.set_ylabel(f"{ylabel}  (variant − {reference_run})")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Agreement matrices: Cohen's κ and McNemar
# ---------------------------------------------------------------------------


def plot_agreement_kappa_heatmap(
    agreement_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
    title: str = "Pairwise Cohen's κ (correctness agreement)",
) -> None:
    if agreement_df.empty:
        return
    M = agreement_df.pivot(index="run_a", columns="run_b", values="kappa").reindex(
        index=runs, columns=runs
    )
    fig, ax = plt.subplots(figsize=(0.9 * len(runs) + 4, 0.9 * len(runs) + 3))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, vmin=-0.2, vmax=1.0,
                square=True, cbar_kws={"label": "Cohen's κ"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, stem)


def plot_agreement_mcnemar_heatmap(
    agreement_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
    significance_col: str = "mcnemar_p_adj",
    alpha: float = 0.05,
    title: str = "Pairwise McNemar effect: signed advantage of row over column",
) -> None:
    if agreement_df.empty:
        return
    M = agreement_df.pivot(index="run_a", columns="run_b", values="mcnemar_effect").reindex(
        index=runs, columns=runs
    )
    if significance_col in agreement_df.columns:
        S = agreement_df.pivot(index="run_a", columns="run_b", values=significance_col).reindex(
            index=runs, columns=runs
        )
    else:
        S = None
    annot = M.copy().astype(object)
    for i in runs:
        for j in runs:
            if i == j or pd.isna(M.loc[i, j]):
                annot.loc[i, j] = ""
                continue
            cell = f"{M.loc[i, j]:+.2f}"
            if S is not None and not pd.isna(S.loc[i, j]) and S.loc[i, j] < alpha:
                cell += "*"
            annot.loc[i, j] = cell
    fig, ax = plt.subplots(figsize=(0.9 * len(runs) + 4, 0.9 * len(runs) + 3))
    sns.heatmap(M.astype(float), annot=annot.values, fmt="", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "(row > col) − (col > row) / n_pairs"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title + ("  (* = BH-FDR-adjusted p < α)" if S is not None else ""))
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Output-similarity heatmap
# ---------------------------------------------------------------------------


def plot_output_similarity_heatmap(
    sim_df: pd.DataFrame,
    *,
    runs: list[str],
    metric_col: str,
    title: str,
    stem: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    if sim_df.empty:
        return
    M = sim_df.pivot(index="run_a", columns="run_b", values=metric_col).reindex(
        index=runs, columns=runs
    )
    fig, ax = plt.subplots(figsize=(0.9 * len(runs) + 4, 0.9 * len(runs) + 3))
    sns.heatmap(M, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax,
                square=True, cbar_kws={"label": metric_col}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, stem)


def plot_log_odds_vs_regression_overlay(
    lor_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    *,
    runs: list[str],
    stem: Path,
) -> None:
    """Overlay marginal log-odds (open marker) and adjusted regression coef
    (filled marker) connected by a thin line. One panel per run."""
    n_runs = len(runs)
    ncols = min(n_runs, 3)
    nrows = (n_runs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows),
                             squeeze=False, sharex=True)
    y_pos = np.arange(len(OPERATORS))
    pal = _run_palette(runs)

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols, idx % ncols]
        marg = lor_df[lor_df["run"] == r].set_index("op").reindex(OPERATORS)
        adj = coef_df[coef_df["run"] == r].set_index("op").reindex(OPERATORS)
        if marg.empty and adj.empty:
            ax.set_title(f"{r} (no data)")
            continue
        m_means = marg["log_odds_ratio"].to_numpy()
        a_means = adj["coef"].to_numpy()
        # connectors
        for i in range(len(OPERATORS)):
            if not (np.isnan(m_means[i]) or np.isnan(a_means[i])):
                ax.plot([m_means[i], a_means[i]], [y_pos[i], y_pos[i]],
                        color=pal[r], alpha=0.4, linewidth=1)
        ax.scatter(m_means, y_pos, facecolors="none", edgecolors=pal[r],
                   label="marginal", s=55)
        ax.scatter(a_means, y_pos, color=pal[r], label="adjusted", s=55)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(OPERATORS, fontsize=8)
        ax.set_title(r, fontsize=10)
        ax.set_xlabel("log-odds")
        ax.invert_yaxis()
        ax.legend(loc="best", fontsize=7)
    for j in range(n_runs, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Marginal log-odds (open) vs adjusted regression coef (filled)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, stem)
