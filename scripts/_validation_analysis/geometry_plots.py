"""Figures for the embedding-geometry analysis (mirrors extra_plots.py style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, stem: Path, dpi: int = 200) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _palette(runs: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {r: cmap(i % 10) for i, r in enumerate(runs)}


def plot_logistic_forest(coef_df: pd.DataFrame, *, runs: list[str], stem: Path, dpi: int = 200) -> None:
    """Per-model logistic coefficients (log-odds, +/- 95% CI), faceted by run."""
    if coef_df.empty:
        return
    preds = list(dict.fromkeys(coef_df["predictor"]))
    runs = [r for r in runs if r in set(coef_df["run"])]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 2.6), sharex=True, squeeze=False)
    pal = _palette(runs)
    for ax, r in zip(axes[0], runs):
        rdf = coef_df[coef_df["run"] == r].set_index("predictor").reindex(preds)
        y = np.arange(len(preds))
        ax.errorbar(rdf["coef"], y,
                    xerr=[rdf["coef"] - rdf["ci_low"], rdf["ci_high"] - rdf["coef"]],
                    fmt="o", color=pal[r], capsize=2)
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(preds)
        ax.set_title(r, fontsize=8); ax.set_xlabel("coef (log-odds, per +1 SD)")
    axes[0][0].invert_yaxis()
    fig.suptitle("Geometry logistic: correct ~ std + alignment + depth", fontsize=9)
    _save(fig, stem, dpi)


def plot_2d_heatmap(grid_df: pd.DataFrame, *, run: str, stem: Path, dpi: int = 200) -> None:
    """std x alignment -> mean correctness for one run (orthogonality cell = high std, low alignment)."""
    sub = grid_df[grid_df["run"] == run]
    if sub.empty:
        return
    nx = sub["ix"].max() + 1
    ny = sub["iy"].max() + 1
    M = np.full((ny, nx), np.nan)
    for _, row in sub.iterrows():
        M[int(row["iy"]), int(row["ix"])] = row["mean_correct"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("std (informativeness) — quantile bin")
    ax.set_ylabel("alignment (anchor coverage) — quantile bin")
    ax.set_title(f"Mean correctness — {run}", fontsize=9)
    fig.colorbar(im, ax=ax, label="P(correct)")
    _save(fig, stem, dpi)


def plot_marginal(binned_df: pd.DataFrame, *, runs: list[str], feature: str, stem: Path, dpi: int = 200) -> None:
    sub = binned_df[binned_df["feature"] == feature]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    pal = _palette(runs)
    for r in runs:
        rdf = sub[sub["run"] == r].sort_values("x_mid")
        if rdf.empty:
            continue
        ax.plot(rdf["x_mid"], rdf["mean"], "-o", ms=3, color=pal[r], label=r)
        ax.fill_between(rdf["x_mid"], rdf["ci_low"], rdf["ci_high"], color=pal[r], alpha=0.15)
    ax.set_xlabel(feature); ax.set_ylabel("P(correct)")
    ax.set_title(f"Correctness vs {feature}", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, stem, dpi)


def plot_crossmodel_interaction(interactions: pd.DataFrame, *, reference_run: str,
                                stem: Path, dpi: int = 200) -> None:
    """Variant - reference slope difference per predictor (filled = BH-FDR sig)."""
    if interactions.empty:
        return
    preds = list(dict.fromkeys(interactions["predictor"]))
    fig, axes = plt.subplots(1, len(preds), figsize=(3.2 * len(preds), 2.6), sharex=True, squeeze=False)
    for ax, p in zip(axes[0], preds):
        sub = interactions[interactions["predictor"] == p]
        y = np.arange(len(sub))
        sig = sub.get("reject_bh", pd.Series([False] * len(sub))).to_numpy()
        ax.errorbar(sub["coef"], y, xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                    fmt="none", ecolor="0.4", capsize=2)
        ax.scatter(sub["coef"], y, c=["C1" if s else "white" for s in sig],
                   edgecolors="C1", zorder=3)
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(sub["variant"], fontsize=7)
        ax.set_title(p, fontsize=8); ax.set_xlabel(f"Δ slope vs {reference_run}")
    axes[0][0].invert_yaxis()
    fig.suptitle("Cross-model interaction: does geometry-reliance differ from ce_base?", fontsize=9)
    _save(fig, stem, dpi)


def plot_basrate_hist(features: pd.DataFrame, *, stem: Path, dpi: int = 200) -> None:
    """Sanity / trace-design check: base-rate and alignment distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    axes[0].hist(features["p"], bins=50, color="C0"); axes[0].set_title("base rate p"); axes[0].set_xlabel("p")
    axes[1].hist(features["std"], bins=50, color="C2"); axes[1].set_title("std (informativeness)"); axes[1].set_xlabel("std")
    axes[2].hist(features["alignment"], bins=50, color="C3"); axes[2].set_title("alignment (anchor coverage)"); axes[2].set_xlabel("alignment")
    _save(fig, stem, dpi)
