"""Figures for the embedding-geometry analysis (norm / variance / orthogonality)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, stem: Path, dpi: int = 200) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _palette(runs):
    cmap = plt.get_cmap("tab10")
    return {r: cmap(i % 10) for i, r in enumerate(runs)}


def plot_marginal(binned: pd.DataFrame, *, runs, feature: str, outcome: str, stem, dpi=200):
    sub = binned[(binned.feature == feature) & (binned.outcome == outcome)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); pal = _palette(runs)
    for r in runs:
        rdf = sub[sub.run == r].sort_values("x_mid")
        if rdf.empty:
            continue
        ax.plot(rdf.x_mid, rdf["mean"], "-o", ms=3, color=pal[r], label=r)
        ax.fill_between(rdf.x_mid, rdf.ci_low, rdf.ci_high, color=pal[r], alpha=0.15)
    ax.set_xlabel(feature); ax.set_ylabel(f"mean {outcome}")
    ax.set_title(f"{outcome} vs {feature}", fontsize=9); ax.legend(fontsize=7)
    _save(fig, stem, dpi)


def plot_stratified(strat: pd.DataFrame, *, runs, stem, dpi=200):
    """Norm slope (+/-95% CI) within variance strata, per model — the Q2 primary."""
    if strat.empty:
        return
    order = ["low", "mid", "high"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4)); pal = _palette(runs)
    xpos = {s: i for i, s in enumerate(order)}
    for r in runs:
        rdf = strat[strat.run == r]
        rdf = rdf[rdf.stratum.isin(order)].assign(_x=rdf.stratum.map(xpos)).sort_values("_x")
        if rdf.empty:
            continue
        ax.errorbar(rdf._x, rdf.norm_coef, yerr=[rdf.norm_coef - rdf.ci_low, rdf.ci_high - rdf.norm_coef],
                    fmt="-o", color=pal[r], capsize=2, label=r)
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_xticks(list(xpos.values())); ax.set_xticklabels(order)
    ax.set_xlabel("variance stratum"); ax.set_ylabel("norm slope (log-odds / +1 SD)")
    ax.set_title("Q2: does norm predict correctness within variance strata?", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, stem, dpi)


def plot_residual_forest(resid: pd.DataFrame, *, runs, stem, dpi=200):
    """Q2 summary: variance (informativeness) & orthogonality coefs per model."""
    if resid.empty:
        return
    preds = ["variance", "orthogonality"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True, squeeze=False)
    rr = [r for r in runs if r in set(resid.run)]
    for ax, p in zip(axes[0], preds):
        sub = resid[resid.predictor == p].set_index("run").reindex(rr)
        y = np.arange(len(rr))
        ax.errorbar(sub.coef, y, xerr=[sub.coef - sub.ci_low, sub.ci_high - sub.coef],
                    fmt="o", capsize=2)
        ax.axvline(0, color="red", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(rr, fontsize=7)
        ax.set_title(p, fontsize=9); ax.set_xlabel("coef (log-odds / +1 SD)")
    axes[0][0].invert_yaxis()
    fig.suptitle("Q2 (FWL residual, depth-adjusted): correct ~ variance + orthogonality", fontsize=9)
    _save(fig, stem, dpi)


def plot_2d_heatmap(grid: pd.DataFrame, *, run: str, stem, dpi=200):
    """variance x norm_resid -> mean correctness (decorrelated axes => grid fills)."""
    sub = grid[grid.run == run]
    if sub.empty:
        return
    nx, ny = sub.ix.max() + 1, sub.iy.max() + 1
    M = np.full((ny, nx), np.nan)
    for _, row in sub.iterrows():
        M[int(row.iy), int(row.ix)] = row["mean"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("variance (informativeness) bin")
    ax.set_ylabel("norm_resid (anchor coverage) bin")
    ax.set_title(f"P(correct) — {run}", fontsize=9)
    fig.colorbar(im, ax=ax, label="P(correct)")
    _save(fig, stem, dpi)


def plot_scatter_ceiling(df: pd.DataFrame, *, run: str, color_by: str, stem, dpi=200, max_pts=8000):
    """(variance, emb_norm) scatter colored by outcome, with the empirical Cauchy-Schwarz
    upper envelope. Orthogonal failures = points far BELOW the envelope at high variance."""
    sub = df[df.run == run]
    if sub.empty:
        return
    if len(sub) > max_pts:
        sub = sub.sample(max_pts, random_state=0)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    sc = ax.scatter(sub.variance, sub.emb_norm, c=sub[color_by], cmap="coolwarm_r",
                    s=4, alpha=0.5, vmin=0, vmax=1 if color_by == "correct" else None)
    # empirical envelope: max emb_norm per variance bin
    g = df.assign(vb=pd.qcut(df.variance, 30, labels=False, duplicates="drop")).groupby("vb")
    env = g.agg(v=("variance", "mean"), n=("emb_norm", "max")).sort_values("v")
    ax.plot(env.v, env.n, "k--", lw=1, label="empirical norm ceiling")
    ax.set_xlabel("variance"); ax.set_ylabel("emb_norm")
    ax.set_title(f"(variance, emb_norm) by {color_by} — {run}", fontsize=9)
    ax.legend(fontsize=7); fig.colorbar(sc, ax=ax, label=color_by)
    _save(fig, stem, dpi)


def plot_crossmodel_interaction(interactions: pd.DataFrame, *, reference_run: str, stem, dpi=200):
    if interactions.empty:
        return
    preds = list(dict.fromkeys(interactions.predictor))
    fig, axes = plt.subplots(1, len(preds), figsize=(3.4 * len(preds), 2.6), sharex=True, squeeze=False)
    for ax, p in zip(axes[0], preds):
        s = interactions[interactions.predictor == p]
        y = np.arange(len(s)); sig = s.get("reject_bh", pd.Series([False] * len(s))).to_numpy()
        ax.errorbar(s.coef, y, xerr=[s.coef - s.ci_low, s.ci_high - s.coef], fmt="none", ecolor="0.4", capsize=2)
        ax.scatter(s.coef, y, c=["C1" if v else "white" for v in sig], edgecolors="C1", zorder=3)
        ax.axvline(0, color="red", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(s.variant, fontsize=7)
        ax.set_title(p, fontsize=9); ax.set_xlabel(f"Δ slope vs {reference_run}")
    axes[0][0].invert_yaxis()
    fig.suptitle("Cross-model: does geometry-reliance differ from ce_base? (filled = BH-FDR sig)", fontsize=9)
    _save(fig, stem, dpi)
