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
    """Q2 summary: variance (informativeness) & norm_resid coefs per model."""
    if resid.empty:
        return
    preds = ["variance", "norm_resid"]
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
    fig.suptitle("Q2 (FWL residual, depth-adjusted): correct ~ variance + norm_resid", fontsize=9)
    _save(fig, stem, dpi)


def plot_geometry_attenuation(att: pd.DataFrame, *, runs, stem, dpi=200):
    """G2 bridge: geometry coef before (base) vs after (adjusted) operator main effects.

    Attenuation toward 0 from base->adjusted = the failure axis is (partly) operator structure;
    a stable coef = structure-independent. One panel per predictor, two markers per model."""
    if att.empty:
        return
    preds = ["variance", "norm_resid"]
    rr = [r for r in runs if r in set(att.run)]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True, squeeze=False)
    off = 0.16
    for ax, p in zip(axes[0], preds):
        sub = att[att.predictor == p].set_index("run").reindex(rr)
        y = np.arange(len(rr))
        ax.errorbar(sub.coef_base, y - off,
                    xerr=[sub.coef_base - sub.base_ci_low, sub.base_ci_high - sub.coef_base],
                    fmt="o", capsize=2, color="tab:gray", label="base")
        ax.errorbar(sub.coef_adjusted, y + off,
                    xerr=[sub.coef_adjusted - sub.adj_ci_low, sub.adj_ci_high - sub.coef_adjusted],
                    fmt="s", capsize=2, color="tab:blue", label="+ operators")
        ax.axvline(0, color="red", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(rr, fontsize=7)
        ax.set_title(p, fontsize=9); ax.set_xlabel("coef (log-odds / +1 SD)")
    axes[0][0].invert_yaxis(); axes[0][1].legend(fontsize=7)
    fig.suptitle("G2: does the geometry effect survive operator adjustment?", fontsize=9)
    _save(fig, stem, dpi)


def plot_geomop_joint_forest(coef: pd.DataFrame, *, runs, outcome: str, stem, dpi=200):
    """Joint geometry+operator model coefficients as a forest (unified RQ2 diagnostic / RQ3
    cross-model). One row per predictor — the two geometry rows first, then the operators,
    separated by a divider — and one colored series per model. Geometry coefs are per +1 SD,
    operator coefs per presence: read sign/significance/cross-model shifts, NOT magnitude
    across the two blocks. With one run it is the clean RQ2 figure; with all runs, the RQ3
    comparison (descriptive — overlapping CIs are NOT a between-model significance test)."""
    sub = coef[coef.outcome == outcome]
    rr = [r for r in runs if r in set(sub.run)]
    if sub.empty or not rr:
        return
    geom = [t for t in sub[sub.kind == "geometry"].term.unique()]
    # order operators by CE-base coef if present, else first available run
    base = rr[0] if "ce_base" not in rr else "ce_base"
    ops = (sub[(sub.kind == "operator") & (sub.run == base)]
           .sort_values("coef").term.tolist())
    ops += [t for t in sub[sub.kind == "operator"].term.unique() if t not in ops]
    order = geom + ops
    ypos = {t: i for i, t in enumerate(order)}
    pal = _palette(rr); n = len(rr)
    fig, ax = plt.subplots(figsize=(5.8, 0.42 * len(order) + 1.3))
    for j, r in enumerate(rr):
        rdf = sub[sub.run == r]
        dy = (j - (n - 1) / 2) * (0.62 / max(n, 1))
        y = [ypos[t] + dy for t in rdf.term]
        ax.errorbar(rdf.coef, y, xerr=[rdf.coef - rdf.ci_low, rdf.ci_high - rdf.coef],
                    fmt="o", ms=4, lw=1, capsize=2, color=pal[r], label=r)
    ax.axvline(0, color="red", ls="--", lw=0.8)
    ax.axhline(len(geom) - 0.5, color="0.7", ls=":", lw=0.8)   # geometry | operator divider
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
    unit = "  (geom: /+1 SD, op: /presence)"
    ax.set_xlabel(("log-odds" if outcome == "correct" else "Δ semantic_distance") + unit)
    title = f"Joint geometry+operator model — {outcome}"
    ax.set_title(title + (f"  [{rr[0]}]" if n == 1 else "  [all models]"), fontsize=9)
    if n > 1:
        ax.legend(fontsize=7)
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


def plot_feasibility_floor(desc: pd.DataFrame, *, stem, dpi=200):
    """RQ1 floor (G1b): greedy semantic-equivalence rate per run (conditioned + ablations),
    +/-95% CI. The gap between the conditioned bar and the ablation bars is the floor."""
    if desc.empty or "semantic_equiv_rate" not in desc.columns:
        return
    sub = desc.reset_index(drop=True)
    x = np.arange(len(sub))
    yerr = [sub.semantic_equiv_rate - sub.semantic_equiv_rate_ci_low,
            sub.semantic_equiv_rate_ci_high - sub.semantic_equiv_rate]
    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(sub), 3.4))
    colors = ["C0" if "ablation" not in r else "0.6" for r in sub.run]
    ax.bar(x, sub.semantic_equiv_rate, color=colors)
    ax.errorbar(x, sub.semantic_equiv_rate, yerr=yerr, fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(sub.run, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("greedy semantic-equiv rate")
    ax.set_title("RQ1 floor: conditioned vs embedding-ablated", fontsize=9)
    _save(fig, stem, dpi)


def plot_operator_geometry_contrast(contrast: pd.DataFrame, *, geom: str, stem, dpi=200):
    """G2: per-operator (with - without) difference in a geometry feature, +/-95% CI.

    For ``geom='norm_resid'`` a negative bar = the operator's targets are more
    anchor-orthogonal (the under-served cell over-represents that operator)."""
    sub = contrast[contrast.geom == geom]
    if sub.empty:
        return
    sub = sub.sort_values("diff")
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar(sub["diff"], y, xerr=[sub["diff"] - sub.ci_low, sub.ci_high - sub["diff"]],
                fmt="o", capsize=2, color="C0")
    ax.axvline(0, color="red", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(sub.op, fontsize=8)
    ax.set_xlabel(f"Δ {geom}  (with operator − without)")
    ax.set_title(f"G2: operator presence vs {geom}", fontsize=9)
    _save(fig, stem, dpi)


def plot_operator_orthogonality_forest(reg: pd.DataFrame, *, response: str, stem, dpi=200):
    """G2: adjusted OLS coefficients of each operator on a z-scored geometry response
    (filled = BH-FDR significant)."""
    sub = reg[reg.response == response]
    if sub.empty:
        return
    sub = sub.sort_values("coef")
    y = np.arange(len(sub))
    sig = sub.get("reject_bh", pd.Series([False] * len(sub))).to_numpy()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar(sub.coef, y, xerr=[sub.coef - sub.ci_low, sub.ci_high - sub.coef],
                fmt="none", ecolor="0.4", capsize=2)
    ax.scatter(sub.coef, y, c=["C1" if s else "white" for s in sig], edgecolors="C1", zorder=3)
    ax.axvline(0, color="red", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(sub.op, fontsize=8)
    ax.set_xlabel(f"coef on z({response})  (depth-adjusted, +1 presence)")
    ax.set_title(f"G2: adjusted operator → {response} (filled = BH-FDR sig)", fontsize=9)
    _save(fig, stem, dpi)


def plot_flip_geometry(profile: pd.DataFrame, *, geom: str, variants, stem, dpi=200):
    """G4: mean of a geometry feature over each flip set (both_correct / regression /
    recovery), per variant, +/-95% CI."""
    sub = profile[profile.geom == geom]
    if sub.empty:
        return
    cats = ["both_correct", "regression", "recovery"]
    cat_col = {"both_correct": "0.5", "regression": "C3", "recovery": "C2"}
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(variants), 3.4))
    xpos = {v: i for i, v in enumerate(variants)}
    off = {"both_correct": -0.22, "regression": 0.0, "recovery": 0.22}
    for cat in cats:
        c = sub[sub.category == cat]
        xs = [xpos[v] + off[cat] for v in c.variant if v in xpos]
        cc = c[c.variant.isin(xpos)]
        ax.errorbar(xs, cc["mean"], yerr=[cc["mean"] - cc.ci_low, cc.ci_high - cc["mean"]],
                    fmt="o", capsize=2, color=cat_col[cat], label=cat)
    ax.set_xticks(list(xpos.values())); ax.set_xticklabels(list(xpos), rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(f"mean {geom}")
    ax.set_title(f"G4: {geom} by flip set vs reference", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, stem, dpi)


def plot_flip_operator_logodds(lor: pd.DataFrame, *, variants, stem, dpi=200):
    """G4: per-variant forest of log-odds that an operator predicts a regression
    (ref-correct → variant-wrong) vs staying correct."""
    vs = [v for v in variants if v in set(lor.variant)]
    if not vs:
        return
    fig, axes = plt.subplots(1, len(vs), figsize=(3.0 * len(vs), 3.2), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], vs):
        s = lor[lor.variant == v].sort_values("op")
        y = np.arange(len(s))
        ax.errorbar(s.log_odds_ratio, y, xerr=[s.log_odds_ratio - s.ci_low, s.ci_high - s.log_odds_ratio],
                    fmt="o", capsize=2)
        ax.axvline(0, color="red", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(s.op, fontsize=8)
        ax.set_title(v, fontsize=8); ax.set_xlabel("log-odds (regression)")
    fig.suptitle("G4: operators predicting RL regression (given ref correct)", fontsize=9)
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
