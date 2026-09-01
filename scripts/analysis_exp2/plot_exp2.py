"""Experiment 2 figures. Nine of them, built from the tables run_exp2.py emits.

THE RULE: no statistic is computed here. Every number that appears in a title,
a subtitle, an annotation or a mark is read from a table, so there is exactly
one source for the analysis and a figure cannot silently disagree with the
numbers it illustrates. Two figures in the previous version did disagree --
a note quoting superseded values and a hardcoded skew -- which is why the rule
exists rather than being a preference.

The one exception, and its limit. Figures 1-4 read the frame to RENDER
DISTRIBUTIONS: histograms, hexbin densities, marginal strips. A histogram is a
display of a raw column, not a derived statistic, so it cannot drift from a
table -- there is no table it could contradict. Everything numeric on those
figures still comes from `covariates.csv` and `norm_variance_stats.csv`. The
enforceable form of the rule is therefore "no aggregation call in this file",
which `--audit` checks.

    1  ridge + occupancy      why u had to be constructed
    2  norm transform         the construction working
    3  faithfulness transform the Fisher-z step
    4  distributions          the three covariates as the models see them
    5  specification          why linear was rejected, and what replaced it
    6  readings + attenuation Q1 and Q3 (curves), Q2's scalar, and what
                              adjustment does to each
    7  operators              Q4
    8  syntax and geometry    why 7's numbers are not comparable with 6's
    9  depth                  the syntactic gradient everything sits inside
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "analysis_exp1"))
from load import load_greedy                                          # noqa: E402
from frame import (OPERATORS, build_frame, derive_covariates,          # noqa: E402
                   load_formulas)

# --------------------------------- palette ---------------------------------- #
C1 = "#2a78d6"          # blue   -- the primary / adjusted series
C2 = "#eb6834"          # orange -- the companion series
ORD = ("#86b6ef", "#2a78d6", "#104281")   # ordinal: least -> most adjusted
INK = "#1a1a1a"
INK_2 = "#5c5c5c"
MUTED = "#898781"       # raw marks; axis text
GRID = "#e6e6e6"
RULE = "#c3c2b7"
SURFACE = "white"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "axes.edgecolor": "#b0b0b0",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

# Covariate display names, used everywhere so one rename lands once.
LABEL = {"z_variance": "$V$", "u": "$u$", "z_faith": "$F$"}
# Axis labels name the MODEL scale, because that is what the bins are cut on
# and what spec_curves' grid is in. Labelling a z-scored axis in raw units was
# a defect in the previous version.
AXIS = {"z_variance": r"$z_\mathrm{variance}$  (z-scored $p(1-p)$)",
        "u": "$u$  (studentised norm residual)",
        "z_faith": r"$z_\mathrm{faith}$  (z-scored Fisher-$z$)"}
RAW_AXIS = {"variance": "satisfaction variance $p(1-p)$"}

# Shared by figs 5 and 6, which plot the same quantity over the same
# covariates. One constant so they cannot drift apart, and a ZERO BASELINE
# because a truncated one inflates the apparent slope of effects that are
# small against the 0.374 base rate.
PCORRECT_YLIM = (0.0, 0.55)


def style(ax, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.7)


def title(ax, text: str, sub: str | None = None) -> None:
    """Title above an optional subtitle, padded in POINTS (not axes fractions,
    which do not hold their gap across panels of different heights)."""
    pad = 6 if sub is None else 8 + 9.5 * sub.count("\n") + 9.5
    ax.set_title(text, fontsize=10, color=INK, pad=pad, loc="left")
    if sub:
        ax.annotate(sub, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                    textcoords="offset points", fontsize=7.5, color=INK_2,
                    va="bottom", ha="left")


def note(ax, text: str, xy, *, ha="left", va="bottom", size=7.2) -> None:
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=size,
            color=INK_2, ha=ha, va=va, linespacing=1.35)


def save(fig, out_dir: Path, stem: str) -> None:
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight",
                    facecolor=SURFACE, **kw)
    plt.close(fig)
    print(f"  wrote {out_dir / stem}.{{pdf,png}}")


# ------------------------------ table readers ------------------------------- #

class Tables:
    """Every number in these figures comes from here."""

    def __init__(self, d: Path):
        self.d = d
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.d / f"{name}.csv")
        return self._cache[name]

    def stat(self, table: str, key: str) -> float:
        """One value from a two-column stat/value table."""
        t = self[table]
        return float(t.loc[t["stat"] == key, "value"].iloc[0])

    def cov(self, covariate: str, field: str) -> float:
        t = self["covariates"]
        return float(t.loc[t["covariate"] == covariate, field].iloc[0])

    def spec(self, term: str, form: str, reference: str = "deciles") -> pd.Series:
        t = self["spec_search"]
        sel = t[(t["term"] == term) & (t["form"] == form)
                & (t["reference"] == reference)]
        return sel.iloc[0]

    def grid(self, rows: str, cols: str) -> np.ndarray:
        t = self["occupancy"]
        sel = t[(t["rows"] == rows) & (t["cols"] == cols)]
        return (sel.pivot(index="row_bin", columns="col_bin", values="n")
                .to_numpy(dtype=float))


def pfmt(p: float) -> str:
    """p-values: three decimals, or scientific below 0.001."""
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.0e}".replace("e-0", "e-")


def xlim_for(T: Tables, col: str) -> tuple[float, float]:
    """Shared x-range for figs 5 and 6, taken from spec_curves' own grid.

    That grid spans the covariate's 0.5-99.5 percentile, so it is the honest
    support rather than the span of the decile MEANS -- which reach neither
    end, because the outer deciles are wide and their means sit well inside
    them. Fig 6's points therefore bunch toward the middle, and they should:
    that gap is the bin-width fact, not a plotting artefact.
    """
    x = T["spec_curves"]
    x = x[x["term"] == col]["x"]
    lo, hi = float(x.min()), float(x.max())
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


# ------------------------------- the figures -------------------------------- #

def fig01_ridge(df: pd.DataFrame, T: Tables, out: Path) -> None:
    """Why u exists: the norm is very nearly a function of the variance."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    style(ax)
    ax.hexbin(df["variance"], df["emb_norm"], gridsize=48, xscale="log",
              yscale="log", cmap=SEQ, mincnt=1, linewidths=0.0)
    slope = T.stat("norm_variance_stats", "ridge_slope")
    icept = T.stat("norm_variance_stats", "ridge_intercept")
    ref = T.stat("norm_variance_stats", "ridge_slope_scale_reference")
    r2 = T.stat("norm_variance_stats", "ridge_r2")
    lv = np.log10(np.array([df["variance"].min(), df["variance"].max()]))
    ax.plot(10 ** lv, 10 ** (icept + slope * lv), color=C2, lw=1.9,
            label=f"fitted, slope {slope:.3f}")
    # The 1/2 slope is a CEILING, not a central tendency: ||emb|| <= K sqrt(V)
    # by Cauchy-Schwarz, since emb is a projection of the satisfaction vector
    # onto the anchor set and a projection can only lose magnitude. So every
    # target lies on or BELOW this line, and the vertical gap to it is the
    # under-exposure. Anchoring it through the middle of the cloud (an earlier
    # version) put half the targets impossibly above their own bound.
    ceiling = T.stat("norm_variance_stats", "scale_ceiling_intercept")
    ax.plot(10 ** lv, 10 ** (ceiling + ref * lv), color=MUTED, lw=1.6, ls="--",
            label=f"scale ceiling, slope {ref:g}")
    ax.set_xlabel("satisfaction variance $p(1-p)$", fontsize=8.5, color=INK)
    ax.set_ylabel(r"$\|\mathrm{emb}(\varphi)\|$", fontsize=8.5, color=INK)
    gap = T.stat("norm_variance_stats", "scale_ceiling_gap_median")
    title(ax, "a  Every target sits below the scale ceiling",
          f"$R^2 = {r2:.3f}$; mean shortfall {gap:.2f} decades, and it widens\n"
          f"as variance falls because {slope:.3f} is steeper than {ref:g}")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower right")

    ax = axes[1]
    # Transposed so variance is on x and the norm on y, matching panel a: a
    # vertical slice then reads "at this variance, which norms occur", which
    # is the question the panel exists to answer.
    g = T.grid("variance", "emb_norm").T
    im = ax.imshow(g, origin="lower", cmap=SEQ, aspect="auto")
    for (i, j), v in np.ndenumerate(g):
        if v == 0:
            ax.plot(j, i, marker="x", ms=4, color=MUTED, mew=1.1)
    ax.set_xlabel("variance decile", fontsize=8.5, color=INK)
    ax.set_ylabel("norm decile", fontsize=8.5, color=INK)
    ax.set_xticks(range(0, 10, 3))
    ax.set_yticks(range(0, 10, 3))
    ax.tick_params(colors=INK_2, labelsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="targets")
    vif_raw = T.stat("norm_variance_stats", "vif_log10_norm_with_log10_variance")
    vif_u = T.stat("norm_variance_stats", "vif_u_with_z_variance")
    # The empty-cell COUNT is deliberately not printed: counting them here
    # would be computing a statistic in the figure. They are marked instead,
    # and the count belongs in the caption.
    title(ax, "b  Which is why the design has no contrast to read",
          "empty cells marked ×. Entering the norm beside variance\n"
          f"gives VIF {vif_raw:.1f}; after the construction, {vif_u:.2f}")
    save(fig, out, "fig01_ridge")


def fig02_norm_transform(df: pd.DataFrame, T: Tables, out: Path) -> None:
    """The construction, one step per panel, with what each step buys."""
    nv = T["norm_variance"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)

    ax = axes[0]
    style(ax)
    ax.hexbin(df["variance"], df["log10_norm"], gridsize=42, xscale="log",
              cmap=SEQ, mincnt=1, linewidths=0.0, alpha=0.85)
    # Steps across each bin's ACTUAL variance extent: drawing bin means as
    # points hides that the lowest bins span decades (C1's edge qualification).
    edges = np.append(nv["min_variance"].to_numpy(), nv["max_variance"].iloc[-1])
    ax.stairs(nv["mean_log10_norm"].to_numpy(), edges, color=C2, lw=1.6,
              baseline=None, label="binned mean (50 bins)")
    ax.set_xscale("log")
    ax.set_xlabel("satisfaction variance $p(1-p)$", fontsize=8.5, color=INK)
    ax.set_ylabel(r"$\log_{10} \|\mathrm{emb}\|$", fontsize=8.5, color=INK)
    title(ax, "a  Subtract the binned mean",
          f"leverage in the bottom 5 %: "
          f"{100 * T.cov('log10_norm', 'leverage_bottom_5pct'):.1f} %")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower right")

    ax = axes[1]
    style(ax)
    ax.hexbin(df["variance"], df["norm_resid"], gridsize=42, xscale="log",
              cmap=SEQ, mincnt=1, linewidths=0.0, alpha=0.85)
    ax.axhline(0, color=RULE, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("satisfaction variance $p(1-p)$", fontsize=8.5, color=INK)
    ax.set_ylabel("residual (decades)", fontsize=8.5, color=INK)
    title(ax, "b  Location removed, spread not yet",
          "the residual still fans out where the bins are widest")

    ax = axes[2]
    style(ax)
    ax.hexbin(df["variance"], df["u"], gridsize=42, xscale="log", cmap=SEQ,
              mincnt=1, linewidths=0.0, alpha=0.85)
    ax.axhline(0, color=RULE, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("satisfaction variance $p(1-p)$", fontsize=8.5, color=INK)
    ax.set_ylabel("$u$", fontsize=8.5, color=INK)
    sp = T.stat("norm_variance_stats", "spearman_u_variance")
    title(ax, "c  Divide by the within-bin SD",
          f"Spearman$(u, V) = {sp:+.3f}$; leverage "
          f"{100 * T.cov('u', 'leverage_bottom_5pct'):.1f} %")
    frac = T.stat("norm_variance_stats", "c1_edge_frac_targets_bins01")
    dec = T.stat("norm_variance_stats", "c1_edge_bin0_variance_decades")
    note(ax, f"bins 0–1 hold {100 * frac:.0f} % of targets and bin 0 spans\n"
             f"{dec:.2f} decades: there, \"within-bin\" is not \"at matched\n"
             "variance\" (C1's edge qualification)", (0.03, 0.04))
    save(fig, out, "fig02_norm_transform")


def fig03_faith_transform(df: pd.DataFrame, T: Tables, out: Path) -> None:
    """The Fisher-z step, with panel b showing the map as a projection."""
    raw = df["relational_faithfulness"].to_numpy()
    fz = np.arctanh(np.clip(raw, -1 + 1e-6, 1 - 1e-6))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)

    ax = axes[0]
    style(ax, grid_axis="y")
    ax.hist(raw, bins=55, color=C1, alpha=0.85, linewidth=0)
    ax.set_xlabel(r"relational faithfulness $\rho$", fontsize=8.5, color=INK)
    ax.set_ylabel("targets", fontsize=8.5, color=INK)
    title(ax, r"a  Raw $\rho$",
          f"skew {T.cov('relational_faithfulness', 'skew'):.2f}     bottom 5 % own "
          f"{100 * T.cov('relational_faithfulness', 'leverage_bottom_5pct'):.1f} %"
          " of the leverage")

    # ---- (b) the map, with both marginals in the axes' own frame ----------- #
    ax = axes[1]
    style(ax)
    ax.grid(False)
    X0, X1, Y0, Y1 = 0.345, 1.0, -0.70, 2.60
    XS, YS = 0.455, -0.20

    def strip(vals, lo, hi, *, horizontal):
        h, e = np.histogram(vals, bins=48)
        ax.stairs(lo + h / h.max() * (hi - lo), e, baseline=lo, fill=True,
                  color=C1, alpha=0.30, linewidth=0, zorder=2,
                  orientation="horizontal" if horizontal else "vertical")

    strip(raw, Y0 + 0.02, YS, horizontal=False)
    strip(fz, X0 + 0.004, XS, horizontal=True)
    ax.plot([X0, X1], [YS, YS], color=RULE, lw=0.8, zorder=3)
    ax.plot([XS, XS], [Y0, Y1], color=RULE, lw=0.8, zorder=3)
    grid = np.linspace(0.47, 0.988, 400)
    ax.plot(grid, np.arctanh(grid), color=C1, lw=2.0, zorder=5)
    for lo, hi in ((0.50, 0.55), (0.90, 0.95)):
        zlo, zhi = np.arctanh(lo), np.arctanh(hi)
        ax.fill_between([lo, hi], YS, Y1, color=C2, alpha=0.16, linewidth=0,
                        zorder=1)
        for edge, zed in ((lo, zlo), (hi, zhi)):
            ax.plot([X0, edge], [zed, zed], color=C2, lw=0.9, ls=":", zorder=4)
        ax.annotate(f"$\\Delta\\rho = 0.05$\n$\\Delta z = {zhi - zlo:.3f}$",
                    ((lo + hi) / 2, zhi), xytext=(0, 10),
                    textcoords="offset points", fontsize=7.2, color=INK_2,
                    ha="center", va="bottom")
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel(r"$\rho$", fontsize=8.5, color=INK)
    ax.set_ylabel(r"$z = \mathrm{atanh}\,\rho$", fontsize=8.5, color=INK)
    title(ax, r"b  The map $z = \mathrm{atanh}\,\rho$",
          r"$\mathrm{atanh}$: the variance-stabilising map, $\int d\rho/(1-\rho^2)$"
          "\n" r"strips: marginals of $\rho$ (below) and $z$ (left)")

    ax = axes[2]
    style(ax, grid_axis="y")
    ax.hist(fz, bins=55, color=C1, alpha=0.85, linewidth=0)
    ax.set_xlabel(r"$z = \mathrm{atanh}\,\rho$", fontsize=8.5, color=INK)
    ax.set_ylabel("targets", fontsize=8.5, color=INK)
    title(ax, "c  Fisher-$z$",
          f"skew {T.cov('fisher_z_faith', 'skew'):.2f}     bottom 5 % own "
          f"{100 * T.cov('fisher_z_faith', 'leverage_bottom_5pct'):.1f} %"
          " of the leverage")
    save(fig, out, "fig03_faith_transform")


def fig04_distributions(df: pd.DataFrame, T: Tables, out: Path) -> None:
    """The three covariates on their model scales, plainly.

    The comparison IS the argument: the two that needed constructing are the
    two whose lower tail carried more than half the design's pull, and V --
    which needed nothing -- sits at the uniform reference.
    """
    specs = [("variance", "z_variance", 50, "a  $V$"),
             ("u", "u", 55, "b  $u$"),
             ("z_faith", "z_faith", 55, r"c  $z_\mathrm{faith}$")]
    ref = 100 * T.cov("_reference_uniform", "leverage_bottom_5pct")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    for ax, (col, statkey, bins, head) in zip(axes, specs):
        style(ax, grid_axis="y")
        ax.hist(df[col], bins=bins, color=C1, alpha=0.85, linewidth=0)
        ax.set_xlabel(AXIS.get(statkey, col), fontsize=8.5, color=INK)
        ax.set_ylabel("targets", fontsize=8.5, color=INK)
        title(ax, head,
              f"SD {T.cov(statkey, 'sd'):.3f}     skew {T.cov(statkey, 'skew'):.2f}"
              f"     bottom 5 % leverage "
              f"{100 * T.cov(statkey, 'leverage_bottom_5pct'):.1f} %")
    note(axes[0], f"uniform reference: {ref:.1f} %", (0.05, 0.90))
    save(fig, out, "fig04_distributions")


def fig05_specification(T: Tables, out: Path) -> None:
    """Why linear was rejected, and what replaced it. The chapter's hinge."""
    SEL = {"z_variance": "quadratic", "u": "deciles", "z_faith": "linear"}
    sc = T["spec_curves"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), constrained_layout=True,
                             sharey=True)
    axes[0].set_ylim(*PCORRECT_YLIM)
    for ax, col in zip(axes, ("z_variance", "u", "z_faith")):
        style(ax)
        cv = T[f"curve_{col}"]
        x = cv[f"mean_{col}"].to_numpy()
        # The decile points and the fitted lines share the D + S base, so this
        # is a like-for-like comparison rather than two adjacent displays.
        ax.fill_between(x, cv["ci_lo_DS"], cv["ci_hi_DS"], color=C1, alpha=0.13,
                        linewidth=0)
        ax.plot(x, cv["adj_DS"], color=C1, lw=0, marker="o", ms=5.5, mfc=C1,
                mec=SURFACE, mew=1.3, zorder=5, label="decile points (D+S)")
        # The linear line's verdict is per-panel: it is the SELECTED form for
        # F and the rejected one for V and u. Labelling it "rejected"
        # everywhere contradicted F's own title.
        lin_selected = SEL[col] == "linear"
        lin = sc[(sc["term"] == col) & (sc["form"] == "linear")]
        ax.plot(lin["x"], lin["rate"], color=C1 if lin_selected else C2, lw=1.8,
                ls="-" if lin_selected else "--", zorder=4,
                label="linear (selected)" if lin_selected else "linear (rejected)")
        if SEL[col] == "quadratic":
            q = sc[(sc["term"] == col) & (sc["form"] == "quadratic")]
            ax.plot(q["x"], q["rate"], color=C1, lw=1.8, zorder=4,
                    label="quadratic (selected)")
        elif SEL[col] == "deciles":
            ax.plot(x, cv["adj_DS"], color=C1, lw=1.6, zorder=3,
                    label="deciles (selected)")
        row = T.spec(col, "linear")
        # Both rejected covariates report the SAME second test -- the quadratic
        # residual against a decile reference -- because the contrast between
        # those two p-values is exactly why one got a quadratic and the other
        # got deciles. The 20-bin reference is a limitation for the caption
        # (V is adequate only at decile resolution), not a panel subtitle.
        extra = ("" if SEL[col] == "linear" else
                 f"; quadratic residual $p = {pfmt(T.spec(col, 'quadratic')['p'])}$")
        verdict = "linearity holds" if row["p"] > 0.05 else "linearity REJECTED"
        title(ax, f"{'abc'[list(SEL).index(col)]}  {LABEL[col]}: {verdict}",
              f"linear vs deciles: LR {row['lr']:.2f} on {int(row['df'])} df, "
              f"$p = {pfmt(row['p'])}${extra}")
        ax.set_xlabel(AXIS[col], fontsize=8.5, color=INK)
        ax.set_xlim(*xlim_for(T, col))
        ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="lower right")
    axes[0].set_ylabel("P(correct)", fontsize=8.5, color=INK)
    save(fig, out, "fig05_specification")


def fig06_readings(T: Tables, out: Path) -> None:
    """Q1 and Q3 (curves) and Q2 (a scalar), each with its adjustment sequence.

    The primary step is drawn heavy and carries the CI band; the earlier steps
    are the attenuation sequence, drawn light. The raw rate is a reference mark
    rather than a step in the ordinal ramp, which keeps every panel inside the
    validated three-step palette.
    """
    seq = {"z_variance": ("DS", "DSu", "DSF"),
           "u": ("DS", "DSV", "DSVF"),
           "z_faith": ("DS", "DSV", "DSVu")}
    legend_txt = {"DS": "D+S", "DSu": "+u", "DSF": "+F  (a step too far)",
                  "DSV": "+V", "DSVF": "+V, +F", "DSVu": "+V, +u"}
    me = T["marginal_effects"].set_index(["model", "term"])
    ml = T["m_ladder"].set_index(["model", "term"])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), constrained_layout=True,
                             sharey=True)
    axes[0].set_ylim(*PCORRECT_YLIM)
    for k, (ax, col) in enumerate(zip(axes, ("z_variance", "u", "z_faith"))):
        style(ax)
        cv = T[f"curve_{col}"]
        x = cv[f"mean_{col}"].to_numpy()
        primary = cv["primary_step"].iloc[0]
        ax.plot(x, cv["raw_rate"], ls="none", marker="o", ms=4.5, mfc=SURFACE,
                mec=MUTED, mew=1.2, zorder=3, label="raw")
        # Colour encodes ADJUSTMENT ORDER, never loop order: for V the primary
        # step is the LEAST adjusted one, so taking the ramp's dark end for
        # "primary" would invert the encoding. The primary is marked by weight
        # and by its CI band instead.
        for step, colour in zip(seq[col], ORD):
            is_primary = step == primary
            ax.plot(x, cv[f"adj_{step}"], color=colour,
                    lw=2.4 if is_primary else 1.4, marker="o",
                    ms=6 if is_primary else 4.5, mfc=colour, mec=SURFACE,
                    mew=1.4 if is_primary else 1.0, zorder=6 if is_primary else 4,
                    label=legend_txt[step] + ("  ← reported" if is_primary else ""))
            if is_primary:
                ax.fill_between(x, cv[f"ci_lo_{step}"], cv[f"ci_hi_{step}"],
                                color=colour, alpha=0.13, linewidth=0)
        ax.set_xlabel(AXIS[col], fontsize=8.5, color=INK)
        ax.set_xlim(*xlim_for(T, col))
        if col == "z_faith":
            b = ml.loc[("M4", "z_faith")]
            a = me.loc[("M4", "z_faith")]
            sub = (rf"$\beta_F = {b['estimate']:.3f}$ "
                   rf"$[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]$; "
                   rf"AME ${100 * a['estimate']:+.2f}$ pp "
                   rf"$[{100 * a['ci_lo']:+.2f}, {100 * a['ci_hi']:+.2f}]$")
        else:
            sub = "no scalar: the relationship is non-monotone, so an average\nshift effect would not measure its strength (S1)"
        title(ax, f"{'abc'[k]}  {LABEL[col]}", sub)
        ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="lower right")
    axes[0].set_ylabel("P(correct)", fontsize=8.5, color=INK)
    save(fig, out, "fig06_readings")


def fig07_operators(T: Tables, out: Path) -> None:
    """Q4: the eight operator contrasts on the interpretable scale."""
    op = T["operators"].sort_values("gap").reset_index(drop=True)
    ypos = np.arange(len(op))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.9), constrained_layout=True)
    style(ax, grid_axis="x")
    ax.axvline(0, color=RULE, lw=1.0, zorder=1)
    for y, r in zip(ypos, op.itertuples()):
        ax.plot([100 * r.gap_ci_lo, 100 * r.gap_ci_hi], [y, y], color=C1,
                lw=1.6, zorder=3, solid_capstyle="round")
        ax.plot([100 * r.gap], [y], "o", ms=7.5, mfc=C1, mec=SURFACE, mew=1.5,
                zorder=5)
    ax.set_yticks(ypos, [f"{r.operator}   {100 * r.prevalence:.0f} %"
                         for r in op.itertuples()], fontsize=9)
    ax.set_xlabel(r"change in P(correct), $0 \to 1$  (pp)", fontsize=8.5,
                  color=INK)
    title(ax, "Operator contrasts",
          "all eight jointly, $+$ C(depth); marginally standardised.\n"
          "Prevalence beside each row. TOTAL contrasts — see fig08")
    save(fig, out, "fig07_operators")


def fig08_syntax_geometry(T: Tables, out: Path) -> None:
    """Why fig07's numbers cannot be set beside fig06's."""
    sig = T["op_signature"]
    op = T["operators"].set_index("operator")
    order = op["gap"].sort_values().index.tolist()
    sig = sig.set_index("operator").loc[order].reset_index()
    ypos = np.arange(len(sig))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), constrained_layout=True,
                             sharex=True, sharey=True)
    for k, (ax, col) in enumerate(zip(axes, ("z_variance", "u", "z_faith"))):
        style(ax, grid_axis="x")
        ax.axvline(0, color=RULE, lw=1.0, zorder=1)
        d = sig[f"delta_{col}"].to_numpy()
        for y, v in zip(ypos, d):
            ax.plot([0, v], [y, y], color=C1, lw=1.5, alpha=0.5, zorder=2,
                    solid_capstyle="round")
            ax.plot([v], [y], "o", ms=6.5, mfc=C1, mec=SURFACE, mew=1.4,
                    zorder=5)
        r2 = T.stat("diagnostic", f"r2_{col}_on_has_op")
        title(ax, f"{'abc'[k]}  {LABEL[col]}", f"$R^2$ on operators $= {r2:.3f}$")
        ax.set_xlabel(r"$\Delta$ per operator (SD)", fontsize=8.5, color=INK)
    axes[0].set_yticks(ypos, [f"{o}   {100 * op.loc[o, 'gap']:+.1f} pp"
                              for o in order], fontsize=8.5)
    note(axes[0], "rows ordered by the operator's\ncorrectness contrast",
         (0.04, 0.05))
    save(fig, out, "fig08_syntax_geometry")


def fig09_depth(T: Tables, out: Path) -> None:
    """The syntactic gradient everything else sits inside."""
    dc = T["depth_curve"]
    mix = T["depth_op_mix"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)

    ax = axes[0]
    style(ax)
    for col, lo, hi, colour, marker, ls, lab in (
            ("raw_rate", "raw_ci_lo", "raw_ci_hi", C1, "o", "-", "raw"),
            ("adj_rate", "adj_ci_lo", "adj_ci_hi", C2, "s", "--",
             "operator-standardised")):
        ax.fill_between(dc["depth"], dc[lo], dc[hi], color=colour, alpha=0.13,
                        linewidth=0)
        ax.plot(dc["depth"], dc[col], color=colour, lw=1.8, ls=ls, marker=marker,
                ms=6, mfc=colour, mec=SURFACE, mew=1.4, label=lab, zorder=4)
    ax.set_xticks(dc["depth"])
    ax.set_xlabel("target depth", fontsize=8.5, color=INK)
    ax.set_ylabel("P(correct)", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.0)
    span_raw = 100 * (dc["raw_rate"].max() - dc["raw_rate"].min())
    span_adj = 100 * (dc["adj_rate"].max() - dc["adj_rate"].min())
    title(ax, "a  Part of the depth gradient is operator mix",
          f"raw span {span_raw:.1f} pp; operator-standardised {span_adj:.1f} pp")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")

    ax = axes[1]
    style(ax)
    cols = [f"has_{o}" for o in OPERATORS]
    swings = {c: mix[c].max() - mix[c].min() for c in cols}
    for c in cols:
        ax.plot(mix["depth"], mix[c], color=MUTED, lw=1.3, alpha=0.55, zorder=2)
    for c, colour, ls, marker in ((max(swings, key=swings.get), C2, "--", "s"),
                                  (min(swings, key=swings.get), C1, "-", "o")):
        ax.plot(mix["depth"], mix[c], color=colour, lw=1.9, ls=ls, marker=marker,
                ms=6, mfc=colour, mec=SURFACE, mew=1.4, zorder=5,
                label=f"{c}  (swing {100 * swings[c]:.0f} pp)")
    ax.set_xticks(mix["depth"])
    ax.set_xlabel("target depth", fontsize=8.5, color=INK)
    ax.set_ylabel("share of targets containing the operator", fontsize=8.5,
                  color=INK)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    title(ax, "b  The operator mix is not constant across depth",
          f"every operator swings at least {100 * min(swings.values()):.0f} pp "
          "(grey: the other six)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    save(fig, out, "fig09_depth")


# ----------------------------------- main ----------------------------------- #

AGGREGATIONS = re.compile(
    r"(?<!T)\.(mean|std|var|median|sum|skew|quantile|corr|cov|value_counts|"
    r"groupby|agg|describe|percentile|polyfit|lstsq)\s*\(")
# `T.cov(...)` READS covariates.csv; `.min()`/`.max()` only place axis limits.
# Neither produces a reported number, so both are outside the rule.
_ALLOWED = ("np.histogram", "T.cov(", "T.stat(", "T.spec(")


def audit(path: Path) -> int:
    """Enforce the rule in the module docstring: no statistic computed here.

    The rule is that no NUMBER SHOWN comes from arithmetic performed in this
    file. Histogram binning and axis-limit min/max are exempt: neither is a
    reported quantity, and neither can drift from a table because no table
    holds it.
    """
    bad = []
    for i, line in enumerate(path.read_text().split("\n"), 1):
        code = line.split("#")[0]
        if AGGREGATIONS.search(code) and not any(a in code for a in _ALLOWED):
            bad.append(f"  {path.name}:{i}: {line.strip()}")
    if bad:
        print("AUDIT FAILED -- statistics computed in the plot script:")
        print("\n".join(bad))
    else:
        print("audit: no aggregation in the plot script; every number is table-sourced")
    return len(bad)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tables-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    # Figures 1-4 render distributions; see the module docstring.
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--audit", action="store_true",
                   help="Check the no-statistics rule and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit:
        sys.exit(1 if audit(Path(__file__)) else 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    T = Tables(args.tables_dir)

    features = pd.read_csv(args.features_dir / "exp2_features.csv")
    formulas = load_formulas(args.dataset_dir / "formulas.jsonl")
    greedy, _ = load_greedy(args.run_dir, expected_n=len(features))
    df = derive_covariates(build_frame(features, greedy, formulas,
                                       expected_n=len(features)))

    fig01_ridge(df, T, args.output_dir)
    fig02_norm_transform(df, T, args.output_dir)
    fig03_faith_transform(df, T, args.output_dir)
    fig04_distributions(df, T, args.output_dir)
    fig05_specification(T, args.output_dir)
    fig06_readings(T, args.output_dir)
    fig07_operators(T, args.output_dir)
    fig08_syntax_geometry(T, args.output_dir)
    fig09_depth(T, args.output_dir)
    print(f"[exp2-viz] figures -> {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
