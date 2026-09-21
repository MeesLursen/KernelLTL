"""Figures for Experiment 3 (reads the CSV tables written by run_exp3.py).

  fig_test1_slopes.{pdf,png}   per-bin slope of delta_variance on u against the
                               bin's mean target variance, misses and valid
                               subsets, with V_0 as a vertical rule when known.
  fig_test1_scatter.{pdf,png}  delta_variance against u for the misses, coloured
                               by variance bin, with the pooled within-bin slope.
  fig_test1_by_range.{pdf,png} pooled within-bin slope per target-variance quartile,
                               unadjusted and adjusted for depth + operators.
  fig_test2_curves.{pdf,png}   six per-scale curves against c (log axis; c = 0
                               drawn as a detached point at the left).
  fig_test2_rho.{pdf,png}      histogram of the per-target Spearman rho between
                               c and generated variance, with the invariant share.

Palette and rcParams follow analysis_exp2/plot_exp2.py so the chapters match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C1 = "#2a78d6"
C2 = "#eb6834"
ORD = ("#86b6ef", "#2a78d6", "#104281")
INK = "#1a1a1a"
INK_2 = "#5c5c5c"
MUTED = "#898781"
GRID = "#e6e6e6"
RULE = "#c3c2b7"
SURFACE = "white"

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "axes.edgecolor": "#b0b0b0",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------- Test 1 ---------------------------------- #

def fig_test1_slopes(per_bin: pd.DataFrame, pooled: pd.DataFrame, out: Path) -> None:
    """Per-bin slopes, unadjusted (filled) and adjusted for depth + operators (hollow)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    v0 = pooled["v0"].iloc[0] if "v0" in pooled.columns and pooled["v0"].notna().any() else None
    for ax, subset, color in zip(axes, ("misses", "valid"), (C1, C2)):
        d = per_bin[per_bin["subset"] == subset].dropna(subset=["slope"])
        p = pooled[pooled["subset"] == subset].iloc[0]
        ax.axhline(0, color=RULE, linewidth=0.8)
        x = d["mean_variance"].to_numpy()
        ax.errorbar(x * 0.97, d["slope"], yerr=[d["slope"] - d["slope_lo"], d["slope_hi"] - d["slope"]],
                    fmt="o", ms=3, color=color, ecolor=color, elinewidth=0.6, capsize=0, label="unadjusted")
        ax.errorbar(x * 1.03, d["slope_adj"],
                    yerr=[d["slope_adj"] - d["slope_adj_lo"], d["slope_adj_hi"] - d["slope_adj"]],
                    fmt="o", ms=3, mfc="white", color=color, ecolor=color, elinewidth=0.6, capsize=0,
                    alpha=0.8, label="adjusted (depth + operators)")
        ax.axhline(p["slope"], color=color, linewidth=1.0, linestyle="--")
        ax.axhline(p["slope_adj"], color=color, linewidth=1.0, linestyle=":")
        if v0 is not None:
            ax.axvline(v0, color=INK_2, linewidth=0.8, linestyle=":")
            ax.text(v0, ax.get_ylim()[1], " $V_0$", color=INK_2, va="top", ha="left", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("bin mean target variance $V_\\phi$")
        ax.set_title(subset, fontsize=9, loc="left")
        ax.text(0.02, 0.97,
                f"pooled: {p['slope']:+.2e} [{p['slope_lo']:+.2e}, {p['slope_hi']:+.2e}]\n"
                f"adjusted: {p['slope_adj']:+.2e} [{p['slope_adj_lo']:+.2e}, {p['slope_adj_hi']:+.2e}]",
                transform=ax.transAxes, va="top", ha="left", fontsize=7, color=color)
        _style(ax)
    axes[0].set_ylabel("slope of $V_{gen} - V_\\phi$ on $u$")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, out, "fig_test1_slopes")


def fig_test1_by_range(by_range: pd.DataFrame, out: Path) -> None:
    """Pooled within-bin slope per target-variance quartile, both arms, both subsets."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for ax, subset, color in zip(axes, ("misses", "valid"), (C1, C2)):
        d = by_range[by_range["subset"] == subset].sort_values("range")
        xs = np.arange(len(d))
        ax.axhline(0, color=RULE, linewidth=0.8)
        for k, (arm, mfc, lab) in enumerate((("", color, "unadjusted"),
                                             ("_adj", "white", "adjusted (depth + operators)"))):
            y = d[f"slope{arm}"].to_numpy()
            ax.errorbar(xs + (k - 0.5) * 0.18, y,
                        yerr=[y - d[f"slope{arm}_lo"], d[f"slope{arm}_hi"] - y],
                        fmt="o", ms=4, color=color, mfc=mfc, ecolor=color, elinewidth=0.9, capsize=2,
                        label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"[{lo:.3g}, {hi:.3g})\n$n$={n}" for lo, hi, n in
                            zip(d["v_lo"], d["v_hi"], d["n"])], fontsize=7)
        ax.set_xlabel("target-variance quartile")
        ax.set_title(subset, fontsize=9, loc="left")
        _style(ax)
    axes[0].set_ylabel("pooled within-bin slope of $V_{gen} - V_\\phi$ on $u$")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, out, "fig_test1_by_range")


def fig_test1_scatter(per_target: pd.DataFrame, pooled: pd.DataFrame, out: Path) -> None:
    d = per_target[~per_target["is_invalid"] & ~per_target["is_semantic_equivalent"]]
    p = pooled[pooled["subset"] == "misses"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.axhline(0, color=RULE, linewidth=0.8)
    sc = ax.scatter(d["u"], d["delta_variance"], c=np.log10(d["variance"]), cmap="Blues",
                    s=6, alpha=0.7, linewidths=0)
    xs = np.linspace(d["u"].min(), d["u"].max(), 2)
    ax.plot(xs, p["slope"] * xs, color=C2, linewidth=1.2, label="pooled within-bin slope")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("$\\log_{10} V_\\phi$")
    ax.set_xlabel("$u$ (studentised within-bin log-norm residual)")
    ax.set_ylabel("$V_{gen} - V_\\phi$  (misses)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    _save(fig, out, "fig_test1_scatter")


# --------------------------------- Test 2 ---------------------------------- #

def _xpos(scales: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Log-axis positions; c = 0 is detached to the left of the smallest positive scale."""
    pos = scales[scales > 0]
    x = scales.astype(float).copy()
    x0 = None
    if (scales == 0).any():
        x0 = float(pos.min()) / 3.0
        x[scales == 0] = x0
    return x, x0


def fig_test2_curves(per_scale: pd.DataFrame, out: Path) -> None:
    d = per_scale.sort_values("scale")
    scales = d["scale"].to_numpy(dtype=float)
    x, x0 = _xpos(scales)
    panels = [
        ("generated_variance", "mean generated variance $V_{gen}$", True),
        ("delta_variance", "mean $V_{gen} - V_\\phi$", False),
        ("equivalence", "semantic-equivalence rate", False),
        ("validity", "validity rate", False),
        ("direction_cosine", "cos$(\\mathrm{emb}(\\hat\\phi), \\mathrm{emb}(\\phi))$", False),
        ("entropy", "mean per-token entropy (nats)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0))
    for ax, (col, label, with_target) in zip(axes.ravel(), panels):
        y, lo, hi = d[col].to_numpy(), d[f"{col}_lo"].to_numpy(), d[f"{col}_hi"].to_numpy()
        m = scales > 0
        ax.fill_between(x[m], lo[m], hi[m], color=C1, alpha=0.15, linewidth=0)
        ax.plot(x[m], y[m], "o-", color=C1, ms=3, linewidth=1.1)
        if x0 is not None:
            ax.errorbar(x[~m], y[~m], yerr=[y[~m] - lo[~m], hi[~m] - y[~m]],
                        fmt="s", ms=3.5, color=INK_2, elinewidth=0.7, capsize=0)
        if with_target:
            ax.axhline(d["target_variance"].iloc[0], color=C2, linewidth=0.9, linestyle="--")
            ax.text(x[m].max(), d["target_variance"].iloc[0], " mean $V_\\phi$", color=C2,
                    va="bottom", ha="right", fontsize=7)
        if col == "delta_variance":
            ax.axhline(0, color=RULE, linewidth=0.8)
        ax.axvline(1.0, color=RULE, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ticks = [c for c in scales if c > 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{c:g}" for c in ticks], fontsize=7)
        if x0 is not None:
            ax.text(x0, ax.get_ylim()[0], "0", ha="center", va="top", fontsize=7, color=INK_2)
        ax.set_xlabel("scale $c$ on $\\mathrm{emb}(\\phi)$")
        ax.set_ylabel(label, fontsize=8)
        _style(ax)
    fig.tight_layout()
    _save(fig, out, "fig_test2_curves")


def fig_test2_rho(per_target: pd.DataFrame, monotonicity: pd.DataFrame, out: Path) -> None:
    rho = per_target["rho_scale_variance"].dropna()
    m = monotonicity.iloc[0]
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.hist(rho, bins=np.linspace(-1, 1, 41), color=C1, alpha=0.85, linewidth=0)
    ax.axvline(0, color=RULE, linewidth=0.8)
    ax.axvline(m["mean_rho"], color=C2, linewidth=1.2, linestyle="--")
    ax.set_xlabel("per-target Spearman $\\rho(c, V_{gen})$, $c > 0$")
    ax.set_ylabel("targets")
    ax.set_title(f"mean $\\rho$ = {m['mean_rho']:+.3f} [{m['mean_rho_lo']:+.3f}, {m['mean_rho_hi']:+.3f}]; "
                 f"increasing {m['share_increasing']:.1%}, invariant {m['share_invariant']:.1%}",
                 fontsize=8)
    _style(ax)
    _save(fig, out, "fig_test2_rho")


# --------------------------------- driver ---------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the Experiment 3 figures from run_exp3.py tables.")
    ap.add_argument("--tables-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    tables = args.tables_dir
    out = args.output_dir or tables / "figures"
    out.mkdir(parents=True, exist_ok=True)

    if (tables / "test1_per_bin.csv").exists():
        per_bin = pd.read_csv(tables / "test1_per_bin.csv")
        pooled = pd.read_csv(tables / "test1_pooled.csv")
        per_target = pd.read_csv(tables / "test1_per_target.csv")
        fig_test1_slopes(per_bin, pooled, out)
        fig_test1_scatter(per_target, pooled, out)
        fig_test1_by_range(pd.read_csv(tables / "test1_by_range.csv"), out)
    if (tables / "test2_per_scale.csv").exists():
        per_scale = pd.read_csv(tables / "test2_per_scale.csv")
        per_target2 = pd.read_csv(tables / "test2_per_target.csv")
        mono = pd.read_csv(tables / "test2_monotonicity.csv")
        fig_test2_curves(per_scale, out)
        fig_test2_rho(per_target2, mono, out)
    print(f"figures under {out}")


if __name__ == "__main__":
    main()
