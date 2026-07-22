#!/usr/bin/env python3
"""Diagnostic plots for the three KernelLTL curriculum training runs.

Parses the SLURM logs in misc/ and emits, for each run, a 2x4 panel figure:
  row 1 -- train loss vs. eval loss (cross-entropy, nats)
  row 2 -- eval_semantic_distance vs. eval_semantic_equivalent_rate (both unitless in [0,1])
one column per curriculum stage. The epoch restored by load_best_model_at_end
(i.e. the checkpoint actually handed to the next stage) is marked on every panel.

Usage:  python scripts/plot_curriculum_diagnostics.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MISC = ROOT / "misc"
OUT = MISC / "figures"

# --- runs -------------------------------------------------------------------
# (job id, short tag, human label, log file, selection metric, greater_is_better)
RUNS = [
    ("24621364", "run3",
     "Run 3 — CE-stopped, LR halved per stage",
     "kernelltl_curriculum_24621364.out", "eval_loss", False),
    ("24643524", "run4_constlr",
     "Run 4 — CE-stopped, constant LR 1e-4",
     "kernelltl_curriculum_24643524.out", "eval_loss", False),
    ("24808377", "run5_semdist",
     "Run 5 — semantic-distance-stopped, constant LR 1e-4",
     "kernelltl_curriculum_24808377.out", "eval_semantic_distance", False),
]

PATIENCE = 10  # early-stopping patience in eval steps (= epochs here)

# --- palette (dataviz reference instance, all-pairs validated in light mode) --
C_TRAIN = "#2a78d6"   # slot 1, blue
C_EVAL = "#eb6834"    # slot 2, orange
C_DIST = "#1baf7a"    # slot 3, aqua
C_RATE = "#4a3aa7"    # slot 7, violet
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"

STAGES = ["stage1", "stage2", "stage3", "stage4"]

# --- parsing ----------------------------------------------------------------
RE_STAGE = re.compile(r"^Starting (stage\d) \(Run")
RE_LR_CFG = re.compile(r"^\s*Learning rate:\s*(\S+)")
RE_EVALHDR = re.compile(r"^\s*Eval @ epoch ([\d.]+) / step (\d+):")
RE_SEMDIST = re.compile(r"^\s*eval_semantic_distance:\s*([\d.]+)")
RE_SEMRATE = re.compile(r"^\s*eval_semantic_equivalent_rate:\s*([\d.]+)")


def parse_log(path: Path):
    """Return {stage: {'lr': float, 'rows': [dict sorted by epoch]}}."""
    stage, pending, stage_lr, rows = None, None, {}, {}
    for line in path.read_text().splitlines():
        if m := RE_STAGE.match(line):
            stage = m.group(1)
            continue
        if stage and (m := RE_LR_CFG.match(line)):
            stage_lr[stage] = float(m.group(1))
            continue
        if stage is None:
            continue
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict) or "epoch" not in d:
                continue
            r = rows.setdefault((stage, float(d["epoch"])),
                                {"epoch": float(d["epoch"])})
            if "loss" in d:
                r["train_loss"] = d["loss"]
                r["grad_norm"] = d.get("grad_norm")
                r["lr"] = d.get("learning_rate")
            if "eval_loss" in d:
                r["eval_loss"] = d["eval_loss"]
            continue
        if m := RE_EVALHDR.match(line):
            pending = float(m.group(1))
            rows.setdefault((stage, pending), {"epoch": pending})["step"] = int(m.group(2))
        elif m := RE_SEMDIST.match(line):
            rows[(stage, pending)]["sem_dist"] = float(m.group(1))
        elif m := RE_SEMRATE.match(line):
            rows[(stage, pending)]["sem_rate"] = float(m.group(1))

    out = {}
    for st in sorted({k[0] for k in rows}):
        rs = sorted((v for k, v in rows.items() if k[0] == st), key=lambda r: r["epoch"])
        out[st] = {"lr": stage_lr.get(st), "rows": rs}
    return out


def selected_epoch(rows, metric, greater_is_better):
    """Epoch restored by load_best_model_at_end."""
    key = {"eval_loss": "eval_loss", "eval_semantic_distance": "sem_dist"}[metric]
    cand = [r for r in rows if key in r]
    pick = max(cand, key=lambda r: r[key]) if greater_is_better else min(cand, key=lambda r: r[key])
    return pick


# --- plotting ---------------------------------------------------------------
def anno(ax, text, xy, dy, xmax, va="center"):
    """Annotate, flipping to the left of the point when it sits near the right edge."""
    right = xy[0] > 0.62 * xmax
    ax.annotate(text, xy=xy, xytext=(-5 if right else 5, dy),
                textcoords="offset points", fontsize=7.5, color=INK_2,
                ha="right" if right else "left", va=va)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)


def plot_run(tag, label, stages, metric, gib, outpath):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    for j, st in enumerate(STAGES):
        d = stages[st]
        rows = d["rows"]
        ep = np.array([r["epoch"] for r in rows])
        tr = np.array([r.get("train_loss", np.nan) for r in rows], dtype=float)
        ev = np.array([r.get("eval_loss", np.nan) for r in rows], dtype=float)
        sd = np.array([r.get("sem_dist", np.nan) for r in rows], dtype=float)
        sr = np.array([r.get("sem_rate", np.nan) for r in rows], dtype=float)

        sel = selected_epoch(rows, metric, gib)
        sel_ep = sel["epoch"]

        # ---- row 1: losses ----
        ax = axes[0, j]
        style_axis(ax)
        ax.plot(ep, tr, color=C_TRAIN, lw=2.0, label="train loss")
        ax.plot(ep, ev, color=C_EVAL, lw=2.0, ls="--", label="eval loss")
        ax.axvline(sel_ep, color=INK_2, lw=1.2, ls=":", zorder=1)
        ax.plot([sel_ep], [sel.get("eval_loss", np.nan)], "o", ms=8,
                mfc=C_EVAL, mec=SURFACE, mew=2, zorder=5)
        anno(ax, f"sel. ep {sel_ep:.0f}\nCE {sel.get('eval_loss', float('nan')):.3f}",
             (sel_ep, sel.get("eval_loss", np.nan)), 14, ep.max())
        ax.set_title(f"{st}  ·  peak LR {d['lr']:.2e}".replace("e-0", "e-"),
                     fontsize=10.5, color=INK, pad=6)
        if j == 0:
            ax.set_ylabel("cross-entropy (nats)", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")

        # ---- row 2: semantics ----
        ax = axes[1, j]
        style_axis(ax)
        ax.plot(ep, sd, color=C_DIST, lw=2.0, label="eval semantic distance (↓)")
        ax.plot(ep, sr, color=C_RATE, lw=2.0, ls="--", label="eval semantic equiv. rate (↑)")
        ax.axvline(sel_ep, color=INK_2, lw=1.2, ls=":", zorder=1)
        ax.plot([sel_ep], [sel.get("sem_dist", np.nan)], "o", ms=8,
                mfc=C_DIST, mec=SURFACE, mew=2, zorder=5)
        ax.plot([sel_ep], [sel.get("sem_rate", np.nan)], "s", ms=8,
                mfc=C_RATE, mec=SURFACE, mew=2, zorder=5)
        # relief rule: direct-label the aqua series at the selection point
        anno(ax, f"d={sel.get('sem_dist', float('nan')):.4f}",
             (sel_ep, sel.get("sem_dist", np.nan)), 12, ep.max())
        anno(ax, f"r={sel.get('sem_rate', float('nan')):.3f}",
             (sel_ep, sel.get("sem_rate", np.nan)), 11, ep.max())

        # the checkpoint an oracle on semantic distance would have kept, when it
        # differs from the one actually restored -- this gap is the selection cost
        best_d = min((r for r in rows if "sem_dist" in r), key=lambda r: r["sem_dist"])
        if best_d["epoch"] != sel_ep:
            ax.plot([best_d["epoch"]], [best_d["sem_dist"]], "o", ms=9,
                    mfc="none", mec=INK_2, mew=1.4, zorder=4)
            anno(ax, f"min d={best_d['sem_dist']:.4f} @ep {best_d['epoch']:.0f}",
                 (best_d["epoch"], best_d["sem_dist"]), 26, ep.max(), va="bottom")
        ax.set_ylim(0, max(np.nanmax(sd), np.nanmax(sr)) * 1.28)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        if j == 0:
            ax.set_ylabel("fraction (unitless)", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")

    fig.suptitle(f"{label}   ·   selection metric: {metric}   ·   patience {PATIENCE}",
                 fontsize=13, color=INK, y=1.03)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return outpath


EVAL_N = {"stage1": 250, "stage2": 500, "stage3": 1000, "stage4": 2000}


def plot_comparison(per_run, outpath):
    """Cross-run comparison at the checkpoint each run actually restored."""
    tags = [t for _, t, *_ in RUNS]
    colors = {tags[0]: C_TRAIN, tags[1]: C_EVAL, tags[2]: C_DIST}
    short = {tags[0]: "run3  CE-stop, decaying LR",
             tags[1]: "run4  CE-stop, const LR",
             tags[2]: "run5  semdist-stop, const LR"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(4)
    w = 0.26

    specs = [
        ("sem_rate", "semantic equivalence rate at selected ckpt  (↑ better)", True),
        ("sem_dist", "semantic distance at selected ckpt  (↓ better)", False),
        ("eval_ce", "eval cross-entropy at selected ckpt  (↓ better)", False),
    ]
    for ax, (key, title, is_rate) in zip(axes, specs):
        style_axis(ax)
        for k, tag in enumerate(tags):
            vals = [per_run[tag][st][key] for st in STAGES]
            off = (k - 1) * w
            err = None
            if is_rate:  # binomial standard error on a finite eval set
                err = [np.sqrt(v * (1 - v) / EVAL_N[st]) for v, st in zip(vals, STAGES)]
            ax.bar(x + off, vals, w - 0.02, color=colors[tag], label=short[tag],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
            if err:
                ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK_2,
                            elinewidth=1.0, capsize=2.5, zorder=4)
            for xi, v in zip(x + off, vals):
                ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=INK_2, rotation=90 if not is_rate else 0)
        ax.set_xticks(x, [s.replace("stage", "stage ") for s in STAGES], fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, pad=6)
        ax.margins(y=0.18)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    axes[0].set_ylabel("fraction (unitless)", fontsize=9, color=INK)
    axes[2].set_ylabel("nats", fontsize=9, color=INK)
    fig.suptitle("Curriculum runs compared at the checkpoint each one restored "
                 "(load_best_model_at_end)", fontsize=12.5, color=INK, y=1.06)
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return outpath


def main():
    table = []
    per_run = {}
    for job, tag, label, fname, metric, gib in RUNS:
        stages = parse_log(MISC / fname)
        p = plot_run(tag, label, stages, metric, gib, OUT / f"curriculum_{tag}.png")
        print(f"wrote {p}")
        per_run[tag] = {}
        for st in STAGES:
            rows = stages[st]["rows"]
            sel = selected_epoch(rows, metric, gib)
            per_run[tag][st] = {
                "sem_rate": sel["sem_rate"], "sem_dist": sel["sem_dist"],
                "eval_ce": sel["eval_loss"],
            }
            table.append((tag, st, stages[st]["lr"], len(rows), sel["epoch"],
                          sel.get("eval_loss"), sel.get("sem_dist"), sel.get("sem_rate"),
                          min(r["sem_dist"] for r in rows if "sem_dist" in r),
                          max(r["sem_rate"] for r in rows if "sem_rate" in r),
                          rows[-1].get("eval_loss")))

    print(f"wrote {plot_comparison(per_run, OUT / 'curriculum_comparison.png')}")

    # table view (required by the relief rule for the aqua series, and useful anyway)
    hdr = ("run", "stage", "lr", "eps", "sel_ep", "sel_CE", "sel_dist", "sel_rate",
           "best_dist", "best_rate", "final_CE")
    print("\n" + " ".join(f"{h:>12}" for h in hdr))
    for r in table:
        cells = [f"{r[0]:>12}", f"{r[1]:>12}", f"{r[2]:>12.2e}", f"{r[3]:>12d}",
                 f"{r[4]:>12.0f}", f"{r[5]:>12.4f}", f"{r[6]:>12.4f}", f"{r[7]:>12.4f}",
                 f"{r[8]:>12.4f}", f"{r[9]:>12.4f}", f"{r[10]:>12.4f}"]
        print(" ".join(cells))


if __name__ == "__main__":
    main()
