#!/usr/bin/env python3
"""Thesis figure: curriculum pretraining curves for the selected base model (run4, constant LR).

For each of the four curriculum stages this emits a 2x4 panel:
  row 1 -- train vs. eval cross-entropy (nats)
  row 2 -- eval semantic distance and eval semantic equivalence rate (unitless, [0,1])

Markers:
  * dotted vertical line + filled dot  -> the restored checkpoint (min eval cross-entropy),
    i.e. the epoch actually handed to the next stage / to finetuning.
  * open circle on the distance curve  -> epoch of minimum semantic distance.
  * open square on the rate curve       -> epoch of maximum semantic equivalence rate.

The offset between the restored epoch and the two semantic optima is the semantic
headroom left open for reinforcement finetuning.

Usage:  python scripts/plot_pretraining_curves.py
Outputs: thesis/Figures/pretraining_curves.pdf  and  misc/figures/pretraining_curves.png
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "misc" / "kernelltl_curriculum_24643524.out"   # run4: constant LR, eval_loss selection
PDF_OUT = ROOT / "thesis" / "Figures" / "pretraining_curves.pdf"
PNG_OUT = ROOT / "misc" / "figures" / "pretraining_curves.png"

STAGES = ["stage1", "stage2", "stage3", "stage4"]

# palette (dataviz reference instance, print-safe with distinct line styles)
C_TRAIN = "#2a78d6"   # blue
C_EVAL = "#eb6834"    # orange
C_DIST = "#1baf7a"    # aqua
C_RATE = "#4a3aa7"    # violet
INK = "#1a1a1a"
INK_2 = "#5c5c5c"
GRID = "#e6e6e6"
RESTORE = "#333333"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#b0b0b0",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

RE_STAGE = re.compile(r"^Starting (stage\d) \(Run")
RE_EVALHDR = re.compile(r"^\s*Eval @ epoch ([\d.]+) / step (\d+):")
RE_SEMDIST = re.compile(r"^\s*eval_semantic_distance:\s*([\d.]+)")
RE_SEMRATE = re.compile(r"^\s*eval_semantic_equivalent_rate:\s*([\d.]+)")


def parse_log(path: Path):
    stage, pending, rows = None, None, {}
    for line in path.read_text().splitlines():
        if m := RE_STAGE.match(line):
            stage = m.group(1)
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
            r = rows.setdefault((stage, float(d["epoch"])), {"epoch": float(d["epoch"])})
            if "loss" in d:
                r["train_loss"] = d["loss"]
            if "eval_loss" in d:
                r["eval_loss"] = d["eval_loss"]
            continue
        if m := RE_EVALHDR.match(line):
            pending = float(m.group(1))
            rows.setdefault((stage, pending), {"epoch": pending})
        elif m := RE_SEMDIST.match(line):
            rows[(stage, pending)]["sem_dist"] = float(m.group(1))
        elif m := RE_SEMRATE.match(line):
            rows[(stage, pending)]["sem_rate"] = float(m.group(1))
    out = {}
    for st in STAGES:
        out[st] = sorted((v for k, v in rows.items() if k[0] == st), key=lambda r: r["epoch"])
    return out


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.7)


def label_point(ax, text, x, y, xmax, dy):
    right = x > 0.6 * xmax
    ax.annotate(text, (x, y), xytext=(-4 if right else 4, dy),
                textcoords="offset points", fontsize=7, color=INK_2,
                ha="right" if right else "left",
                va="bottom" if dy > 0 else "top")


def main():
    data = parse_log(LOG)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), constrained_layout=True)

    for j, st in enumerate(STAGES):
        rows = data[st]
        ep = np.array([r["epoch"] for r in rows])
        tr = np.array([r.get("train_loss", np.nan) for r in rows], float)
        ev = np.array([r.get("eval_loss", np.nan) for r in rows], float)
        sd = np.array([r.get("sem_dist", np.nan) for r in rows], float)
        sr = np.array([r.get("sem_rate", np.nan) for r in rows], float)

        restored = min((r for r in rows if "eval_loss" in r), key=lambda r: r["eval_loss"])
        best_d = min((r for r in rows if "sem_dist" in r), key=lambda r: r["sem_dist"])
        best_r = max((r for r in rows if "sem_rate" in r), key=lambda r: r["sem_rate"])
        re_ep = restored["epoch"]

        # ---- row 1: cross-entropy ----
        ax = axes[0, j]
        style(ax)
        ax.plot(ep, tr, color=C_TRAIN, lw=1.8, label="train")
        ax.plot(ep, ev, color=C_EVAL, lw=1.8, ls="--", label="eval")
        ax.axvline(re_ep, color=RESTORE, lw=1.0, ls=":", zorder=1)
        ax.plot([re_ep], [restored["eval_loss"]], "o", ms=7, mfc=C_EVAL,
                mec="white", mew=1.5, zorder=5)
        label_point(ax, f"restored\nep {re_ep:.0f}", re_ep, restored["eval_loss"], ep.max(), 9)
        ax.set_title(f"Stage {j+1}", fontsize=11, color=INK, pad=5)
        if j == 0:
            ax.set_ylabel("cross-entropy (nats)", fontsize=9, color=INK)

        # ---- row 2: semantic metrics ----
        ax = axes[1, j]
        style(ax)
        ax.plot(ep, sd, color=C_DIST, lw=1.8, label="semantic distance")
        ax.plot(ep, sr, color=C_RATE, lw=1.8, ls="--", label="equivalence rate")
        ax.axvline(re_ep, color=RESTORE, lw=1.0, ls=":", zorder=1)
        top = max(np.nanmax(sd), np.nanmax(sr))
        ax.set_ylim(0, top * 1.32)
        # small optimum markers, labelled above the marker to clear axis and curves
        ax.plot([best_d["epoch"]], [best_d["sem_dist"]], "o", ms=6, mfc="none",
                mec=C_DIST, mew=1.4, zorder=5)
        ax.plot([best_r["epoch"]], [best_r["sem_rate"]], "s", ms=6, mfc="none",
                mec=C_RATE, mew=1.4, zorder=5)
        label_point(ax, f"min d, ep {best_d['epoch']:.0f}", best_d["epoch"],
                    best_d["sem_dist"], ep.max(), 8)
        label_point(ax, f"max r, ep {best_r['epoch']:.0f}", best_r["epoch"],
                    best_r["sem_rate"], ep.max(), 8)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        if j == 0:
            ax.set_ylabel("fraction (unitless)", fontsize=9, color=INK)

    handles = [
        Line2D([], [], color=C_TRAIN, lw=1.8, label="train cross-entropy"),
        Line2D([], [], color=C_EVAL, lw=1.8, ls="--", label="eval cross-entropy"),
        Line2D([], [], color=C_DIST, lw=1.8, label="semantic distance"),
        Line2D([], [], color=C_RATE, lw=1.8, ls="--", label="equivalence rate"),
        Line2D([], [], color=RESTORE, lw=1.0, ls=":", marker="o", mfc=C_EVAL,
               mec="white", mew=1.2, ms=7, label="restored checkpoint"),
        Line2D([], [], color="none", marker="o", mfc="none", mec=C_DIST, mew=1.4,
               ms=6, label="min semantic distance"),
        Line2D([], [], color="none", marker="s", mfc="none", mec=C_RATE, mew=1.4,
               ms=6, label="max equivalence rate"),
    ]
    fig.legend(handles=handles, ncol=7, frameon=False, fontsize=8,
               labelcolor=INK_2, loc="lower center", bbox_to_anchor=(0.5, -0.075),
               handletextpad=0.5, columnspacing=1.3)

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PDF_OUT}")
    print(f"wrote {PNG_OUT}")
    # echo the marked epochs for the caption
    for st in STAGES:
        rows = data[st]
        restored = min((r for r in rows if "eval_loss" in r), key=lambda r: r["eval_loss"])
        best_d = min((r for r in rows if "sem_dist" in r), key=lambda r: r["sem_dist"])
        best_r = max((r for r in rows if "sem_rate" in r), key=lambda r: r["sem_rate"])
        print(f"  {st}: restored ep{restored['epoch']:.0f} (CE {restored['eval_loss']:.4f}) | "
              f"min d ep{best_d['epoch']:.0f} ({best_d['sem_dist']:.4f}) | "
              f"max r ep{best_r['epoch']:.0f} ({best_r['sem_rate']:.4f})")


if __name__ == "__main__":
    main()
