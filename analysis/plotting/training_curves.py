"""Thesis figure: curriculum training curves of the thesis model.

For each of the four curriculum stages this emits a 2x4 panel:
  row 1 -- train vs. eval cross-entropy (nats)
  row 2 -- eval semantic distance over valid generations, eval semantic equivalence
           rate and eval invalid rate (unitless, [0,1])

Markers:
  * dotted vertical line + filled dot  -> the restored checkpoint (min eval cross-entropy),
    i.e. the epoch actually handed to the next stage.
  * open circle on the distance curve  -> epoch of minimum semantic distance (valid).
  * open square on the rate curve       -> epoch of maximum semantic equivalence rate.

Input: the per-stage ``logs/metrics_history.jsonl`` files of the trained model,
written at full precision by the training run's metrics logger.

Semantic distance over valid generations. The logger records the mean distance
over all targets, with an invalid generation scored as distance 1 (the semantic
callback initialises every distance to 1 and overwrites it only for outputs that
parse), next to the invalid rate. The mean over valid generations therefore
follows exactly, per epoch, as (d - r_inv) / (1 - r_inv); it is the distance
convention of Experiment 1.

Usage (from the repository root, as a module)::

    python -m analysis.plotting.training_curves \
        --model-dir <artifacts>/models/CE/final \
        --output-dir <artifacts>/analysis/figures/training

Outputs: training_curves.png and training_curves.pdf in --output-dir. Every
analysis writes its figures to its own folder under <artifacts>/analysis/figures/
(training, exp1, geometry, exp2).
"""

from __future__ import annotations

import sys

if not __package__:
    sys.exit("Run this as a module from the repository root:  python -m analysis.plotting.training_curves ...")

import argparse
import json
from pathlib import Path

from analysis.plotting.style import (AQUA, BLUE, DARK, INK, INK_2, ORANGE, VIOLET,
                                     YELLOW, label_point, style_axes)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

STAGES = ["stage1", "stage2", "stage3", "stage4"]

# colour roles in this figure
C_TRAIN, C_EVAL, C_DIST, C_RATE, C_INV, RESTORE = BLUE, ORANGE, AQUA, VIOLET, YELLOW, DARK


def valid_only_distance(mean_distance: float, invalid_rate: float) -> float:
    """Mean distance over valid generations, from the logged mean that scores an invalid one as 1."""
    if invalid_rate >= 1.0:
        return float("nan")
    return (mean_distance - invalid_rate) / (1.0 - invalid_rate)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Trained model dir holding stage1..stage4/logs/ (e.g. <artifacts>/models/CE/final).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where training_curves.png/.pdf are written (e.g. <artifacts>/analysis/figures/training).")
    return p.parse_args()


def load_metrics(model_dir: Path) -> dict[str, list[dict]]:
    """Per stage, one row per epoch: train/eval cross-entropy and the two semantic metrics.

    Refuses files that hold more than one run (the logger appends, so a rerun
    into the same output dir would add a second run whose epoch counter restarts).
    """
    out = {}
    for st in STAGES:
        path = model_dir / st / "logs" / "metrics_history.jsonl"
        train, evals = {}, {}
        last = {"train_epoch_end": -1.0, "eval_epoch_end": -1.0}
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                kind = r.get("event_type")
                if kind not in last:
                    continue
                ep = float(r["epoch"])
                if ep <= last[kind]:
                    raise ValueError(f"{path}: epoch counter goes back ({last[kind]} -> {ep}); "
                                     "the file holds more than one run")
                last[kind] = ep
                if kind == "train_epoch_end":
                    train[ep] = r["train_loss_mean"]
                else:
                    evals[ep] = r
        if set(train) != set(evals):
            raise ValueError(f"{path}: train and eval epochs differ")
        out[st] = [{"epoch": ep,
                    "train_loss": train[ep],
                    "eval_loss": evals[ep]["eval_loss"],
                    "sem_dist": valid_only_distance(evals[ep]["eval_semantic_distance"],
                                                    evals[ep]["eval_invalid_rate"]),
                    "sem_rate": evals[ep]["eval_semantic_equivalent_rate"],
                    "inv_rate": evals[ep]["eval_invalid_rate"]}
                   for ep in sorted(train)]
    return out


def main():
    args = parse_args()
    data = load_metrics(args.model_dir)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), constrained_layout=True)

    for j, st in enumerate(STAGES):
        rows = data[st]
        ep = np.array([r["epoch"] for r in rows])
        tr = np.array([r.get("train_loss", np.nan) for r in rows], float)
        ev = np.array([r.get("eval_loss", np.nan) for r in rows], float)
        sd = np.array([r.get("sem_dist", np.nan) for r in rows], float)
        sr = np.array([r.get("sem_rate", np.nan) for r in rows], float)
        ir = np.array([r.get("inv_rate", np.nan) for r in rows], float)

        restored = min((r for r in rows if "eval_loss" in r), key=lambda r: r["eval_loss"])
        best_d = min((r for r in rows if "sem_dist" in r), key=lambda r: r["sem_dist"])
        best_r = max((r for r in rows if "sem_rate" in r), key=lambda r: r["sem_rate"])
        re_ep = restored["epoch"]

        # ---- row 1: cross-entropy ----
        ax = axes[0, j]
        style_axes(ax)
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
        style_axes(ax)
        ax.plot(ep, sd, color=C_DIST, lw=1.8, label="semantic distance (valid)")
        ax.plot(ep, sr, color=C_RATE, lw=1.8, ls="--", label="equivalence rate")
        ax.plot(ep, ir, color=C_INV, lw=1.8, ls="-.", label="invalid rate")
        ax.axvline(re_ep, color=RESTORE, lw=1.0, ls=":", zorder=1)
        top = max(np.nanmax(sd), np.nanmax(sr), np.nanmax(ir))
        ax.set_ylim(0, top * 1.32)
        # direct label for the invalid-rate curve (its colour sits below 3:1 on white),
        # where the curve rises clear of the axis; a flat zero line needs none
        if np.nanmax(ir) >= 0.05:
            i_lab = int(np.nanargmax(ir))
            label_point(ax, "invalid", ep[i_lab], ir[i_lab], ep.max(), 5)
        # small optimum markers, labelled above the marker to clear axis and curves
        ax.plot([best_d["epoch"]], [best_d["sem_dist"]], "o", ms=6, mfc="none",
                mec=C_DIST, mew=1.4, zorder=5)
        ax.plot([best_r["epoch"]], [best_r["sem_rate"]], "s", ms=6, mfc="none",
                mec=C_RATE, mew=1.4, zorder=5)
        # put the label under the marker where the invalid curve runs above it
        dy_d = -8 if best_d["inv_rate"] > best_d["sem_dist"] else 8
        label_point(ax, f"min d, ep {best_d['epoch']:.0f}", best_d["epoch"],
                    best_d["sem_dist"], ep.max(), dy_d)
        label_point(ax, f"max r, ep {best_r['epoch']:.0f}", best_r["epoch"],
                    best_r["sem_rate"], ep.max(), 8)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        if j == 0:
            ax.set_ylabel("fraction (unitless)", fontsize=9, color=INK)

    handles = [
        Line2D([], [], color=C_TRAIN, lw=1.8, label="train cross-entropy"),
        Line2D([], [], color=C_EVAL, lw=1.8, ls="--", label="eval cross-entropy"),
        Line2D([], [], color=C_DIST, lw=1.8, label="semantic distance (valid)"),
        Line2D([], [], color=C_RATE, lw=1.8, ls="--", label="equivalence rate"),
        Line2D([], [], color=C_INV, lw=1.8, ls="-.", label="invalid rate"),
        Line2D([], [], color=RESTORE, lw=1.0, ls=":", marker="o", mfc=C_EVAL,
               mec="white", mew=1.2, ms=7, label="restored checkpoint"),
        Line2D([], [], color="none", marker="o", mfc="none", mec=C_DIST, mew=1.4,
               ms=6, label="min semantic distance"),
        Line2D([], [], color="none", marker="s", mfc="none", mec=C_RATE, mew=1.4,
               ms=6, label="max equivalence rate"),
    ]
    fig.legend(handles=handles, ncol=8, frameon=False, fontsize=8,
               labelcolor=INK_2, loc="lower center", bbox_to_anchor=(0.5, -0.075),
               handletextpad=0.5, columnspacing=1.3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = args.output_dir / "training_curves.pdf"
    png_out = args.output_dir / "training_curves.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_out}")
    print(f"wrote {png_out}")
    # echo the marked epochs for the caption
    for st in STAGES:
        rows = data[st]
        restored = min((r for r in rows if "eval_loss" in r), key=lambda r: r["eval_loss"])
        best_d = min((r for r in rows if "sem_dist" in r), key=lambda r: r["sem_dist"])
        best_r = max((r for r in rows if "sem_rate" in r), key=lambda r: r["sem_rate"])
        print(f"  {st}: restored ep{restored['epoch']:.0f} (CE {restored['eval_loss']:.4f}, "
              f"d_valid {restored['sem_dist']:.4f}, invalid {restored['inv_rate']:.4f}) | "
              f"min d ep{best_d['epoch']:.0f} ({best_d['sem_dist']:.4f}) | "
              f"max r ep{best_r['epoch']:.0f} ({best_r['sem_rate']:.4f}) | "
              f"min invalid after restore {min(r['inv_rate'] for r in rows if r['epoch'] >= restored['epoch']):.4f}")


if __name__ == "__main__":
    main()
