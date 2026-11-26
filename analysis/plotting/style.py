"""Shared look for every thesis figure produced under ``analysis/plotting/``.

Importing this module selects the non-interactive Agg backend and applies the
common matplotlib settings, so figure modules import it before drawing.
Colours are named by hue; each figure module assigns them roles.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# palette (dataviz reference instance, print-safe with distinct line styles)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
YELLOW = "#f2a600"   # reference slot 4; 2.05:1 on white, so ship it with a direct label
INK = "#1a1a1a"      # titles and axis labels
INK_2 = "#5c5c5c"    # tick labels, annotations, legend text
GRID = "#e6e6e6"
DARK = "#333333"     # reference lines (e.g. a restored checkpoint)

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


def style_axes(ax) -> None:
    """Light grid behind the data, no top/right spines, muted ticks."""
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.7)


def label_point(ax, text, x, y, xmax, dy) -> None:
    """Annotate (x, y), flipping the label to the left near the right edge of the axis."""
    right = x > 0.6 * xmax
    ax.annotate(text, (x, y), xytext=(-4 if right else 4, dy),
                textcoords="offset points", fontsize=7, color=INK_2,
                ha="right" if right else "left",
                va="bottom" if dy > 0 else "top")
