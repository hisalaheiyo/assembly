"""Shared plotting style for paper figures.

Enforces 6 figure-style requirements:
  1. no titles (captions live in LaTeX)
  2. Times New Roman look (Liberation Serif = metric-compatible TNR replacement)
  3. compact / content-dense
  4. large, legible fonts
  5. pleasant structure & color palette
  6. no overlapping elements (constrained_layout + careful placement)

Usage: `import plotstyle as ps; ps.apply()` then build figures; `ps.save(fig, "F2")`.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from paths import FONTS_DIR as FONT_DIR
from paths import FIGURES_DIR as FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Curated, colorblind-friendly, print-pleasant palette.
PALETTE = {
    "MECCANO": "#4C72B0",   # blue
    "HAViD":   "#DD8452",   # orange
    "IMPACT":  "#55A868",   # green
    "verb":    "#4C72B0",
    "noun":    "#C44E52",   # red
    "both":    "#55A868",   # green (both-correct)
    "neither": "#8C8C8C",   # grey
    "accent":  "#8172B3",   # purple
    "atomic":  "#BFBFBF",
    "shared":  "#4C72B0",
    "indep":   "#55A868",
    "c2c":     "#8172B3",
}
# Stacked error-cell colors (both / verb_only / noun_only / neither)
CELL = {"both":"#55A868", "verb_only":"#DD8452", "noun_only":"#C44E52", "neither":"#B0B0B0"}

def apply():
    for ttf in FONT_DIR.glob("LiberationSerif-*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    plt.rcParams.update({
        "font.family": "serif",
        # Liberation Serif registers under this family name; fall back to DejaVu Serif.
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Serif",
        "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold",
        # large, legible (req. 4)
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13,
        "figure.titlesize": 16,
        # clean structure (req. 5)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
    })

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    return FIG_DIR / f"{name}.pdf"
