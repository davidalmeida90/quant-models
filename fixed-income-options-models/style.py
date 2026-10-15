"""Chart look and the charts folder (figures plus facts.json): dark and light panels, one accent palette."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "charts"            # quant-models convention: figures in charts/
RESULTS.mkdir(exist_ok=True)
FACTS = RESULTS / "facts.json"

DARK, INK, GRID = "#0d1117", "#e6edf3", "#21262d"
LIGHT, LIGHT_INK, LIGHT_GRID = "#ffffff", "#1b1f24", "#d9dee4"
CYAN, AMBER, GREEN, VIOLET = "#56d4dd", "#ffa657", "#7ee787", "#d2a8ff"
PINK, INDIGO, TEAL, GRAPE = "#ff7b95", "#4c6ef5", "#0ca678", "#7048e8"
ORANGE = "#e8590c"


def figure(w=6.2, h=3.9, light=False):
    fig, ax = plt.subplots(figsize=(w, h), dpi=170)
    fig.patch.set_facecolor(LIGHT if light else DARK)
    ax.set_facecolor(LIGHT if light else DARK)
    t, g = (LIGHT_INK, LIGHT_GRID) if light else (INK, GRID)
    ax.tick_params(colors=t, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color(g)
    ax.grid(color=g, lw=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    return fig, ax


def labels(ax, x, y, title, light=False):
    t = LIGHT_INK if light else INK
    ax.set_xlabel(x, color=t, fontsize=9.5)
    ax.set_ylabel(y, color=t, fontsize=9.5)
    ax.set_title(title, color=t, fontsize=10.5, fontweight="bold", loc="left")


def legend(ax, light=False, **kw):
    return ax.legend(frameon=False, fontsize=8.3, labelcolor=LIGHT_INK if light else INK, **kw)


def save(fig, name, light=False):
    fig.tight_layout(pad=0.5)
    fig.savefig(RESULTS / name, facecolor=LIGHT if light else DARK)
    plt.close(fig)
    print("   ", name)


def record(key, value):
    """Merge one model's numbers into results/facts.json."""
    f = json.loads(FACTS.read_text()) if FACTS.exists() else {}
    f[key] = value
    FACTS.write_text(json.dumps(f, indent=2, default=float))
