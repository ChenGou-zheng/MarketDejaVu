"""
Visualize v3 vs v2 vs EW comparison.
Red = beat EW (higher ann excess), Green = below EW (lower ann excess).
Darker = larger margin.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Data: ann_excess values
# Structure: {pool: {period: {method: value}}}
data = {
    "MTM-filtered": {
        "T0(2020+)": {"v3 (方案F)": 28.92, "v2 (Rank)": 36.61, "EW": 26.67},
        "T1(2024+)": {"v3 (方案F)": 19.59, "v2 (Rank)": 29.06, "EW": 38.42},
        "T2(2025+)": {"v3 (方案F)": 39.52, "v2 (Rank)": 34.88, "EW": 43.44},
        "T3(2026+)": {"v3 (方案F)": -21.24, "v2 (Rank)": -55.43, "EW": 5.20},
    },
    "MTM-filtered+ETF": {
        "T0(2020+)": {"v3 (方案F)": 39.91, "v2 (Rank)": 13.74, "EW": 20.94},
        "T1(2024+)": {"v3 (方案F)": 19.28, "v2 (Rank)": 10.47, "EW": 29.24},
        "T2(2025+)": {"v3 (方案F)": 38.88, "v2 (Rank)": 13.71, "EW": 31.64},
        "T3(2026+)": {"v3 (方案F)": -21.24, "v2 (Rank)": -39.47, "EW": -0.29},
    },
}

# Also include WR and DD for display
wr_data = {
    "MTM-filtered": {
        "T0(2020+)": {"v3": 50.3, "v2": 57.1, "EW": 62.2},
        "T1(2024+)": {"v3": 50.0, "v2": 55.0, "EW": 70.0},
        "T2(2025+)": {"v3": 55.9, "v2": 61.8, "EW": 76.5},
        "T3(2026+)": {"v3": 50.0, "v2": 37.5, "EW": 50.0},
    },
    "MTM-filtered+ETF": {
        "T0(2020+)": {"v3": 53.9, "v2": 56.4, "EW": 67.1},
        "T1(2024+)": {"v3": 50.0, "v2": 55.0, "EW": 70.0},
        "T2(2025+)": {"v3": 55.9, "v2": 47.1, "EW": 70.6},
        "T3(2026+)": {"v3": 50.0, "v2": 37.5, "EW": 25.0},
    },
}

dd_data = {
    "MTM-filtered": {
        "T0(2020+)": {"v3": 51.70, "v2": 39.56, "EW": 30.22},
        "T1(2024+)": {"v3": 39.47, "v2": 39.56, "EW": 18.95},
        "T2(2025+)": {"v3": 17.50, "v2": 29.60, "EW": 7.87},
        "T3(2026+)": {"v3": 16.59, "v2": 28.75, "EW": 6.26},
    },
    "MTM-filtered+ETF": {
        "T0(2020+)": {"v3": 44.33, "v2": 29.31, "EW": 26.08},
        "T1(2024+)": {"v3": 39.47, "v2": 20.50, "EW": 13.20},
        "T2(2025+)": {"v3": 17.50, "v2": 20.50, "EW": 6.93},
        "T3(2026+)": {"v3": 16.59, "v2": 20.50, "EW": 6.93},
    },
}

methods = ["v3 (方案F)", "v2 (Rank)", "EW"]
periods = ["T0(2020+)", "T1(2024+)", "T2(2025+)", "T3(2026+)"]
period_labels = ["T0\n2020+", "T1\n2024+", "T2\n2025+", "T3\n2026+"]

# Colors: red for beating EW, green for below EW
# Chinese convention: red = good (up), green = bad (down)
REDS = LinearSegmentedColormap.from_list("reds", ["#ffffff", "#d73027"], N=256)
GREENS = LinearSegmentedColormap.from_list("greens", ["#ffffff", "#1a9850"], N=256)
GRAYS = "#f0f0f0"


def color_cell(val, ew_val, vmin=-60, vmax=50):
    """Return (bg_color, text_color) for a cell."""
    if np.isnan(val) or np.isnan(ew_val):
        return GRAYS, "#333333"
    diff = val - ew_val
    # Normalize diff to [0, 1] for color intensity
    max_diff = max(abs(vmin), abs(vmax))
    intensity = min(abs(diff) / max_diff, 1.0)
    if diff >= 0:
        # Red (beat EW) — darker red for larger positive diff
        return REDS(intensity), "#333333" if intensity < 0.5 else "white"
    else:
        # Green (below EW) — darker green for larger negative diff
        return GREENS(intensity), "#333333" if intensity < 0.5 else "white"


def make_pool_chart(pool_name, ax_ann, ax_wr, ax_dd):
    """Draw one pool's comparison on the given axes."""
    pool_data = data[pool_name]
    pool_wr = wr_data[pool_name]
    pool_dd = dd_data[pool_name]

    n_methods = len(methods)
    n_periods = len(periods)
    cell_w = 0.9
    cell_h = 0.85

    for mi, method in enumerate(methods):
        for pi, period in enumerate(periods):
            ann = pool_data[period].get(method, np.nan)
            ew_val = pool_data[period]["EW"]
            wr = pool_wr[period].get(method.replace(" (方案F)", "").replace(" (Rank)", ""), np.nan)
            dd = pool_dd[period].get(method.replace(" (方案F)", "").replace(" (Rank)", ""), np.nan)

            # -- Ann excess heatmap --
            bg, tc = color_cell(ann, ew_val)
            x, y = pi, mi
            ax_ann.add_patch(Rectangle((x - cell_w / 2, y - cell_h / 2),
                                       cell_w, cell_h, facecolor=bg, edgecolor="white", linewidth=1.5))
            label = f"{ann:+.2f}%" if not np.isnan(ann) else "N/A"
            ax_ann.text(x, y, label, ha="center", va="center", fontsize=9,
                        fontweight="bold", color=tc)

            # -- Block WR --
            bg_wr, tc_wr = color_cell(wr, pool_wr[period]["EW"], vmin=-30, vmax=30)
            ax_wr.add_patch(Rectangle((x - cell_w / 2, y - cell_h / 2),
                                      cell_w, cell_h, facecolor=bg_wr, edgecolor="white", linewidth=1.5))
            ax_wr.text(x, y, f"{wr:.1f}%" if not np.isnan(wr) else "N/A",
                       ha="center", va="center", fontsize=9, fontweight="bold", color=tc_wr)

            # -- Max DD --
            # For DD, lower is better. So "beating EW" = DD < EW_DD → red, DD > EW_DD → green
            bg_dd, tc_dd = color_cell(-dd if not np.isnan(dd) else np.nan,
                                       -pool_dd[period]["EW"] if not np.isnan(pool_dd[period]["EW"]) else np.nan,
                                       vmin=-40, vmax=40)
            ax_dd.add_patch(Rectangle((x - cell_w / 2, y - cell_h / 2),
                                      cell_w, cell_h, facecolor=bg_dd, edgecolor="white", linewidth=1.5))
            ax_dd.text(x, y, f"{dd:.2f}%" if not np.isnan(dd) else "N/A",
                       ha="center", va="center", fontsize=9, fontweight="bold", color=tc_dd)

    # Axis formatting
    for ax, title in [(ax_ann, "Ann. Excess"), (ax_wr, "Block WR"), (ax_dd, "Max DD")]:
        ax.set_xlim(-0.6, n_periods - 0.4)
        ax.set_ylim(-0.6, n_methods - 0.4)
        ax.set_xticks(range(n_periods))
        ax.set_xticklabels(period_labels, fontsize=9)
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels([m.replace(" (方案F)", "\n(Score)").replace(" (Rank)", "\n(Rank)")
                            for m in methods], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.invert_yaxis()
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)


def main():
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25,
                          height_ratios=[1, 1])

    # Row 0: MTM-filtered
    ax_ann_0 = fig.add_subplot(gs[0, 0])
    ax_wr_0 = fig.add_subplot(gs[0, 1])
    ax_dd_0 = fig.add_subplot(gs[0, 2])
    make_pool_chart("MTM-filtered", ax_ann_0, ax_wr_0, ax_dd_0)
    ax_ann_0.set_ylabel("MTM-filtered", fontsize=12, fontweight="bold", labelpad=10)

    # Row 1: MTM-filtered+ETF
    ax_ann_1 = fig.add_subplot(gs[1, 0])
    ax_wr_1 = fig.add_subplot(gs[1, 1])
    ax_dd_1 = fig.add_subplot(gs[1, 2])
    make_pool_chart("MTM-filtered+ETF", ax_ann_1, ax_wr_1, ax_dd_1)
    ax_ann_1.set_ylabel("MTM-filtered\n+ETF", fontsize=12, fontweight="bold", labelpad=10)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", label="Above EW (better)"),
        Patch(facecolor="#1a9850", label="Below EW (worse)"),
        Patch(facecolor="#ffffff", edgecolor="#cccccc", label="Same as EW"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("MTM-filtered: v3 (Score) vs v2 (Rank) vs EW — Heatmap Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    p = FIG_DIR / "ablation_v3_vs_v2_vs_EW.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
