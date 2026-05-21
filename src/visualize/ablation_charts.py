"""
Ablation result charts: 5 academic-standard figures for presentation.
Data from output/tables/ablation_results/v2/v3 and baselines.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

POOLS = ["MTM-unfiltered", "MTM-filtered", "MTM-unfiltered+ETF", "MTM-filtered+ETF"]
METHODS = ["v1 (FDR+N=20)", "v2 (Rank+N=10)", "v3 (Score+N=10)", "EW (baseline)"]
PERIODS = ["T0(2020+)", "T1(2024+)", "T2(2025+)", "T3(2026+)"]
PERIOD_SHORT = ["T0\n2020+", "T1\n2024+", "T2\n2025+", "T3\n2026+"]

COLORS = {
    "v1 (FDR+N=20)": "#95a5a6",
    "v2 (Rank+N=10)": "#2980b9",
    "v3 (Score+N=10)": "#27ae60",
    "EW (baseline)": "#e67e22",
}

POOL_COLORS = {
    "MTM-unfiltered": "#3498db",
    "MTM-filtered": "#e74c3c",
    "MTM-unfiltered+ETF": "#2ecc71",
    "MTM-filtered+ETF": "#9b59b6",
}


def load_data():
    v1 = pd.read_csv(TABLES_DIR / "ablation_results.csv")
    v2 = pd.read_csv(TABLES_DIR / "ablation_v2_results.csv")
    v3 = pd.read_csv(TABLES_DIR / "ablation_v3_results.csv")
    ew_raw = pd.read_csv(TABLES_DIR / "ablation_baselines.csv")

    ew = {}
    for _, row in ew_raw.iterrows():
        pool = row["pool"].replace("Equal-weight (", "").rstrip(")")
        ew[pool] = {}
        for p in PERIODS:
            ew[pool][p] = {
                "ann_excess": row[f"{p}_ann_excess"],
                "block_wr": row[f"{p}_block_wr"],
                "max_dd": row[f"{p}_max_dd"],
            }

    def _get(df, pool, period):
        r = df[(df["pool"] == pool) & (df["period"] == period)]
        if r.empty:
            return None
        return r.iloc[0]

    # Build matrix: {method: {pool: {period: {ann, bw, dd}}}}
    data = {}
    for method, df in [("v1 (FDR+N=20)", v1), ("v2 (Rank+N=10)", v2), ("v3 (Score+N=10)", v3)]:
        data[method] = {}
        for pool in POOLS:
            data[method][pool] = {}
            for p in PERIODS:
                r = _get(df, pool, p)
                if r is not None:
                    data[method][pool][p] = {
                        "ann": parse_pct(r["ann_excess"]),
                        "bw": parse_pct(r["block_wr"]),
                        "dd": parse_pct(r["max_dd"]),
                    }
    data["EW (baseline)"] = ew
    return data


def parse_pct(s: str) -> float:
    if isinstance(s, str):
        return float(s.replace("%", "").replace("+", "").replace("−", "-").strip()) / 100
    return float(s)


def fig1_heatmap_T0(data):
    """4×4 heatmap: pools × methods, color = T0 ann excess."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    matrix = np.full((len(METHODS), len(POOLS)), np.nan)
    for i, method in enumerate(METHODS):
        for j, pool in enumerate(POOLS):
            d = data[method].get(pool, {}).get("T0(2020+)", {})
            matrix[i, j] = d.get("ann", np.nan)

    cmap = LinearSegmentedColormap.from_list(
        "rd_gn", ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"],
        N=256,
    )
    vmax = max(abs(np.nanmax(matrix)), abs(np.nanmin(matrix))) if not np.all(np.isnan(matrix)) else 1.0
    vmax = max(vmax, 0.5)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(POOLS)))
    ax.set_xticklabels([p.replace("MTM-", "").replace("+ETF", "\n+ETF") for p in POOLS], fontsize=9)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([m.replace(" (", "\n(") for m in METHODS], fontsize=8)
    ax.set_xlabel("Strategy Pool", fontsize=10)
    ax.set_ylabel("Method", fontsize=10)
    ax.set_title("T0 Annualized Excess Return by Pool × Method", fontsize=12, fontweight="bold")

    for i in range(len(METHODS)):
        for j in range(len(POOLS)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt = f"{v:+.0%}"
                color = "white" if abs(v) > vmax * 0.6 else "#333333"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Annualized Excess Return", fontsize=9)

    fig.tight_layout()
    p = FIG_DIR / "ablation_heatmap_T0.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  [SAVED] {p}")
    plt.close(fig)


def fig2_bar_T0(data):
    """Grouped bar chart: T0 ann excess per pool, 4 method bars per pool."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    n_pools = len(POOLS)
    n_methods = len(METHODS)
    x = np.arange(n_pools)
    width = 0.18

    for i, method in enumerate(METHODS):
        vals = []
        for pool in POOLS:
            d = data[method].get(pool, {}).get("T0(2020+)", {})
            vals.append(d.get("ann", 0))
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=COLORS[method], edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            if abs(v) > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.02 if v >= 0 else -0.08),
                        f"{v:+.0%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("MTM-", "").replace("+ETF", "\n+ETF") for p in POOLS], fontsize=9)
    ax.set_ylabel("Annualized Excess Return (T0)", fontsize=10)
    ax.set_title("T0 Annualized Excess Return Comparison", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="#333333", linewidth=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "ablation_bar_T0.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  [SAVED] {p}")
    plt.close(fig)


def fig3_blockwr_ann(data):
    """Scatter: Block WR vs Ann Excess for all (pool, method) combos at T0."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for method in METHODS:
        xs, ys = [], []
        for pool in POOLS:
            d = data[method].get(pool, {}).get("T0(2020+)", {})
            bw = d.get("bw")
            ann = d.get("ann")
            if bw is not None and ann is not None and not (np.isnan(bw) or np.isnan(ann)):
                xs.append(bw)
                ys.append(ann)
        ax.scatter(xs, ys, s=60, c=COLORS[method], label=method, edgecolors="#333", linewidths=0.3, zorder=5)
        for pool, x, y in zip(POOLS, xs, ys):
            label = pool.replace("MTM-", "").replace("+ETF", "+E")
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6, alpha=0.7)

    ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax.axvline(x=0.5, color="#999", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Block Win Rate (T0)", fontsize=10)
    ax.set_ylabel("Annualized Excess Return (T0)", fontsize=10)
    ax.set_title("Block WR vs. Ann Excess — T0", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "ablation_blockwr_vs_ann.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  [SAVED] {p}")
    plt.close(fig)


def fig4_risk_return(data):
    """Scatter: Max DD vs Ann Excess. Best = upper-left quadrant (high ann, low DD)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for method in METHODS:
        xs, ys = [], []
        for pool in POOLS:
            d = data[method].get(pool, {}).get("T0(2020+)", {})
            dd = d.get("dd")
            ann = d.get("ann")
            if dd is not None and ann is not None and not (np.isnan(dd) or np.isnan(ann)):
                xs.append(dd)
                ys.append(ann)
        ax.scatter(xs, ys, s=60, c=COLORS[method], label=method, edgecolors="#333", linewidths=0.3, zorder=5)
        for pool, x, y in zip(POOLS, xs, ys):
            label = pool.replace("MTM-", "").replace("+ETF", "+E")
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6, alpha=0.7)

    ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Max Drawdown (T0)", fontsize=10)
    ax.set_ylabel("Annualized Excess Return (T0)", fontsize=10)
    ax.set_title("Risk-Return: Ann Excess vs. Max DD — T0", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "ablation_risk_return.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  [SAVED] {p}")
    plt.close(fig)


def fig5_timeseries(data):
    """Line chart across T0-T3 for each pool's best method."""
    best_methods = {
        "MTM-unfiltered": "v2 (Rank+N=10)",
        "MTM-filtered": "v2 (Rank+N=10)",
        "MTM-unfiltered+ETF": "v3 (Score+N=10)",
        "MTM-filtered+ETF": "v3 (Score+N=10)",
    }
    best_colors = {
        "MTM-unfiltered": "#3498db",
        "MTM-filtered": "#e74c3c",
        "MTM-unfiltered+ETF": "#2ecc71",
        "MTM-filtered+ETF": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: ann excess
    ax1 = axes[0]
    for pool in POOLS:
        method = best_methods[pool]
        vals = []
        for p in PERIODS:
            d = data[method].get(pool, {}).get(p, {})
            vals.append(d.get("ann", np.nan))
        ax1.plot(range(len(PERIODS)), vals, "-o", color=best_colors[pool], linewidth=1.5,
                 markersize=6, label=pool.replace("MTM-", "").replace("+ETF", "+ETF"))
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax1.text(i, v, f"{v:+.0%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6)

    ax1.set_xticks(range(len(PERIODS)))
    ax1.set_xticklabels(PERIOD_SHORT, fontsize=8)
    ax1.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Annualized Excess Return", fontsize=10)
    ax1.set_title("Best Method Per Pool — Ann Excess Across Time", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=6, loc="upper right")

    # Right: block WR
    ax2 = axes[1]
    for pool in POOLS:
        method = best_methods[pool]
        vals = []
        for p in PERIODS:
            d = data[method].get(pool, {}).get(p, {})
            vals.append(d.get("bw", np.nan))
        ax2.plot(range(len(PERIODS)), vals, "-s", color=best_colors[pool], linewidth=1.5,
                 markersize=6, label=pool.replace("MTM-", "").replace("+ETF", "+ETF"))
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax2.text(i, v, f"{v:.0%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6)

    ax2.set_xticks(range(len(PERIODS)))
    ax2.set_xticklabels(PERIOD_SHORT, fontsize=8)
    ax2.axhline(y=0.5, color="#999", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Block Win Rate", fontsize=10)
    ax2.set_title("Best Method Per Pool — Block WR Across Time", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=6, loc="lower left")

    fig.suptitle("Robustness Across Time Windows (T0–T3)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    p = FIG_DIR / "ablation_timeseries.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  [SAVED] {p}")
    plt.close(fig)


def main():
    print("Loading ablation data...")
    data = load_data()

    print("Figure 1: Heatmap T0...")
    fig1_heatmap_T0(data)

    print("Figure 2: Bar chart T0...")
    fig2_bar_T0(data)

    print("Figure 3: Block WR vs Ann...")
    fig3_blockwr_ann(data)

    print("Figure 4: Risk-Return scatter...")
    fig4_risk_return(data)

    print("Figure 5: Time series...")
    fig5_timeseries(data)

    print("\n[DONE] 5 figures saved to output/figures/")


if __name__ == "__main__":
    main()
