"""
fig03: 块超额收益分布直方图
P0 — 验证收益来源: 少数大赢交易日 vs 稳定输出
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import COLORS, FIG_DIR, savefig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES = PROJECT_ROOT / "output" / "tables"


def main():
    k3 = pd.read_csv(TABLES / "backtest_knn_k3.csv", parse_dates=["trade_date"])

    # group by block_id, compute block excess = (1 + daily_excess).prod() - 1
    block_excess = k3.groupby("block_id")["excess_return"].apply(
        lambda x: (1 + x).prod() - 1
    ).dropna()

    block_wr = (block_excess > 0).mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = np.where(block_excess > 0, COLORS["knn_k3"], COLORS["nonsignificant"])
    alpha_vals = np.where(block_excess > 0, 0.8, 0.4)

    ax.bar(
        range(len(block_excess)),
        sorted(block_excess.values),
        color=COLORS["knn_k3"],
        alpha=0.7,
        width=0.85,
        edgecolor="white",
        linewidth=0.3,
    )

    # percentile lines
    p25, p50, p75 = block_excess.quantile([0.25, 0.50, 0.75])
    for p, ls, color in [(p25, "--", "#d62728"), (p50, "-", "#333333"), (p75, "--", "#2ca02c")]:
        ax.axhline(y=p, color=color, linestyle=ls, linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Block Index (sorted by excess return)")
    ax.set_ylabel("Block Excess Return (20-day, log scale)")
    ax.set_yscale("symlog")
    ax.set_title("Distribution of 20-day Block Excess Returns", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # annotation box
    textstr = f"Total blocks: {len(block_excess)}\n"
    textstr += f"Block WR: {block_wr:.1%}\n"
    textstr += f"Mean: {block_excess.mean():+.2%}\n"
    textstr += f"Median: {block_excess.median():+.2%}\n"
    textstr += f"P25 / P75: {p25:+.2%} / {p75:+.2%}\n"
    textstr += f"Max win: {block_excess.max():+.2%}\n"
    textstr += f"Max loss: {block_excess.min():+.2%}"

    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7)
    ax.text(
        0.97,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    fig.tight_layout()
    p = FIG_DIR / "fig03_block_excess_hist.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
