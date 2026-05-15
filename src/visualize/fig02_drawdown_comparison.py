"""
fig02: 回撤对比图 (K=1 vs K=3 vs K=5)
P0 — Top-K 分散效果验证
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


def compute_drawdown(nav: pd.Series) -> pd.Series:
    return 1 - nav / nav.cummax()


def main():
    # ── load ──
    k1 = pd.read_csv(TABLES / "backtest_knn_k1.csv", parse_dates=["trade_date"])
    k2 = pd.read_csv(TABLES / "backtest_knn_k2.csv", parse_dates=["trade_date"])
    k3 = pd.read_csv(TABLES / "backtest_knn_k3.csv", parse_dates=["trade_date"])
    k5 = pd.read_csv(TABLES / "backtest_knn_k5.csv", parse_dates=["trade_date"])

    dd_k1 = compute_drawdown(k1["portfolio_nav"])
    dd_k2 = compute_drawdown(k2["portfolio_nav"])
    dd_k3 = compute_drawdown(k3["portfolio_nav"])
    dd_k5 = compute_drawdown(k5["portfolio_nav"])

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

    configs = [
        ("K=1", dd_k1, k1["trade_date"], COLORS["knn_k1"], f"Max DD: {dd_k1.max():.1%}"),
        ("K=2", dd_k2, k2["trade_date"], COLORS["knn_k2"], f"Max DD: {dd_k2.max():.1%}"),
        ("K=3", dd_k3, k3["trade_date"], COLORS["knn_k3"], f"Max DD: {dd_k3.max():.1%}"),
        ("K=5", dd_k5, k5["trade_date"], COLORS["knn_k5"], f"Max DD: {dd_k5.max():.1%}"),
    ]

    for ax, (label, dd, dates, color, note) in zip(axes, configs):
        ax.fill_between(dates, 0, dd.values, color=color, alpha=0.35, linewidth=0)
        ax.plot(dates, dd.values, color=color, linewidth=0.6)
        ax.axhline(y=dd.max(), color=color, linestyle="--", linewidth=0.6, alpha=0.5)
        y_max = dd.max()
        ax.text(
            dates.iloc[int(len(dates) * 0.85)],
            y_max * 0.55,
            note,
            fontsize=8,
            color=color,
            fontweight="bold",
        )
        ax.set_ylabel("Drawdown")
        ax.set_ylim(0, max(d.max() * 1.25 for d in [dd_k1, dd_k2, dd_k3, dd_k5]))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title(label, loc="left", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Date")
    fig.suptitle("Drawdown Comparison by Top-K", fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    p = FIG_DIR / "fig02_drawdown_comparison.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
