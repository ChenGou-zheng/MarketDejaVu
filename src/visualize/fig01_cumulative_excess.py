"""
fig01: 累计超额净值曲线
P0 — 核心结论验证

对比: KNN K=3 (QF ON) vs 等权非MTM vs 最佳单一 vs v2.4基线 vs HS300
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import COLORS, FIG_DIR, aligned_date_range, savefig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES = PROJECT_ROOT / "output" / "tables"
SNAPSHOT = PROJECT_ROOT / "data" / "processed" / "snapshot"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def main():
    # ── load backtest data ──
    k3_qf = pd.read_csv(TABLES / "backtest_knn_qf_on.csv", parse_dates=["trade_date"])
    h20 = pd.read_csv(TABLES / "backtest_knn_h20.csv", parse_dates=["trade_date"])
    k3_nof = pd.read_csv(TABLES / "backtest_knn_k3.csv", parse_dates=["trade_date"])

    # ── equal-weight non-MTM baseline ──
    rr = pd.read_parquet(SNAPSHOT / "rebuilt_returns.parquet")
    bc = pd.read_parquet(SNAPSHOT / "benchmark_close.parquet")
    hs300_ret = bc["HS300_close"].pct_change()

    non_mtm = [c for c in rr.columns if not c.startswith("交易记录")]
    common = rr.index.intersection(hs300_ret.index)
    common = common[common >= k3_qf["trade_date"].min()]
    hs300_s = hs300_ret.reindex(common)
    ew = rr[non_mtm].reindex(common).mean(axis=1)
    ew_excess = ew - hs300_s

    # ── best single strategy: 动量轮动 ──
    mom = "交易记录etf动量轮动"
    if mom in rr.columns:
        mom_ret = rr[mom].reindex(common)
        mom_excess = mom_ret - hs300_s
    else:
        mom_excess = pd.Series(index=common, dtype=float)

    # ── compute cumulative excess ──
    def cum_excess(df, col="excess_return"):
        dates = pd.to_datetime(df["trade_date"])
        cum = (1 + df[col].values).cumprod()
        return dates, cum

    fig, ax = plt.subplots(figsize=(10, 5.5))

    lines = []

    d, v = cum_excess(k3_qf)
    (l1,) = ax.plot(d, v, color=COLORS["knn_k3"], linewidth=1.2, label="KNN K=3 (QF ON)")
    lines.append(l1)

    d, v = cum_excess(h20)
    (l2,) = ax.plot(d, v, color=COLORS["h20_baseline"], linewidth=1.0, linestyle="--", label="v2.4 Baseline (H=20)")
    lines.append(l2)

    # equal weight non-MTM
    ew_nav = (1 + ew_excess).cumprod()
    base_date = common[0]
    ew_nav_aligned = ew_nav.reindex(pd.to_datetime(k3_qf["trade_date"]), method="ffill").fillna(1.0)
    (l3,) = ax.plot(
        pd.to_datetime(k3_qf["trade_date"]),
        ew_nav_aligned.values,
        color=COLORS["equal_weight"],
        linewidth=0.8,
        linestyle=":",
        label="Equal-weight Non-MTM",
    )
    lines.append(l3)

    # best single
    mom_nav = (1 + mom_excess).cumprod()
    mom_aligned = mom_nav.reindex(pd.to_datetime(k3_qf["trade_date"]), method="ffill").fillna(1.0)
    (l4,) = ax.plot(
        pd.to_datetime(k3_qf["trade_date"]),
        mom_aligned.values,
        color=COLORS["best_single"],
        linewidth=0.9,
        linestyle="-.",
        label="Momentum (Best Single)",
    )
    lines.append(l4)

    # HS300 (flat at 1.0)
    ax.axhline(y=1.0, color=COLORS["hs300"], linewidth=0.7, linestyle="-", alpha=0.6)
    ax.text(
        pd.to_datetime(k3_qf["trade_date"]).max(),
        1.0,
        " HS300",
        fontsize=8,
        color=COLORS["hs300"],
        va="center",
    )

    fig.legend(
        handles=lines,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.88),
        framealpha=0.85,
        ncol=1,
        fontsize=8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Excess Return (log scale)")
    ax.set_yscale("log")
    ax.set_title("Cumulative Excess Return Comparison", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    fig.tight_layout()
    p = FIG_DIR / "fig01_cumulative_excess.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
