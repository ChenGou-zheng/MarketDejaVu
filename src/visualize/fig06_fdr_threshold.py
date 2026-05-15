"""
fig06: FDR 拒绝阈值可视化
P1 — 展示某个决策日 p 值排序 + BH 阈值线, 标注被拒绝的策略
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import COLORS, FIG_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES = PROJECT_ROOT / "output" / "tables"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FDR_Q = 0.1


def main():
    decisions = pd.read_csv(TABLES / "similarity_decisions.csv", parse_dates=["decision_date"])

    # pick 3 representative dates with some significant results
    date_counts = (
        decisions[decisions["is_significant"]]
        .groupby("decision_date")
        .size()
        .sort_values(ascending=False)
    )
    top_dates = date_counts.head(10).index.tolist()

    # pick 3 spread across time
    target_dates = []
    for d in top_dates:
        dt = pd.Timestamp(d)
        if dt.year not in [pd.Timestamp(td).year for td in target_dates]:
            target_dates.append(d)
        if len(target_dates) >= 3:
            break

    fig, axes = plt.subplots(1, len(target_dates), figsize=(14, 4.5))
    if len(target_dates) == 1:
        axes = [axes]

    for ax, date in zip(axes, target_dates):
        dd = decisions[decisions["decision_date"] == date].copy()
        if dd.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        dd = dd.sort_values("p_value").reset_index(drop=True)
        m = len(dd)
        dd["rank"] = np.arange(1, m + 1)
        dd["bh_threshold"] = (dd["rank"] / m) * FDR_Q
        dd["is_rejected"] = dd["p_value"] <= dd["bh_threshold"]

        colors = np.where(dd["is_rejected"], COLORS["significant"], COLORS["nonsignificant"])

        # p-value scatter
        ax.scatter(
            dd["rank"], dd["p_value"],
            c=colors, s=15, alpha=0.8, edgecolors="none", zorder=5
        )

        # BH threshold line
        ax.plot(
            dd["rank"], dd["bh_threshold"],
            color=COLORS["fdr_line"], linewidth=1.2, linestyle="--",
            label=f"BH threshold (q={FDR_Q})", zorder=4
        )

        # reference line at p=0.05
        ax.axhline(y=0.05, color="#555555", linewidth=0.6, linestyle=":", alpha=0.5)

        n_sig = dd["is_rejected"].sum()
        n_total = len(dd)
        ax.set_xlabel("Rank of p-value")
        ax.set_ylabel("p-value")
        ax.set_title(f"{date}\n{n_sig}/{n_total} significant", fontsize=10, fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("FDR Control: Benjamini-Hochberg Procedure", fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = FIG_DIR / "fig06_fdr_threshold.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
