"""
fig04: KNN 近邻时间分布图
P1 — 展示某个典型决策日 t 的 30 个近邻在历史时间轴上的位置
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
SNAPSHOT = PROJECT_ROOT / "data" / "processed" / "snapshot"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FEATURE_WEIGHTS = {
    "momentum_20d": 1.0, "momentum_60d": 1.0, "rsi_14": 1.0,
    "ma5_above_ma20": 1.0, "skew_20d": 0.8, "up_down_ratio": 0.8,
    "recovery_days": 0.8, "realized_vol_20d": 1.0, "max_dd_60d": 1.0,
    "yield_slope": 1.5, "net_total": 1.0, "margin_balance": 1.0,
    "pmi_mfg": 1.5, "cpi_yoy": 1.5, "m2_yoy": 1.5, "ppi_yoy": 1.5,
}

N = 20
K = 30


def weighted_euclidean(ft: np.ndarray, f_hist: np.ndarray, w: np.ndarray) -> np.ndarray:
    diff = f_hist - ft
    return np.sqrt((diff ** 2 * w).sum(axis=1))


def main():
    features = pd.read_parquet(SNAPSHOT / "features.parquet")
    pos = pd.Series(range(len(features)), index=features.index)

    # pick 3 representative decision dates: 2020-03 (COVID crash), 2021-12 (peak), 2024-09 (recent)
    target_dates = [
        pd.Timestamp("2020-03-16"),
        pd.Timestamp("2021-12-13"),
        pd.Timestamp("2024-09-30"),
    ]
    target_dates = [d for d in target_dates if d in features.index]

    weights = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in features.columns])
    halflife_days = int(3 * 252)
    lambda_decay = np.log(2) / halflife_days

    fig, axes = plt.subplots(len(target_dates), 1, figsize=(11, 6), sharex=True)

    if len(target_dates) == 1:
        axes = [axes]

    for ax, t in zip(axes, target_dates):
        t_pos = pos[t]
        # τ+N < t constraint
        avail_mask = pos < (t_pos - N)
        avail_idx = features.index[avail_mask.values]
        n_avail = len(avail_idx)

        if n_avail == 0:
            ax.text(0.5, 0.5, "No available history", transform=ax.transAxes, ha="center")
            continue

        # standardize using available data
        F_avail = features.loc[avail_idx]
        mu = F_avail.mean()
        sigma = F_avail.std().replace(0, np.nan)
        F_std = ((F_avail - mu) / sigma).values
        f_t = ((features.loc[t] - mu) / sigma).values

        # distance
        dist = weighted_euclidean(f_t, F_std, weights)

        k_actual = min(K, n_avail)
        nearest_idx = np.argpartition(dist, k_actual - 1)[:k_actual]
        nearest_order = dist[nearest_idx].argsort()
        nearest_idx = nearest_idx[nearest_order]

        neigh_dates = avail_idx[nearest_idx]
        neigh_dists = dist[nearest_idx]

        # time decay weights
        pos_neigh = pos[neigh_dates].values
        time_decay_w = np.exp(-lambda_decay * (t_pos - pos_neigh))
        time_decay_w = time_decay_w / time_decay_w.sum() * k_actual

        # plot all available history density
        ax.hist(avail_idx, bins=60, color=COLORS["nonsignificant"], alpha=0.4, label="Available history")

        # scatter neighbors colored by distance
        scatter = ax.scatter(
            neigh_dates,
            np.full(len(neigh_dates), 0),
            c=neigh_dists,
            cmap="YlOrRd",
            s=40 * time_decay_w / time_decay_w.mean(),
            edgecolors="#333333",
            linewidths=0.4,
            zorder=5,
        )
        ax.axvline(x=t, color=COLORS["knn_k3"], linewidth=1.5, linestyle="-", alpha=0.8, label=f"Decision date {t.date()}")
        ax.set_ylabel("Count")
        ax.set_title(f"KNN neighbors for {t.date()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

        # add colorbar
        if ax == axes[-1]:
            cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.15, aspect=40)
            cbar.set_label("Distance", fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("KNN Neighbor Distribution in Historical Timeline", fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = FIG_DIR / "fig04_knn_neighbor_time.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
