"""
fig05: PCA 降维 + KNN 邻域可视化
P1 — 在 2D 投影中展示特征空间中的近邻分布
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import COLORS, FIG_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT = PROJECT_ROOT / "data" / "processed" / "snapshot"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

N = 20
K = 30

FEATURE_WEIGHTS = {
    "momentum_20d": 1.0, "momentum_60d": 1.0, "rsi_14": 1.0,
    "ma5_above_ma20": 1.0, "skew_20d": 0.8, "up_down_ratio": 0.8,
    "recovery_days": 0.8, "realized_vol_20d": 1.0, "max_dd_60d": 1.0,
    "yield_slope": 1.5, "net_total": 1.0, "margin_balance": 1.0,
    "pmi_mfg": 1.5, "cpi_yoy": 1.5, "m2_yoy": 1.5, "ppi_yoy": 1.5,
}


def main():
    features = pd.read_parquet(SNAPSHOT / "features.parquet")
    pos = pd.Series(range(len(features)), index=features.index)

    # PCA on full feature space
    scaler = StandardScaler()
    F_scaled = scaler.fit_transform(features.values)
    pca = PCA(n_components=2, random_state=42)
    F_pca = pca.fit_transform(F_scaled)

    # pick 3 target dates
    target_dates = [
        pd.Timestamp("2020-03-16"),
        pd.Timestamp("2021-12-13"),
        pd.Timestamp("2024-09-30"),
    ]
    target_dates = [d for d in target_dates if d in features.index]

    fig, axes = plt.subplots(1, len(target_dates), figsize=(14, 4.5))
    if len(target_dates) == 1:
        axes = [axes]

    weights = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in features.columns])

    for ax, t in zip(axes, target_dates):
        t_pos = pos[t]
        avail_mask = pos < (t_pos - N)
        avail_idx = features.index[avail_mask.values]

        # KNN in original space
        F_avail = features.loc[avail_idx]
        mu = F_avail.mean()
        sigma = F_avail.std().replace(0, np.nan)
        F_std = ((F_avail - mu) / sigma).values
        f_t = ((features.loc[t] - mu) / sigma).values

        diff = F_std - f_t
        dist = np.sqrt((diff ** 2 * weights).sum(axis=1))

        k_actual = min(K, len(avail_idx))
        nearest_idx = np.argpartition(dist, k_actual - 1)[:k_actual]
        nearest_order = dist[nearest_idx].argsort()
        nearest_idx = nearest_idx[nearest_order]
        neigh_dates = avail_idx[nearest_idx]
        neigh_dists = dist[nearest_idx]

        # map to PCA space
        pos_map = pd.Series(range(len(features)), index=features.index)
        neigh_pca_pos = [pos_map[d] for d in neigh_dates if d in features.index]

        # plot all history in PCA space
        ax.scatter(
            F_pca[:, 0], F_pca[:, 1],
            c=COLORS["nonsignificant"], alpha=0.25, s=3, label="History"
        )

        # highlight neighbors
        ax.scatter(
            F_pca[neigh_pca_pos, 0], F_pca[neigh_pca_pos, 1],
            c=neigh_dists[:len(neigh_pca_pos)],
            cmap="YlOrRd", s=25, edgecolors="#333333", linewidths=0.4,
            label=f"K={k_actual} neighbors"
        )

        # highlight target
        t_pca_pos = pos_map[t]
        ax.scatter(
            F_pca[t_pca_pos, 0], F_pca[t_pca_pos, 1],
            c="#d62728", s=80, marker="*", edgecolors="#000",
            linewidths=0.6, zorder=10, label=f"Target: {t.date()}"
        )

        var_explained = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0f}% var)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0f}% var)")
        ax.set_title(f"{t.date()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", markerscale=0.8)

    fig.suptitle("PCA Projection of Feature Space with KNN Neighbors", fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = FIG_DIR / "fig05_pca_knn.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
