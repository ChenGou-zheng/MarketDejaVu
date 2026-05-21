"""
Feature-Strategy Pearson correlation heatmap.

For each feature F_i (i=1..16) and each strategy S_j:
  r_ij = PearsonCorr(F_i(t), excess_S_j(t))  over t = 1..2670

Output: assets/feature_corr_heatmap.png
         assets/feature_corr_table.csv
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
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "processed" / "snapshot"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FEATURE_LABELS = [
    "realized_vol_20d", "momentum_20d", "up_down_ratio",
    "margin_balance", "max_dd_60d", "cpi_yoy",
    "skew_20d", "rsi_14", "pmi_mfg",
    "recovery_days", "yield_slope", "ma5_above_ma20",
    "momentum_60d", "ppi_yoy", "m2_yoy", "net_total",
]

FEATURE_CATEGORIES = {
    "Vol/Risk": ["realized_vol_20d", "max_dd_60d"],
    "Momentum": ["momentum_20d", "momentum_60d"],
    "Technical": ["rsi_14", "ma5_above_ma20", "skew_20d", "up_down_ratio", "recovery_days"],
    "Capital Flow": ["yield_slope", "net_total", "margin_balance"],
    "Macro": ["pmi_mfg", "cpi_yoy", "m2_yoy", "ppi_yoy"],
}


def main():
    features = pd.read_parquet(SNAPSHOT_DIR / "features.parquet")
    labels = pd.read_parquet(SNAPSHOT_DIR / "labels.parquet")

    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    print(f"Features: {features.shape}, Labels: {labels.shape}")
    print(f"Date range: {common[0].date()} ~ {common[-1].date()}")

    feature_cols = [c for c in FEATURE_LABELS if c in features.columns]
    strategy_cols = list(labels.columns)

    # ── compute Pearson r for each feature × each strategy ──
    F = features[feature_cols].values.astype(np.float64)
    L = labels.values.astype(np.float64)

    # mask NaN labels
    mask = np.isfinite(L)

    n_feat = len(feature_cols)
    n_strat = len(strategy_cols)
    r_mat = np.full((n_feat, n_strat), np.nan)
    p_mat = np.full((n_feat, n_strat), np.nan)

    for i in range(n_feat):
        f = F[:, i]
        f_valid = np.isfinite(f)
        for j in range(n_strat):
            both = f_valid & mask[:, j]
            n = both.sum()
            if n < 30:
                continue
            fv = f[both]
            lv = L[both, j]
            # clip labels to [-3, 3] to avoid extreme outliers dominating
            lv = np.clip(lv, -3.0, 3.0)
            if np.std(fv) > 0 and np.std(lv) > 0:
                r = np.corrcoef(fv, lv)[0, 1]
                r_mat[i, j] = r
                # p-value via t-distribution
                import scipy.stats as st
                t_stat = r * np.sqrt((n - 2) / max(1 - r * r, 1e-10))
                p_mat[i, j] = 2 * st.t.sf(abs(t_stat), df=n - 2)

    # ── sort features by mean |r| ──
    mean_abs_r = np.nanmean(np.abs(r_mat), axis=1)
    sort_idx = np.argsort(-mean_abs_r)
    r_sorted = r_mat[sort_idx]
    p_sorted = p_mat[sort_idx]
    feat_sorted = [feature_cols[i] for i in sort_idx]

    print("\nFeature ranking by mean |r|:")
    for rank, (fi, mr) in enumerate(zip(feat_sorted, mean_abs_r[sort_idx]), 1):
        max_r = np.nanmax(np.abs(r_sorted[rank - 1]))
        print(f"  {rank:2d}. {fi:20s}  mean|r|={mr:.4f}  max|r|={max_r:.4f}")

    # ── build category color bar ──
    cat_colors = {
        "Momentum": "#1f77b4",
        "Technical": "#ff7f0e",
        "Vol/Risk": "#2ca02c",
        "Capital Flow": "#d62728",
        "Macro": "#9467bd",
    }
    feat_to_cat = {}
    for cat, flist in FEATURE_CATEGORIES.items():
        for f in flist:
            feat_to_cat[f] = cat

    # ── plot heatmap ──
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.97, 0.03], hspace=0, wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    cmap = LinearSegmentedColormap.from_list(
        "divergent",
        ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
        N=256,
    )

    # clip to [-0.3, 0.3] for visual clarity (values beyond are rare)
    vmax = 0.3
    im = ax.imshow(
        np.clip(r_sorted, -vmax, vmax),
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="none",
    )

    # y-axis: feature names with category color markers
    y_labels = []
    y_colors = []
    for f in feat_sorted:
        cat = feat_to_cat.get(f, "Other")
        display = f.replace("_", " ")
        # add category prefix
        y_labels.append(f"{display}")
        y_colors.append(cat_colors.get(cat, "#999999"))

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, color in zip(ax.get_yticklabels(), y_colors):
        tick.set_color(color)

    # x-axis: strategy keys (shortened)
    short_names = [s.replace("交易记录", "").replace("策略_", "").replace("ETF_长持_", "")
                   for s in strategy_cols]
    ax.set_xticks(range(n_strat))
    ax.set_xticklabels(short_names, fontsize=5, rotation=60, ha="right")

    # statistical significance markers
    for i in range(n_feat):
        for j in range(n_strat):
            pv = p_sorted[i, j]
            rv = r_sorted[i, j]
            if np.isnan(pv) or np.isnan(rv):
                continue
            if pv < 0.01:
                marker = "**"
                color = "white" if abs(rv) > 0.12 else "#333333"
                ax.text(j, i, marker, ha="center", va="center",
                        fontsize=4, color=color, fontweight="bold")
            elif pv < 0.05:
                marker = "*"
                color = "white" if abs(rv) > 0.12 else "#333333"
                ax.text(j, i, marker, ha="center", va="center",
                        fontsize=4, color=color)

    ax.set_xlabel("Strategy", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.set_title(
        "Feature-Strategy Pearson Correlation  $r(F_i, \\text{excess}_{S_j})$",
        fontsize=13, fontweight="bold", pad=10,
    )

    # colorbar
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("Pearson $r$", fontsize=10)

    # annotation box with category legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cat_colors[c], label=c) for c in cat_colors
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=7,
        title="Feature Category",
        title_fontsize=8,
        framealpha=0.85,
    )

    out_png = ASSETS_DIR / "feature_corr_heatmap.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\n[SAVED] {out_png}")
    plt.close(fig)

    # ── save summary table ──
    rows = []
    for rank, fi in enumerate(feat_sorted, 1):
        idx = feature_cols.index(fi)
        mr = mean_abs_r[sort_idx[rank - 1]]
        max_r = np.nanmax(np.abs(r_sorted[rank - 1]))
        cat = feat_to_cat.get(fi, "Other")
        w = "1.5" if cat == "Macro" else ("0.8" if cat == "Technical" else "1.0")
        rows.append({
            "rank": rank,
            "feature": fi,
            "category": cat,
            "mean_abs_r": f"{mr:.4f}",
            "max_abs_r": f"{max_r:.4f}",
            "weight": w,
        })

    summary = pd.DataFrame(rows)
    out_csv = ASSETS_DIR / "feature_corr_table.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
