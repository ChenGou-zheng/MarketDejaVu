from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "knn_k3": "#1f77b4",
    "knn_k1": "#ff7f0e",
    "knn_k2": "#2ca02c",
    "knn_k5": "#d62728",
    "h20_baseline": "#9467bd",
    "equal_weight": "#8c564b",
    "best_single": "#e377c2",
    "hs300": "#7f7f7f",
    "significant": "#1f77b4",
    "nonsignificant": "#d3d3d3",
    "fdr_line": "#d62728",
}


def savefig(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=400)
    print(f"  [SAVED] {path}")
    return path


def aligned_date_range(dfs: list[pd.DataFrame], date_col: str = "trade_date"):
    """Find common date range across multiple DataFrames."""
    import pandas as pd
    starts = [pd.to_datetime(d[date_col].min()) for d in dfs]
    ends = [pd.to_datetime(d[date_col].max()) for d in dfs]
    return max(starts), min(ends)
