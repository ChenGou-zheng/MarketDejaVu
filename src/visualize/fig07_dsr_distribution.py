"""
fig07: DSR (Deflated Sharpe Ratio) 分布
P1 — 模拟多重比较下的 SR 零分布 vs 实际系统 SR, 验证统计显著性
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

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

N_TRIALS = 10000
SEED = 42


def compute_sr(excess_ret: np.ndarray) -> float:
    T = len(excess_ret)
    if T < 10:
        return 0.0
    sr = excess_ret.mean() / excess_ret.std() * np.sqrt(252) if excess_ret.std() > 0 else 0.0
    return sr


def expected_max_sharpe(m: int, n_trials: int = N_TRIALS, seed: int = SEED) -> float:
    """E[max_Z] for m independent trials under null."""
    gamma = 0.5772156649
    e_max = (1 - gamma) * norm.ppf(1 - 1 / m) + gamma * norm.ppf(1 - 1 / (m * np.e))
    return e_max


def main():
    metrics = pd.read_csv(TABLES / "dynamic_backtest_metrics.csv")
    actual_sr = metrics["sharpe_annual"].iloc[0]
    actual_dsr_z = metrics["deflated_z"].iloc[0]
    actual_dsr_p = metrics["deflated_p_value"].iloc[0]
    n_strategies = int(metrics["n_trials"].iloc[0])
    n_obs = int(metrics["trading_days"].iloc[0])

    rng = np.random.default_rng(SEED)

    # simulate null distribution of max Sharpe
    # under H0: returns are N(0, sigma)
    null_max_srs = []
    for _ in range(N_TRIALS):
        null_rets = rng.standard_t(df=5, size=(n_strategies, n_obs)) * 0.01
        srs = np.array([compute_sr(null_rets[i]) for i in range(n_strategies)])
        null_max_srs.append(srs.max())

    null_max_srs = np.array(null_max_srs)

    # expected max under null
    e_max_z = expected_max_sharpe(n_strategies)
    e_max_sr = e_max_z * (null_max_srs.std())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── panel 1: null distribution of max Sharpe ──
    ax1 = axes[0]
    ax1.hist(null_max_srs, bins=60, color=COLORS["nonsignificant"], alpha=0.6, density=True, label="Null (random)")
    ax1.axvline(x=actual_sr, color=COLORS["knn_k3"], linewidth=1.8, linestyle="-", label=f"Actual SR={actual_sr:.2f}")
    ax1.axvline(x=e_max_sr, color=COLORS["fdr_line"], linewidth=1.2, linestyle="--", label=f"E[max] under null={e_max_sr:.2f}")
    ax1.set_xlabel("Sharpe Ratio (annualized)")
    ax1.set_ylabel("Density")
    ax1.set_title("Null Distribution of Max Sharpe", fontweight="bold")
    ax1.legend(fontsize=8)

    # annotation
    percentile = (null_max_srs < actual_sr).mean()
    textstr = f"Actual SR: {actual_sr:.2f}\nNull max SR (mean): {null_max_srs.mean():.2f}\nActual > null: {percentile:.1%} of simulations"
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7)
    ax1.text(0.97, 0.95, textstr, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right", bbox=props)

    # ── panel 2: DSR Z-statistic ──
    ax2 = axes[1]
    # simulate null DSR Z
    gamma = 0.5772156649
    null_z = np.sort(null_max_srs) / (null_max_srs.std() / np.sqrt(n_obs) + 1e-10)

    ax2.hist(null_z, bins=60, color=COLORS["nonsignificant"], alpha=0.6, density=True, label="Null DSR Z")
    ax2.axvline(x=actual_dsr_z, color=COLORS["knn_k3"], linewidth=1.8, linestyle="-", label=f"Actual DSR Z={actual_dsr_z:.2f}")
    ax2.axvline(x=0, color="#555555", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("DSR Z-statistic")
    ax2.set_ylabel("Density")
    ax2.set_title("Deflated Sharpe Ratio (DSR)", fontweight="bold")
    ax2.legend(fontsize=8)

    textstr2 = f"DSR Z: {actual_dsr_z:.2f}\nDSR p: {actual_dsr_p:.4f}\nSignificant: {'YES' if actual_dsr_p < 0.05 else 'NO'}"
    ax2.text(0.97, 0.95, textstr2, transform=ax2.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right", bbox=props)

    fig.suptitle("Statistical Significance via Deflated Sharpe Ratio", fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = FIG_DIR / "fig07_dsr_distribution.png"
    fig.savefig(p, dpi=400)
    print(f"  [SAVED] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
