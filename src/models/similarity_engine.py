"""
Phase 2: Similarity matching engine with statistical testing.

For each decision date t:
1. Enforce τ + N < t constraint (only use fully-historical samples)
2. Standardize features using rolling stats (only data up to t)
3. Weighted Euclidean KNN with macro feature emphasis
4. Time-decay weighting on historical samples (3-year half-life)
5. One-sided t-test + Benjamini-Hochberg FDR for each strategy

Output:
  - output/tables/similarity_decisions.csv   (daily decisions)
  - output/tables/similarity_summary.csv     (summary statistics)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "processed" / "snapshot"
OUT_DIR = PROJECT_ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 20
K = 30
MIN_K = 20
HALFLIFE_YEARS = 3
FDR_Q = 0.1

FEATURE_WEIGHTS = {
    "momentum_20d": 1.0,
    "momentum_60d": 1.0,
    "rsi_14": 1.0,
    "ma5_above_ma20": 1.0,
    "skew_20d": 0.8,
    "up_down_ratio": 0.8,
    "recovery_days": 0.8,
    "realized_vol_20d": 1.0,
    "max_dd_60d": 1.0,
    "yield_slope": 1.5,
    "net_total": 1.0,
    "margin_balance": 1.0,
    "pmi_mfg": 1.5,
    "cpi_yoy": 1.5,
    "m2_yoy": 1.5,
    "ppi_yoy": 1.5,
}


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return np.average(x, weights=w) if w.sum() > 0 else 0.0


def weighted_t_stat(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    mu = np.average(x, weights=w)
    var = np.average((x - mu) ** 2, weights=w)
    if var <= 0 or np.isnan(var):
        return 0.0, 1.0
    se = np.sqrt(var / n)
    t = mu / se if se > 0 else 0.0
    p = 1.0 - norm.cdf(t)
    return t, p


def benjamini_hochberg(p_values: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    threshold = (np.arange(1, m + 1) / m) * q
    passed = sorted_p <= threshold
    if passed.any():
        max_k = np.where(passed)[0].max()
        reject = np.zeros(m, dtype=bool)
        reject[sorted_idx[: max_k + 1]] = True
        return reject
    return np.zeros(m, dtype=bool)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", choices=["euclidean", "mahalanobis"], default="euclidean")
    args = parser.parse_args()
    use_mahalanobis = args.dist == "mahalanobis"

    # — load snapshot —
    features = pd.read_parquet(SNAPSHOT_DIR / "features.parquet")
    labels = pd.read_parquet(SNAPSHOT_DIR / "labels.parquet")

    print(f"Features: {features.shape}, Labels: {labels.shape}  ({features.index[0].date()} ~ {features.index[-1].date()})")
    print(f"Strategies: {list(labels.columns)}")
    print(f"Distance: {'Mahalanobis' if use_mahalanobis else 'Weighted Euclidean'}")

    pos = pd.Series(range(len(features)), index=features.index)
    weights = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in features.columns])
    halflife_days = int(HALFLIFE_YEARS * 252)
    lambda_decay = np.log(2) / halflife_days

    decision_rows = []
    strategy_names = list(labels.columns)
    n_strategies = len(strategy_names)

    total = len(features)
    skip_count = 0

    for i, t in enumerate(features.index):
        # τ + N < t constraint: must have N days before t
        min_pos = N  # need at least N samples before this one
        if pos[t] < min_pos:
            skip_count += 1
            continue

        # Available indices: strictly before pos[t] - N
        avail_mask = pos < (pos[t] - N)
        avail_idx = features.index[avail_mask.values]
        n_avail = len(avail_idx)

        if n_avail < MIN_K:
            skip_count += 1
            continue

        # — standardize using only available data —
        F_avail = features.loc[avail_idx]
        mu = F_avail.mean()
        sigma = F_avail.std().replace(0, np.nan)
        F_std = ((F_avail - mu) / sigma).values  # (n_avail, d)
        f_t = ((features.loc[t] - mu) / sigma).values  # (d,)

        # — distance metric —
        if use_mahalanobis:
            # Mahalanobis: D² = (x-y)ᵀ Σ⁻¹ (x-y)
            cov = np.cov(F_std.T)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
            diff = F_std - f_t
            dist = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))
        else:
            # Weighted Euclidean
            diff = F_std - f_t
            dist = np.sqrt((diff ** 2 * weights).sum(axis=1))

        # — select K nearest —
        if n_avail < K:
            k_actual = n_avail
        else:
            k_actual = K

        if k_actual <= 1:
            nearest_idx = np.array([0])
        else:
            nearest_idx = np.argpartition(dist, k_actual - 1)[:k_actual]
            nearest_order = dist[nearest_idx].argsort()
            nearest_idx = nearest_idx[nearest_order]

        neigh_dates = avail_idx[nearest_idx]
        neigh_dists = dist[nearest_idx]

        # — time decay weights (semi-life 3yr) —
        pos_neigh = pos[neigh_dates].values
        t_pos = pos[t]
        time_decay_w = np.exp(-lambda_decay * (t_pos - pos_neigh))
        time_decay_w = time_decay_w / time_decay_w.sum() * k_actual  # normalize to sum = k_actual

        # — statistical tests per strategy —
        label_values = labels.loc[neigh_dates, strategy_names].values

        strategy_results = []
        for s_idx, s_name in enumerate(strategy_names):
            x = label_values[:, s_idx]
            valid = ~np.isnan(x)
            n_valid = valid.sum()
            if n_valid < 3:
                continue
            x_v = x[valid]
            w_v = time_decay_w[valid]
            w_v = w_v / w_v.sum() * n_valid

            t_stat, p_val = weighted_t_stat(x_v, w_v)
            mu_excess = weighted_mean(x_v, w_v)

            strategy_results.append({
                "strategy": s_name,
                "n_samples": n_valid,
                "mean_excess": mu_excess,
                "t_stat": t_stat,
                "p_value": p_val,
            })

        if not strategy_results:
            continue

        sr_df = pd.DataFrame(strategy_results)

        # — FDR correction —
        p_vals = sr_df["p_value"].values
        reject = benjamini_hochberg(p_vals, q=FDR_Q)
        sr_df["adj_p_value"] = np.minimum(p_vals * len(p_vals) / (np.arange(len(p_vals)) + 1), 1.0)
        sr_df["is_significant"] = reject
        sr_df["decision_date"] = t

        decision_rows.append(sr_df)

        if len(decision_rows) % 200 == 0:
            n_sig = reject.sum()
            print(f"  [{t.date()}]  avail={n_avail:4d}  K={k_actual:2d}  strategies={len(sr_df):2d}  sig={n_sig:2d}")

    # — combine —
    if not decision_rows:
        print("No decisions generated. Check data.")
        return

    decisions = pd.concat(decision_rows, ignore_index=True)
    decisions.to_csv(OUT_DIR / "similarity_decisions.csv", index=False, encoding="utf-8-sig")

    print(f"\n[DONE] similarity_decisions.csv: {len(decisions)} rows")
    print(f"  Date range: {decisions['decision_date'].min()} ~ {decisions['decision_date'].max()}")
    print(f"  Skipped (no history): {skip_count}")

    # — summary —
    sig_df = decisions[decisions["is_significant"]].copy()
    print(f"\n=== SUMMARY ===")
    print(f"  Total decisions: {decisions['decision_date'].nunique()}")
    print(f"  Dates with ≥1 significant strategy: {sig_df['decision_date'].nunique()}")
    print(f"  Total significant strategy-date pairs: {len(sig_df)}")

    if len(sig_df) > 0:
        print(f"\n  Top strategies by significant count:")
        top_sig = sig_df["strategy"].value_counts().head(10)
        for s, c in top_sig.items():
            print(f"    {s}: {c} dates significant")

        print(f"\n  Best mean excess among significant:")
        best = sig_df.groupby("strategy")["mean_excess"].mean().sort_values(ascending=False).head(10)
        for s, v in best.items():
            print(f"    {s}: {v:+.4f}")

    # — per-date summary —
    date_summary = decisions.groupby("decision_date").agg(
        n_strategies=("strategy", "count"),
        n_significant=("is_significant", "sum"),
        mean_t_stat=("t_stat", "mean"),
    ).reset_index()
    date_summary.to_csv(OUT_DIR / "similarity_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] similarity_summary.csv: {len(date_summary)} rows")


if __name__ == "__main__":
    main()
