"""
XGBoost 特征重要性分析 + Zigzag 对照组。

输入: data/processed/
输出: output/tables/feature_importance.csv
      output/tables/zigzag_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TABLES = PROJECT_ROOT / "output" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)


def run_xgboost_importance() -> pd.DataFrame:
    """用 XGBoost 对 HMM 状态做分类 → 输出特征重要性"""
    import sys; sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from models.regime_hmm import build_features

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "aligned_daily.parquet")
    df.index = pd.to_datetime(df.index)
    F = build_features(df).iloc[60:].ffill().bfill()

    states_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "hmm_states.csv")
    states_df["trade_date"] = pd.to_datetime(states_df["trade_date"])
    states_map = dict(zip(states_df["trade_date"], states_df["state"]))

    F["state"] = F.index.map(states_map)
    F = F.dropna(subset=["state"])
    y = F.pop("state").astype(int)
    X = F.values

    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    clf.fit(X, y)

    imp = pd.DataFrame({
        "feature": list(F.columns),
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp.to_csv(OUT_TABLES / "feature_importance.csv", index=False, encoding="utf-8-sig")

    print("Feature importance (XGBoost):")
    for _, row in imp.head(15).iterrows():
        bar = "#" * int(row["importance"] * 100)
        print(f"  {row['feature']:25s} {row['importance']:.4f} {bar}")

    return imp


def run_zigzag_baseline() -> pd.DataFrame:
    """Zigzag 分段对照组 — 与 HMM 结果对比"""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "aligned_daily.parquet")
    df.index = pd.to_datetime(df.index)
    close = df["HS300_close"]

    # Zigzag: 波段极值检测
    turns = [0]  # 0=flat, 1=up, -1=down
    peak = trough = close.iloc[0]
    peak_idx = trough_idx = 0

    up_thresh = 0.15
    down_thresh = -0.10
    min_days = 20

    zigzag_states = np.zeros(len(close), dtype=int)
    current_state = 0

    for i in range(1, len(close)):
        p = close.iloc[i]
        if p > peak:
            peak = p
            peak_idx = i
        if p < trough:
            trough = p
            trough_idx = i

        # 检测趋势翻转
        if current_state <= 0 and (peak / trough - 1) > up_thresh and (i - trough_idx) >= min_days:
            current_state = 1
            trough = p
            trough_idx = i
        elif current_state >= 0 and (trough / peak - 1) < down_thresh and (i - peak_idx) >= min_days:
            current_state = -1
            peak = p
            peak_idx = i

        zigzag_states[i] = current_state

    # 对比 HMM
    states_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "hmm_states.csv")
    states_df["trade_date"] = pd.to_datetime(states_df["trade_date"])
    hmm_states = states_df["state"].values

    # 只在对齐日期上对比
    common_dates = states_df[states_df["trade_date"].isin(df.index)]
    if len(common_dates) > 0:
        from sklearn.metrics import adjusted_rand_score
        zz = zigzag_states[df.index.isin(common_dates["trade_date"])]
        hh = common_dates["state"].values
        ari = adjusted_rand_score(zz, hh)
        print(f"\nZigzag vs HMM Adjusted Rand Index: {ari:.4f}")

    zigzag_df = pd.DataFrame({"trade_date": df.index, "zigzag_state": zigzag_states})
    zigzag_df.to_csv(OUT_TABLES / "zigzag_states.csv", index=False, encoding="utf-8-sig")
    print("Zigzag states saved to output/tables/zigzag_states.csv")

    return zigzag_df


if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost Feature Importance")
    print("=" * 60)
    run_xgboost_importance()

    print("\n" + "=" * 60)
    print("Zigzag Baseline")
    print("=" * 60)
    run_zigzag_baseline()
