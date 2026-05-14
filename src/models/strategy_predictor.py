"""
直接预测模型：每个策略训练一个 XGBoost 分类器，从 26 维市场特征预测
该策略未来 20 天内是否跑赢沪深300，输出概率作为置信度。

Train: 2016-2024, Test: 2025-2026
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
NAV_DIR = PROCESSED / "strategy_nav"
OUT_TABLES = PROJECT_ROOT / "output" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

FORWARD_WINDOW = 20
TRAIN_END = "2024-12-31"
MIN_TRADING_DAYS = 200


def load_features() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from models.regime_hmm import build_features
    df = pd.read_parquet(PROCESSED / "aligned_daily.parquet")
    df.index = pd.to_datetime(df.index)
    return build_features(df).iloc[60:].ffill().bfill()


def load_strategies() -> dict[str, pd.Series]:
    reg = pd.read_csv(PROCESSED / "strategy_registry.csv")
    valid = reg[reg["trading_days"] >= MIN_TRADING_DAYS]
    valid = valid[~valid["strategy_key"].str.contains("综合全")]
    aligned = pd.read_parquet(PROCESSED / "aligned_daily.parquet")
    aligned.index = pd.to_datetime(aligned.index)
    all_dates = aligned.index
    strategies = {}
    for _, row in valid.iterrows():
        key = row["strategy_key"]
        safe = key.replace("/", "_").replace("\\", "_")
        path = NAV_DIR / f"{safe}_nav.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path); df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        nav = df["nav"].reindex(all_dates, method="ffill").dropna()
        ret = nav.pct_change(fill_method=None).dropna().clip(-0.5, 0.5)
        strategies[key] = ret
    return strategies


def build_xy(
    features: pd.DataFrame, strategy_rets: pd.Series, bench_rets: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    common = features.index.intersection(strategy_rets.index).intersection(bench_rets.index)
    F = features.loc[common]; R = strategy_rets.loc[common]; B = bench_rets.loc[common]

    X_list, y_list, d_list = [], [], []
    for i in range(len(F) - FORWARD_WINDOW):
        x = F.iloc[i].values
        strat_forward = (1 + R.iloc[i+1 : i+1+FORWARD_WINDOW]).prod() - 1
        bench_forward = (1 + B.iloc[i+1 : i+1+FORWARD_WINDOW]).prod() - 1
        y = 1 if strat_forward > bench_forward else 0
        if not np.isfinite(x).all():
            continue
        X_list.append(x); y_list.append(y); d_list.append(F.index[i])
    return np.array(X_list), np.array(y_list), pd.DatetimeIndex(d_list)


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[XGBClassifier, dict]:
    if y.mean() < 0.01 or y.mean() > 0.99:
        return None, {"error": "no_class_balance"}
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X, y)

    tscv = TimeSeriesSplit(n_splits=5)
    oos_probs = np.zeros(len(y))
    for ti, vi in tscv.split(X):
        if len(ti) < 50 or len(vi) < 10:
            continue
        m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
        m.fit(X[ti], y[ti])
        oos_probs[vi] = m.predict_proba(X[vi])[:, 1]

    # AUC on OOS predictions
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y[oos_probs > 0], oos_probs[oos_probs > 0])
    except Exception:
        auc = 0.5
    accuracy = (np.round(oos_probs) == y).mean()
    return model, {"auc": auc, "accuracy": accuracy, "n": len(y), "pos_ratio": y.mean()}


def main():
    print("Loading...")
    F = load_features()
    strategies = load_strategies()
    bench = strategies.get("交易记录300", None)
    if bench is None:
        bench = pd.read_parquet(PROCESSED / "aligned_daily.parquet")
        bench.index = pd.to_datetime(bench.index)
        bench = bench["HS300_ret"]
    else:
        bench = bench
    print(f"Features: {F.shape}, Strategies: {len(strategies)}")

    # ── 逐策略训练 ──
    all_preds = []
    model_info = []

    for sk, rets in sorted(strategies.items()):
        X_all, y_all, dates_all = build_xy(F, rets, bench)
        if len(X_all) < 100 or y_all.mean() < 0.01 or y_all.mean() > 0.99:
            continue

        train_mask = np.array(dates_all) <= pd.Timestamp(TRAIN_END)
        X_tr = X_all[train_mask]; y_tr = y_all[train_mask]
        X_te = X_all[~train_mask]; y_te = y_all[~train_mask]
        dates_te = dates_all[~train_mask]

        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        model, info = train_model(X_tr, y_tr)
        if model is None:
            continue

        model_info.append({"strategy_key": sk, **info})

        if info["auc"] > 0.51 and len(X_te) > 0:
            probs = model.predict_proba(X_te)[:, 1]
            for i in range(len(dates_te)):
                all_preds.append({
                    "trade_date": dates_te[i],
                    "strategy_key": sk,
                    "prob_beat_benchmark": probs[i],
                    "did_beat": y_te[i],
                    "auc_cv": info["auc"],
                })

    # ── 汇总 ──
    info_df = pd.DataFrame(model_info).sort_values("auc", ascending=False)
    info_df.to_csv(OUT_TABLES / "strategy_model_quality.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] strategy_model_quality.csv: {len(info_df)} strategies")
    print(f"  AUC > 0.5: {(info_df['auc']>0.5).sum()}/{len(info_df)}")
    print(f"  Top by AUC:")
    for _, r in info_df.head(8).iterrows():
        print(f"    {r['strategy_key']:30s} AUC={r['auc']:.4f}  acc={r['accuracy']:.3f}  n={r['n']}")

    # ── 推荐准确率 ──
    preds_df = pd.DataFrame(all_preds)
    if preds_df.empty:
        print("[WARN] No predictions"); return
    preds_df.to_csv(OUT_TABLES / "strategy_predictions_daily.csv", index=False, encoding="utf-8-sig")

    # 按日期评估：选 top-3 概率的策略，看它们实际打败基准的比例
    dates = sorted(preds_df["trade_date"].unique())
    top1_hit, top3_hit = [], []
    for d in dates:
        day = preds_df[preds_df["trade_date"] == d].sort_values("prob_beat_benchmark", ascending=False)
        if len(day) < 3:
            continue
        top1_hit.append(day.iloc[0]["did_beat"])
        top3_hit.append(day.iloc[:3]["did_beat"].any())

    print(f"\n=== Test Set Recommendation Accuracy ({len(dates)} dates) ===")
    print(f"  Top-1 hit rate: {np.mean(top1_hit):.1%}")
    print(f"  Top-3 hit rate: {np.mean(top3_hit):.1%}")
    print(f"  Baseline (random top-1): {preds_df['did_beat'].mean():.1%}")

    # ── 最新推荐 ──
    latest = preds_df[preds_df["trade_date"] == dates[-1]].sort_values("prob_beat_benchmark", ascending=False)
    print(f"\n=== Latest ({dates[-1].date()}) ===")
    for _, r in latest.head(8).iterrows():
        marker = "+" if r["did_beat"] else "-"
        print(f"  [{marker}] {r['strategy_key']:28s} prob={r['prob_beat_benchmark']:.1%}  AUC={r['auc_cv']:.3f}")


if __name__ == "__main__":
    main()

