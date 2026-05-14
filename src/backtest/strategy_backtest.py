"""
策略回溯：计算每个策略在每个 HMM 训练集状态下的表现指标。

输入:
  data/processed/strategy_nav/*_nav.csv    策略日频净值
  data/processed/hmm_states_train.csv      HMM 训练集状态序列
  data/processed/strategy_registry.csv     策略注册表（过滤短覆盖）
  data/processed/aligned_daily.parquet     HS300 基准（计算超额）

输出:
  output/tables/strategy_performance_by_state.csv
  output/tables/state_strategy_recommendation.csv
  output/tables/test_2026_evaluation.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
NAV_DIR = PROCESSED / "strategy_nav"
OUT_TABLES = PROJECT_ROOT / "output" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

MIN_TRADING_DAYS = 200  # 排除交易天数不足的策略
MIN_STATE_DAYS = 50       # 状态内最少天数，低于此阈值不推荐


def load_strategies() -> dict[str, pd.DataFrame]:
    """加载所有策略净值，返回 {策略名: DataFrame(date, nav, daily_return)}。"""

    reg = pd.read_csv(PROCESSED / "strategy_registry.csv")
    valid = reg[reg["trading_days"] >= MIN_TRADING_DAYS].copy()
    valid = valid[~valid["strategy_key"].str.contains("综合全")]

    print(f"Loading {len(valid)} strategies (>= {MIN_TRADING_DAYS} trading days)...")

    # 使用 aligned_daily 的交易日历作为统一基准
    aligned = pd.read_parquet(PROCESSED / "aligned_daily.parquet")
    aligned.index = pd.to_datetime(aligned.index)
    all_dates = aligned.index

    strategies = {}
    for _, row in valid.iterrows():
        key = row["strategy_key"]
        safe = key.replace("/", "_").replace("\\", "_")
        path = NAV_DIR / f"{safe}_nav.csv"
        if not path.exists():
            print(f"  [WARN] {key} nav not found")
            continue
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()

        # 前向填充 NAV 到所有交易日
        nav = df["nav"]
        nav_filled = nav.reindex(all_dates, method="ffill")
        nav_filled = nav_filled.dropna()

        # 从填充后的每日 NAV 计算日收益率
        daily_ret = nav_filled.pct_change(fill_method=None).dropna()
        daily_ret = daily_ret.clip(-0.5, 0.5)  # 剔除极端单日涨跌（停牌/分红可能引起）

        strategies[key] = daily_ret

    print(f"Loaded {len(strategies)} strategies")
    return strategies


def load_hmm_states() -> pd.Series:
    df = pd.read_csv(PROCESSED / "hmm_states_train.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["state"].astype(int)


def load_benchmark() -> pd.Series:
    df = pd.read_parquet(PROCESSED / "aligned_daily.parquet")
    df.index = pd.to_datetime(df.index)
    return df["HS300_ret"]


def main():
    # ── 加载 ──
    strategies = load_strategies()
    hmm_states = load_hmm_states()
    benchmark = load_benchmark()

    state_labels = {
        0: "高波震荡", 1: "持续阴跌", 2: "高波下跌",
        3: "低波上涨", 4: "恐慌暴跌", 5: "慢牛趋势",
    }

    # ── 逐策略计算 ──
    rows = []
    for strategy_key, ret_series in strategies.items():
        # ret_series 是已前向填充后的日收益率 Series
        ret = ret_series.dropna()
        common = ret.index.intersection(hmm_states.index)
        if len(common) < 50:
            continue

        ret_aligned = ret[common]
        state_aligned = hmm_states[common]
        bench_aligned = benchmark.reindex(common).fillna(0)

        for state in sorted(state_aligned.unique()):
            mask = state_aligned == state
            ret_s = ret_aligned[mask]
            bench_s = bench_aligned[mask]
            n = mask.sum()
            if n < MIN_STATE_DAYS:
                continue

            # 段内累计收益（日复利）
            cum_ret = (1 + ret_s).prod() - 1
            # 年化波动率
            ann_vol = ret_s.std() * np.sqrt(252)
            # 最大回撤
            nav_s = (1 + ret_s).cumprod()
            max_dd = (nav_s / nav_s.cummax() - 1).min()
            # Sharpe
            sharpe = ret_s.mean() / ret_s.std() * np.sqrt(252) if ret_s.std() > 0 else 0
            # 超额 vs HS300
            excess = cum_ret - ((1 + bench_s).prod() - 1)
            # 胜率（正收益日占比）
            win_rate = (ret_s > 0).mean()

            rows.append({
                "strategy_key": strategy_key,
                "state": state,
                "state_label": state_labels.get(state, f"State_{state}"),
                "n_days": n,
                "cum_return": cum_ret,
                "ann_vol": ann_vol,
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "excess_return": excess,
                "win_rate": win_rate,
            })

    perf_df = pd.DataFrame(rows)
    perf_df = perf_df.sort_values(["state", "sharpe"], ascending=[True, False])
    perf_df.to_csv(OUT_TABLES / "strategy_performance_by_state.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] strategy_performance_by_state.csv: {len(perf_df)} rows x {len(perf_df.columns)} cols")

    # ── 推荐规则表 ──
    # 对每个 state，按 sharpe 选 top-3 策略
    recommendations = []
    for state in sorted(perf_df["state"].unique()):
        top = perf_df[perf_df["state"] == state].head(3)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            recommendations.append({
                "state": state,
                "state_label": row["state_label"],
                "rank": rank,
                "strategy": row["strategy_key"],
                "sharpe": round(row["sharpe"], 2),
                "cum_return": f"{row['cum_return']:.2%}",
                "excess_return": f"{row['excess_return']:.2%}",
                "win_rate": f"{row['win_rate']:.1%}",
                "n_days": row["n_days"],
            })

    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(OUT_TABLES / "state_strategy_recommendation.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] state_strategy_recommendation.csv: {len(rec_df)} rows")

    # ── 打印推荐 ──
    print("\n=== Strategy Recommendations by HMM State ===")
    for state in sorted(perf_df["state"].unique()):
        top1 = perf_df[perf_df["state"] == state].iloc[0]
        label = state_labels.get(state, f"State_{state}")
        print(f"  [{label}] -> {top1['strategy_key']} (Sharpe={top1['sharpe']:.2f}, cum={top1['cum_return']:.2%}, excess={top1['excess_return']:.2%})")

    # ── 2026 测试集评估 ──
    print("\n=== 2026 Test Set Evaluation ===")
    hmm_test = pd.read_csv(PROCESSED / "hmm_states_test.csv")
    hmm_test["trade_date"] = pd.to_datetime(hmm_test["trade_date"])
    hmm_test = hmm_test.set_index("trade_date")["state"].astype(int)

    test_rows = []
    for strategy_key, ret_series in strategies.items():
        ret = ret_series.dropna()
        common = ret.index.intersection(hmm_test.index)
        if len(common) < 3:
            continue
        ret_aligned = ret[common]
        state_aligned = hmm_test[common]
        bench_aligned = benchmark.reindex(common).fillna(0)

        for state in sorted(state_aligned.unique()):
            mask = state_aligned == state
            ret_s = ret_aligned[mask]
            bench_s = bench_aligned[mask]
            n = mask.sum()
            if n < 2:
                continue
            cum_ret = (1 + ret_s).prod() - 1
            excess = cum_ret - ((1 + bench_s).prod() - 1)
            test_rows.append({
                "strategy_key": strategy_key,
                "state": state,
                "state_label": state_labels.get(state, f"State_{state}"),
                "n_days_2026": n,
                "cum_return_2026": cum_ret,
                "excess_return_2026": excess,
            })

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUT_TABLES / "test_2026_evaluation.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] test_2026_evaluation.csv: {len(test_df)} rows")
    print(f"  2026 test days: {len(hmm_test)}, states: {sorted(hmm_test.unique())}")


if __name__ == "__main__":
    main()
