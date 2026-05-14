"""
从 data/processed/ 下所有交易记录 CSV 重建策略日频净值序列。

输出：
  - data/processed/strategy_nav/{策略名}_nav.csv  每日净值+收益率
  - data/processed/strategy_registry.csv          策略注册表
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NAV_OUT_DIR = PROCESSED_DIR / "strategy_nav"
REGISTRY_PATH = PROCESSED_DIR / "strategy_registry.csv"


def collect_trade_records() -> dict[str, list[Path]]:
    """遍历 processed 目录，收集所有含 cash_balance+posi_balance 的 CSV。

    Returns:
        {strategy_key: [csv_path, ...]}  -- 同一策略可能在不同源文件中出现
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    for root, dirs, files in os.walk(PROCESSED_DIR):
        # 跳过输出子目录
        if os.path.basename(root) in ("strategy_nav",):
            continue
        for f in files:
            if not f.endswith(".csv"):
                continue
            path = Path(root) / f
            try:
                df = pd.read_csv(path, nrows=2)
                cols = list(df.columns)
                if "cash_balance" not in cols or "posi_balance" not in cols:
                    continue
                strategy_key = f.replace(".csv", "")
                groups[strategy_key].append(path)
            except Exception:
                continue

    return dict(groups)


def build_daily_nav(csv_paths: list[Path]) -> pd.DataFrame | None:
    """合并同一策略的多个 CSV，重建日频净值。

    Returns:
        DataFrame columns: [trade_date, nav, daily_return]
        若数据异常返回 None
    """
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["trade_time"] = pd.to_datetime(df["trade_time"])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("trade_time").reset_index(drop=True)

    # 按交易日取最后一条记录
    df["trade_date"] = df["trade_time"].dt.date
    daily = df.groupby("trade_date").last().reset_index()

    # 计算日频总资产 = 现金 + 持仓市值
    daily["nav"] = daily["cash_balance"].astype(float) + daily["posi_balance"].astype(float)

    # 日收益率（银证转入日不计算，标记为 NaN）
    daily["daily_return"] = daily["nav"].pct_change()

    # 基本校验
    if daily["nav"].isna().all() or len(daily) < 5:
        return None

    result = daily[["trade_date", "nav", "daily_return"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result


def main():
    NAV_OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = collect_trade_records()
    print(f"Found {len(groups)} unique strategy keys from CSVs\n")

    registry_rows = []
    errors = []

    for strategy_key, csv_paths in sorted(groups.items()):
        nav_df = build_daily_nav(csv_paths)
        if nav_df is None:
            errors.append(strategy_key)
            continue

        # 保存净值 CSV
        safe_name = strategy_key.replace("/", "_").replace("\\", "_")
        out_path = NAV_OUT_DIR / f"{safe_name}_nav.csv"
        nav_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # 注册表信息
        registry_rows.append({
            "strategy_key": strategy_key,
            "start_date": str(nav_df["trade_date"].min().date()),
            "end_date": str(nav_df["trade_date"].max().date()),
            "trading_days": len(nav_df),
            "total_return": float((nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) - 1),
            "annual_vol": float(nav_df["daily_return"].std() * np.sqrt(252)) if nav_df["daily_return"].std() > 0 else 0.0,
            "max_drawdown": float((1 - nav_df["nav"] / nav_df["nav"].cummax()).min()),
            "source_files": len(csv_paths),
        })
        print(f"  {strategy_key:30s}  {len(nav_df):5d} days  {registry_rows[-1]['start_date']} ~ {registry_rows[-1]['end_date']}")

    # 写注册表
    reg_df = pd.DataFrame(registry_rows)
    reg_df = reg_df.sort_values("trading_days", ascending=False)
    reg_df.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[OK] {len(registry_rows)} strategies saved to {NAV_OUT_DIR}")
    print(f"[OK] Registry -> {REGISTRY_PATH}")

    if errors:
        print(f"\n[WARN] {len(errors)} strategies skipped (data issues):")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
