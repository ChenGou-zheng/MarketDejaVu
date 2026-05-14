"""
Build synthetic strategy NAVs from aligned_daily index/SW data.

Strategies per asset:
  - 长持: buy & hold (passive)
  - 均线: MA5 > MA20 → long, else cash (0 return)
  - 动量: 20d momentum > 0 → long, else cash

Sources:
  - aligned_daily: HS300, ZZ500, CYB index closes (clean daily data)
  - sw_industry: 申万一级行业 (subset of sectors)

Output:
  - data/processed/strategy_nav/ETF_{type}_{name}_nav.csv
  - registry updated
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_PATH = PROJECT_ROOT / "data" / "processed" / "aligned_daily.parquet"
SW_PATH = PROJECT_ROOT / "data" / "external" / "sw_industry.parquet"
NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_registry.csv"

STRATEGY_TYPES = {"长持": "long_only", "均线": "ma_crossover", "动量": "momentum"}

INDEX_SOURCES = [
    ("HS300", "沪深300", "HS300_close"),
    ("ZZ500", "中证500", "ZZ500_close"),
    ("CYB", "创业板", "CYB_close"),
]

SW_SECTORS = [
    ("801010", "农业"),
    ("801030", "食品饮料"),
    ("801050", "有色金属"),
    ("801080", "电子"),
    ("801110", "医药生物"),
    ("801120", "食品饮料"),
    ("801150", "医药生物"),
    ("801160", "公用事业"),
    ("801170", "交通运输"),
    ("801180", "房地产"),
    ("801200", "商贸零售"),
    ("801210", "休闲服务"),
    ("801230", "综合"),
    ("801710", "建筑材料"),
    ("801720", "建筑装饰"),
    ("801730", "电气设备"),
    ("801740", "国防军工"),
    ("801750", "计算机"),
    ("801760", "传媒"),
    ("801770", "通信"),
    ("801780", "银行"),
    ("801790", "非银金融"),
    ("801880", "汽车"),
]

# Deduplicate SW sectors (some codes map to same name)
SW_UNIQUE = {}
for code, name in SW_SECTORS:
    if code not in SW_UNIQUE:
        SW_UNIQUE[code] = name
SW_SECTORS = list(SW_UNIQUE.items())


def build_long_only_nav(close: pd.Series, name: str) -> pd.DataFrame:
    nav = close / close.iloc[0]
    result = pd.DataFrame({"trade_date": close.index, "nav": nav.values})
    result["daily_return"] = result["nav"].pct_change()
    return result


def build_ma_crossover_nav(close: pd.Series, name: str) -> pd.DataFrame:
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    in_market = ma5 > ma20
    daily_ret = close.pct_change()
    strat_ret = daily_ret.where(in_market.shift(1), 0.0).fillna(0.0)
    nav = (1 + strat_ret).cumprod()
    result = pd.DataFrame({"trade_date": close.index, "nav": nav.values})
    result["daily_return"] = result["nav"].pct_change()
    return result


def build_momentum_nav(close: pd.Series, name: str) -> pd.DataFrame:
    mom_20d = close.pct_change(20)
    in_market = mom_20d > 0
    daily_ret = close.pct_change()
    strat_ret = daily_ret.where(in_market.shift(1), 0.0).fillna(0.0)
    nav = (1 + strat_ret).cumprod()
    result = pd.DataFrame({"trade_date": close.index, "nav": nav.values})
    result["daily_return"] = result["nav"].pct_change()
    return result


BUILDERS = {
    "long_only": build_long_only_nav,
    "ma_crossover": build_ma_crossover_nav,
    "momentum": build_momentum_nav,
}


def main():
    # Load data
    aligned = pd.read_parquet(ALIGNED_PATH)
    aligned.index = pd.to_datetime(aligned.index)

    sw = None
    if SW_PATH.exists():
        sw = pd.read_parquet(SW_PATH)
        sw.index = pd.to_datetime(sw.index)

    # Existing registry
    existing_reg = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()
    existing_keys = set(existing_reg["strategy_key"].tolist()) if not existing_reg.empty else set()

    new_strategies = []
    total_created = 0

    # — index-based strategies —
    print("=== Index-based strategies ===")
    for code, label, col in INDEX_SOURCES:
        if col not in aligned.columns:
            print(f"  {label}: {col} not found, skip")
            continue
        close = aligned[col].dropna()
        if len(close) < 200:
            print(f"  {label}: too few data ({len(close)} days), skip")
            continue

        for strategy_name, strategy_type in STRATEGY_TYPES.items():
            builder = BUILDERS[strategy_type]
            nav_df = builder(close, label)

            strategy_key = f"ETF_{strategy_name}_{label}"
            if strategy_key in existing_keys:
                continue

            safe_name = f"etf_{strategy_name}_{label}"
            out_path = NAV_DIR / f"{safe_name}_nav.csv"
            nav_df.to_csv(out_path, index=False, encoding="utf-8-sig")

            reg_row = _make_reg_row(strategy_key, nav_df)
            new_strategies.append(reg_row)
            existing_keys.add(strategy_key)
            total_created += 1
            print(f"  [{strategy_name}] {label}: {reg_row['trading_days']}d tot={reg_row['total_return']:+.2%} vol={reg_row['annual_vol']:.0%}")

    # — sector-based strategies (long-only) —
    print("\n=== Sector-based strategies (long-only) ===")
    if sw is not None:
        for code, name in SW_SECTORS:
            if code not in sw.columns:
                continue
            close = sw[code].dropna()
            if len(close) < 200:
                continue
            # Index the close series to aligned dates for consistency
            close = close.reindex(aligned.index, method="ffill").dropna()

            strategy_key = f"ETF_长持_{name}"
            if strategy_key in existing_keys:
                continue

            nav_df = build_long_only_nav(close, name)
            safe_name = f"etf_长持_{name}"
            out_path = NAV_DIR / f"{safe_name}_nav.csv"
            nav_df.to_csv(out_path, index=False, encoding="utf-8-sig")

            reg_row = _make_reg_row(strategy_key, nav_df)
            new_strategies.append(reg_row)
            existing_keys.add(strategy_key)
            total_created += 1
            print(f"  [长持] {name}: {reg_row['trading_days']}d tot={reg_row['total_return']:+.2%} vol={reg_row['annual_vol']:.0%}")

    if new_strategies:
        new_df = pd.DataFrame(new_strategies)
        combined = pd.concat([existing_reg, new_df], ignore_index=True)
        combined = combined.sort_values("trading_days", ascending=False)
        combined.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Created {total_created} new synthetic strategies. Registry has {len(combined)} entries.")


def _make_reg_row(strategy_key: str, nav_df: pd.DataFrame) -> dict:
    total_return = float(nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1)
    ann_vol = float(nav_df["daily_return"].std() * np.sqrt(252)) if nav_df["daily_return"].std() > 0 else 0.0
    max_dd = float((1 - nav_df["nav"] / nav_df["nav"].cummax()).min())
    return {
        "strategy_key": strategy_key,
        "start_date": str(nav_df["trade_date"].min().date()),
        "end_date": str(nav_df["trade_date"].max().date()),
        "trading_days": len(nav_df),
        "total_return": total_return,
        "annual_vol": ann_vol,
        "max_drawdown": max_dd,
        "source_files": 0,
    }


if __name__ == "__main__":
    main()
