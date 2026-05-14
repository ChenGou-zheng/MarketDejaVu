"""
Map total_summary strategies to real index/ETF codes via rqdata.
Fetch and cache daily prices, determine shortest data range.

Strategy → Code mapping (manual curation based on strategy name):
  - "{行业}ETF增强" / "{行业}etf增强" → sector ETF
  - "{指数}增强" → tracking index
  - Others → best-guess index or ETF

Output:
  - data/external/strategy_prices.parquet  (all strategy price series)
  - data/processed/strategy_nav/ETF_{name}_nav.csv  (per-strategy NAVs)
  - Cleansed strategy_registry.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_registry.csv"
CACHE_PATH = EXTERNAL_DIR / "strategy_prices.parquet"

START_DATE = "2016-01-08"
END_DATE = "2026-04-30"

# Mapping: strategy_name → (code, type, label)
# type: 'index' or 'etf'
STRATEGY_MAP = {
    "沪深300增强策略": ("000300.XSHG", "index", "沪深300"),
    "中证500增强": ("000905.XSHG", "index", "中证500"),
    "中证1000增强": ("000852.XSHG", "index", "中证1000"),
    "中证800增强": ("000906.XSHG", "index", "中证800"),
    "中证2000增强": ("932373.CSI", "index", "中证2000"),
    "创业板增强": ("399006.XSHE", "index", "创业板指"),
    "双创50增强": ("931643.CSI", "index", "科创创业50"),
    "煤炭周期优选动态轮动策略": ("000820.XSHG", "index", "中证煤炭"),
    "半导体优选策略": ("990001.CSI", "index", "中华半导体芯片"),
    "计算机ETF优选策略": ("930851.CSI", "index", "中证计算机"),
    "军工etf增强": ("399967.XSHE", "index", "中证军工"),
    "医疗etf增强": ("399989.XSHE", "index", "中证医疗"),
    "通信etf增强": ("931160.CSI", "index", "中证通信"),
    "旅游etf增强": ("930633.CSI", "index", "中证旅游"),
    "国企etf增强": ("000955.XSHG", "index", "中证国企"),
    "游戏etf增强": ("930901.CSI", "index", "中证动漫游戏"),
    "酒etf增强": ("399987.XSHE", "index", "中证酒"),
    "食品etf增强": ("930653.CSI", "index", "中证食品"),
    "化工ETF优选策略": ("930707.CSI", "index", "中证化工"),
    "机器人ETF优选策略": ("H30590.CSI", "index", "中证机器人"),
    "养殖etf增强": ("930707.CSI", "index", "中证养殖"),  # 近似: 中证农业
    "成长红利量化选股": ("000922.XSHG", "index", "中证红利"),
    "动量趋势策略": ("000300.XSHG", "index", "沪深300"),  
    "etf动量改": ("000300.XSHG", "index", "沪深300"),
    "行业etf增强": ("000300.XSHG", "index", "沪深300"),
    "策略ETF": ("H30269.CSI", "index", "中证策略"),
    "策略etf2": ("000300.XSHG", "index", "沪深300"),
    "全球etf增强": ("513100.XSHG", "etf", "纳指ETF"),
    "百亿etf 1只": ("000300.XSHG", "index", "沪深300"),
    "百亿etf 2只": ("000905.XSHG", "index", "中证500"),
    "桥水全天候": ("000300.XSHG", "index", "沪深300"),
    "申万动量": ("000300.XSHG", "index", "沪深300"),
    "均衡持仓": ("000300.XSHG", "index", "沪深300"),
    "杠铃": ("000300.XSHG", "index", "沪深300"),
    "etf动量轮动": ("000300.XSHG", "index", "沪深300"),
    "形态识别": ("000300.XSHG", "index", "沪深300"),
    "综合全": ("000300.XSHG", "index", "沪深300"),
    "综合拆分1": ("000300.XSHG", "index", "沪深300"),
    "综合拆分2": ("000300.XSHG", "index", "沪深300"),
    "锤子": ("000300.XSHG", "index", "沪深300"),
}

# Also add existing 13 ETFs from the original universe
# (already in strategy_registry as ETF_长持_{name})
EXISTING_ETFS = {
    "沪深300ETF": "510300.XSHG",
    "中证500ETF": "510500.XSHG",
    "创业板ETF": "159915.XSHE",
    "上证50ETF": "510050.XSHG",
    "中证1000ETF": "512100.XSHG",
    "科创50ETF": "588000.XSHG",
    "红利ETF": "510880.XSHG",
    "证券ETF": "512880.XSHG",
    "酒ETF": "512690.XSHG",
    "房地产ETF": "512200.XSHG",
    "黄金ETF": "518880.XSHG",
    "纳指ETF": "513100.XSHG",
    "中概互联ETF": "513050.XSHG",
}


def fetch_price(code: str, code_type: str) -> pd.Series | None:
    import rqdatac
    rqdatac.init()
    try:
        df = rqdatac.get_price(code, start_date=START_DATE, end_date=END_DATE, fields=["close"])
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(-1)
        return df["close"].rename(code)
    except Exception as e:
        print(f"    FAIL: {e}")
        return None


def make_nav(close: pd.Series, name: str) -> pd.DataFrame:
    nav = close / close.iloc[0]
    daily_ret = nav.pct_change()
    return pd.DataFrame({
        "trade_date": close.index,
        "nav": nav.values,
        "daily_return": daily_ret.values,
    })


def main():
    # Load existing registry
    existing = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()
    existing_keys = set(existing["strategy_key"].tolist()) if not existing.empty else set()

    # We'll collect all price series
    all_prices = {}
    nav_entries = []

    # Phase A: fetch mapped strategy prices
    print("=== Fetching strategy-mapped index/ETF prices ===")
    for strat_name, (code, ctype, label) in STRATEGY_MAP.items():
        key = f"策略_{strat_name}"
        if key in existing_keys:
            print(f"  [SKIP] {strat_name} (already in registry)")
            continue

        print(f"  [{strat_name}] → {code} ({ctype})")
        close = fetch_price(code, ctype)
        if close is None or len(close) < 60:
            print(f"    SKIP: insufficient data")
            continue

        all_prices[f"{code}_{strat_name}"] = close
        nav_df = make_nav(close, strat_name)

        safe_name = f"策略_{strat_name}"
        out_path = NAV_DIR / f"{safe_name}_nav.csv"
        nav_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        total_return = float(nav_df["nav"].iloc[-1] - 1)
        ann_vol = float(nav_df["daily_return"].std() * (252 ** 0.5)) if nav_df["daily_return"].std() > 0 else 0.0
        max_dd = float((1 - nav_df["nav"] / nav_df["nav"].cummax()).min())

        nav_entries.append({
            "strategy_key": key,
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "trading_days": len(close),
            "total_return": total_return,
            "annual_vol": ann_vol,
            "max_drawdown": max_dd,
            "source_files": 0,
        })
        existing_keys.add(key)
        print(f"    {len(close)} days, {close.index[0].date()} ~ {close.index[-1].date()}, vol={ann_vol:.0%}")

    # Phase B: also fetch new ETF codes not yet in registry
    # (the existing 13 are already handled by build_etf_strategies.py)
    # New ETFs to add beyond the 13:
    new_etfs = [
        ("512480.XSHG", "半导体ETF", "etf"),
        ("512660.XSHG", "军工ETF", "etf"),
        ("512170.XSHG", "医疗ETF", "etf"),
        ("515880.XSHG", "通信ETF", "etf"),
        ("159766.XSHE", "旅游ETF", "etf"),
        ("510270.XSHG", "国企ETF", "etf"),
        ("516010.XSHG", "游戏ETF", "etf"),
        ("515710.XSHG", "食品ETF", "etf"),
        ("562500.XSHG", "机器人ETF", "etf"),
        ("159865.XSHE", "养殖ETF", "etf"),
        ("515800.XSHG", "中证800ETF", "etf"),
        ("563300.XSHG", "中证2000ETF", "etf"),
        ("159783.XSHE", "双创ETF", "etf"),
    ]

    print("\n=== Fetching additional ETF prices ===")
    for code, label, ctype in new_etfs:
        key = f"ETF_长持_{label}"
        if key in existing_keys:
            print(f"  [SKIP] {label} (already in registry)")
            continue

        print(f"  [{label}] → {code}")
        close = fetch_price(code, ctype)
        if close is None or len(close) < 60:
            print(f"    SKIP: insufficient data")
            continue

        all_prices[f"etf_{code}"] = close
        nav_df = make_nav(close, label)

        safe_name = f"ETF_长持_{label}"
        out_path = NAV_DIR / f"{safe_name}_nav.csv"
        nav_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        total_return = float(nav_df["nav"].iloc[-1] - 1)
        ann_vol = float(nav_df["daily_return"].std() * (252 ** 0.5)) if nav_df["daily_return"].std() > 0 else 0.0
        max_dd = float((1 - nav_df["nav"] / nav_df["nav"].cummax()).min())

        nav_entries.append({
            "strategy_key": key,
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "trading_days": len(close),
            "total_return": total_return,
            "annual_vol": ann_vol,
            "max_drawdown": max_dd,
            "source_files": 0,
        })
        existing_keys.add(key)
        print(f"    {len(close)} days, {close.index[0].date()} ~ {close.index[-1].date()}, vol={ann_vol:.0%}")

    # Save cache
    if all_prices:
        cache_df = pd.DataFrame(all_prices)
        cache_df.index = pd.to_datetime(cache_df.index)
        cache_df.to_parquet(CACHE_PATH)
        print(f"\n[OK] Cached {len(all_prices)} price series -> {CACHE_PATH}")

    # Update registry
    if nav_entries:
        new_df = pd.DataFrame(nav_entries)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values("trading_days", ascending=False)
        combined.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] Registry updated: {len(combined)} entries (+{len(nav_entries)} new)")

    # Phase C: determine shortest data range (latest start date)
    print("\n=== Data range analysis ===")
    registry = pd.read_csv(REGISTRY_PATH)
    registry["start_date"] = pd.to_datetime(registry["start_date"])
    registry = registry.sort_values("start_date", ascending=False)
    print(f"Latest 10 start dates (shortest-history strategies):")
    for _, r in registry.head(10).iterrows():
        print(f"  {r['strategy_key']:35s}  start={r['start_date'].date()}  days={r['trading_days']}")

    print(f"\n  Earliest start:  {registry['start_date'].min().date()}")
    print(f"  Latest start:    {registry['start_date'].max().date()}")
    print(f"  Median start:    {registry['start_date'].median().date()}")
    print(f"  Strategies with start > 2022: {(registry['start_date'] > pd.Timestamp('2022-01-01')).sum()}")
    print(f"  Strategies with start > 2023: {(registry['start_date'] > pd.Timestamp('2023-01-01')).sum()}")
    print(f"  Strategies with start > 2024: {(registry['start_date'] > pd.Timestamp('2024-01-01')).sum()}")

    # Suggested cutoff
    p90 = registry["start_date"].quantile(0.9)
    p95 = registry["start_date"].quantile(0.95)
    print(f"\n  90th percentile start: {p90.date()}")
    print(f"  95th percentile start: {p95.date()}")
    print(f"  Suggested cutoff: {p90.date()} (covers 90% of strategies)")
    print(f"  After cutoff: {len(registry[registry['start_date'] > p90])} strategies excluded")


if __name__ == "__main__":
    main()
