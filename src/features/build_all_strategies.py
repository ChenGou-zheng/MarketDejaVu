"""
Build all strategy NAVs from docs/strategy_definitions.csv.

For ETF and mapped strategies: fetch rqdata prices → build long-only NAV.
For MTM strategies: handled by build_snapshot.py internally.

Single source of truth: docs/strategy_definitions.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_registry.csv"
DEFINITIONS_PATH = PROJECT_ROOT / "docs" / "strategy_definitions.csv"

EXTERNAL_ETF = EXTERNAL_DIR / "etf_prices.parquet"
EXTERNAL_STRATEGY = EXTERNAL_DIR / "strategy_prices.parquet"


def fetch_price(code: str) -> pd.Series | None:
    import rqdatac
    rqdatac.init()
    try:
        df = rqdatac.get_price(code, start_date="2016-01-08", end_date="2026-04-30", fields=["close"])
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(-1)
        return df["close"].rename(code)
    except Exception:
        return None


def make_nav(close: pd.Series) -> pd.DataFrame:
    nav = close / close.iloc[0]
    daily_ret = nav.pct_change()
    return pd.DataFrame({"trade_date": close.index, "nav": nav.values, "daily_return": daily_ret.values})


def main():
    definitions = pd.read_csv(DEFINITIONS_PATH)
    print(f"Loaded {len(definitions)} strategy definitions")

    existing = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()
    existing_keys = set(existing["strategy_key"].tolist()) if not existing.empty else set()

    nav_entries = []
    combined_prices = {}

    # Process ETF and mapped strategies
    for _, row in definitions.iterrows():
        key = row["strategy_key"]
        stype = row["source_type"]
        code = str(row["rqdata_code"]).strip()

        if stype == "mtm":
            continue  # handled by build_snapshot.py MTM rebuild

        if not code:
            print(f"  [SKIP] {key}: no code")
            continue

        # Check if NAV already exists
        safe = key.replace("/", "_")
        nav_path = NAV_DIR / f"{safe}_nav.csv"
        if nav_path.exists():
            continue

        print(f"  [{stype:6s}] {key} -> {code}")
        close = fetch_price(code)
        if close is None or len(close) < 60:
            print(f"    FAIL: no data")
            continue

        combined_prices[f"{code}_{key}"] = close

        nav_df = make_nav(close)
        nav_df.to_csv(nav_path, index=False, encoding="utf-8-sig")

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
        print(f"    {len(close)}d  {close.index[0].date()} ~ {close.index[-1].date()}  tot={total_return:+.2%}")

    # Cache prices
    if combined_prices:
        cache = pd.DataFrame(combined_prices)
        cache.index = pd.to_datetime(cache.index)
        cache.to_parquet(EXTERNAL_STRATEGY)
        print(f"\n[OK] Cached {len(combined_prices)} price series")

    # Rebuild registry from definition file + MTM
    # For each definition, check if NAV file exists
    reg_rows = []
    for _, row in definitions.iterrows():
        key = row["strategy_key"]
        safe = key.replace("/", "_")
        nav_path = NAV_DIR / f"{safe}_nav.csv"

        if key.startswith("交易记录"):
            # MTM strategies: check existing registry
            m = existing[existing["strategy_key"] == key]
            if not m.empty:
                reg_rows.append(m.iloc[0].to_dict())
            continue

        if not nav_path.exists():
            continue

        nav_df = pd.read_csv(nav_path, nrows=1)
        if nav_df.empty:
            continue
        # Get date range from NAV file
        nav_full = pd.read_csv(nav_path, parse_dates=["trade_date"])
        start = nav_full["trade_date"].min()
        end = nav_full["trade_date"].max()
        days = len(nav_full)
        total_ret = float(nav_full["nav"].iloc[-1] / nav_full["nav"].iloc[0] - 1)
        ann_vol = float(nav_full["daily_return"].std() * (252 ** 0.5)) if nav_full["daily_return"].std() > 0 else 0.0
        max_dd = float((1 - nav_full["nav"] / nav_full["nav"].cummax()).min())

        reg_rows.append({
            "strategy_key": key,
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "trading_days": days,
            "total_return": total_ret,
            "annual_vol": ann_vol,
            "max_drawdown": max_dd,
            "source_files": 0,
        })

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Registry: {len(reg_df)} strategies from {DEFINITIONS_PATH}")


if __name__ == "__main__":
    main()
