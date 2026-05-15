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
            # MTM strategies: NAVs are built by rebuild_mtm_nav.py
            # Just check if NAV file exists for registry
            safe = key.replace("/", "_")
            nav_path = NAV_DIR / f"{safe}_nav.csv"
            nav_entries.append({
                "strategy_key": key,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "trading_days": row["trading_days"],
                "total_return": row["total_return"],
                "annual_vol": row["annual_vol"],
                "max_drawdown": row["max_drawdown"],
                "source_files": 0,
            })
            existing_keys.add(key)
            continue

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

    # Build registry from nav_entries (includes ETF, mapped, and MTM)
    reg_df = pd.DataFrame(nav_entries)
    reg_df.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Registry: {len(reg_df)} strategies from {DEFINITIONS_PATH}")


if __name__ == "__main__":
    main()
