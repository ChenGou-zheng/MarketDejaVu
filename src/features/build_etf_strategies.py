"""
Build long-only strategy NAVs from cached rqdata ETF prices.

Loads data/external/etf_prices.parquet (cached by fetcher.py),
creates buy-&-hold NAV for each ETF, removes old synthetic strategies.

Output:
  - data/processed/strategy_nav/ETF_{name}_nav.csv
  - Cleaned strategy_registry.csv (old synthetics removed, real ETFs + MTM strategies)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_registry.csv"
ETF_PRICES_PATH = EXTERNAL_DIR / "etf_prices.parquet"

START_DATE = "2016-01-08"
END_DATE = "2026-04-30"

# Map from ETF close column name to strategy key suffix
ETF_COLUMNS = {
    "沪深300ETF_close": "沪深300ETF",
    "中证500ETF_close": "中证500ETF",
    "创业板ETF_close": "创业板ETF",
    "上证50ETF_close": "上证50ETF",
    "中证1000ETF_close": "中证1000ETF",
    "科创50ETF_close": "科创50ETF",
    "红利ETF_close": "红利ETF",
    "证券ETF_close": "证券ETF",
    "酒ETF_close": "酒ETF",
    "房地产ETF_close": "房地产ETF",
    "黄金ETF_close": "黄金ETF",
    "纳指ETF_close": "纳指ETF",
    "中概互联ETF_close": "中概互联ETF",
}


def main():
    if not ETF_PRICES_PATH.exists():
        print(f"[FAIL] ETF prices not found. Run `python src/data/fetcher.py` first.")
        return

    prices = pd.read_parquet(ETF_PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)
    print(f"Loaded ETF prices: {len(prices)} days, {len(prices.columns)} cols")
    print(f"  Range: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    # Load existing registry
    existing = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()

    # Remove old synthetic ETF strategies (keep MTM-clean strategies)
    etf_types = ["ETF_长持", "ETF_均线", "ETF_动量"]
    original = existing[~existing["strategy_key"].str.startswith(tuple(etf_types))].copy()
    print(f"Removed {len(existing) - len(original)} old synthetic strategies from registry")

    # Also remove old ETF nav files
    for f in NAV_DIR.glob("ETF_*_nav.csv"):
        f.unlink()
    print("Removed old ETF nav files")

    # Build new ETF long-only NAVs
    new_entries = []
    for col, name in ETF_COLUMNS.items():
        if col not in prices.columns:
            print(f"  [SKIP] {col} not found")
            continue

        close = prices[col].dropna()
        if len(close) < 100:
            print(f"  [SKIP] {name}: too few data ({len(close)} days)")
            continue

        nav = close / close.iloc[0]
        daily_ret = nav.pct_change()

        strategy_key = f"ETF_长持_{name}"
        safe_name = f"ETF_长持_{name}"

        # Save NAV CSV
        out_df = pd.DataFrame({
            "trade_date": close.index,
            "nav": nav.values,
            "daily_return": daily_ret.values,
        })
        out_path = NAV_DIR / f"{safe_name}_nav.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        total_return = float(nav.iloc[-1] - 1)
        ann_vol = float(daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0
        max_dd = float((1 - nav / nav.cummax()).min())

        reg_row = {
            "strategy_key": strategy_key,
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "trading_days": len(close),
            "total_return": total_return,
            "annual_vol": ann_vol,
            "max_drawdown": max_dd,
            "source_files": 0,
        }
        new_entries.append(reg_row)
        print(f"  [长持] {name:15s}  {len(close):4d}d  tot={total_return:+.2%}  vol={ann_vol:.0%}")

    # Merge registry
    new_df = pd.DataFrame(new_entries)
    combined = pd.concat([original, new_df], ignore_index=True)
    combined = combined.sort_values("trading_days", ascending=False)
    combined.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Registry: {len(combined)} entries ({len(new_entries)} real ETF strategies)")


if __name__ == "__main__":
    main()
