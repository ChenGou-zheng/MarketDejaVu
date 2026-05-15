"""
Rebuild strategy daily NAVs from raw trade CSVs using rqdata stock close prices.

For each strategy:
  1. Parse raw CSV chronologically
  2. Track per-stock positions (shares) and cash balance
  3. At each day's end: NAV = cash + Σ(shares × rqdata_close_price)
  4. Daily return = NAV(t) / NAV(t-1) - 1
  5. First 银证转入 amount = initial capital (NAV start)

Stock prices from data/external/stock_prices.parquet (cached from rqdata).
Output NAVs to data/processed/strategy_nav/{key}_nav.csv
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NAV_DIR = PROCESSED_DIR / "strategy_nav"
STOCK_PRICES_PATH = PROJECT_ROOT / "data" / "external" / "stock_prices.parquet"
REGISTRY_PATH = PROCESSED_DIR / "strategy_registry.csv"
DEFINITIONS_PATH = PROJECT_ROOT / "docs" / "strategy_definitions.csv"

MIN_DAYS = 60
MAX_CAPITAL_TRADE_GAP = 30  # days; if > 30, CSV is incomplete (missing trades)


def to_rqdata(symbol: str) -> str:
    """Convert CSV symbol format to rqdata code."""
    if symbol.startswith("SHSE."):
        return symbol.replace("SHSE.", "") + ".XSHG"
    elif symbol.startswith("SZSE."):
        return symbol.replace("SZSE.", "") + ".XSHE"
    return symbol


def rebuild_mtm_nav(strategy_key: str, stock_prices: pd.DataFrame) -> pd.DataFrame | None:
    """Rebuild daily NAV for a strategy using MTM with stock close prices."""
    safe = strategy_key.replace("/", "_").replace("\\", "_")
    csv_path = PROCESSED_DIR / f"{safe}.csv"
    if not csv_path.exists():
        return None

    try:
        raw = pd.read_csv(csv_path)
    except Exception:
        return None

    required = {"trade_time", "cash_balance", "posi_balance", "symbol", "btype"}
    if not required.issubset(raw.columns):
        return None

    # Clean (keep 银证转入 rows even if symbol is NaN)
    raw = raw[~raw["trade_time"].astype(str).str.contains("以上是", na=False)]
    raw = raw.dropna(subset=["trade_time", "btype"])
    # Only drop NaN symbol for non-银证转入 rows
    raw = raw[~((raw["symbol"].isna()) & (raw["btype"] != "银证转入"))]
    raw = raw[~raw["symbol"].astype(str).str.contains("以上是", na=False)]
    if len(raw) < 5:
        return None

    raw = raw.sort_values("trade_time").reset_index(drop=True)
    raw["trade_time"] = pd.to_datetime(raw["trade_time"])

    # Determine initial capital from first 银证转入
    initial_capital = None
    capital_time = None
    for _, row in raw.iterrows():
        if str(row["btype"]) == "银证转入":
            try:
                initial_capital = float(row["cash_balance"])
                capital_time = pd.to_datetime(row["trade_time"])
            except (ValueError, TypeError):
                initial_capital = 0
            break

    if initial_capital is None or initial_capital <= 0:
        return None

    # Check gap between initial capital and first trade
    if capital_time is not None:
        for _, row in raw.iterrows():
            if str(row.get("btype", "")) == "银证转入":
                continue
            if pd.notna(row.get("trade_time")):
                first_trade_time = pd.to_datetime(row["trade_time"])
                gap = (first_trade_time - capital_time).days
                if gap > MAX_CAPITAL_TRADE_GAP:
                    return None  # CSV is missing intermediate trades
                break

    # Track positions per stock: {rqcode: shares}
    positions: dict[str, float] = defaultdict(float)

    # Process every row, record state after each trade
    daily_states: dict = {}  # {date: (cash, positions_dict)}
    last_cash = initial_capital

    for _, row in raw.iterrows():
        try:
            cash = float(row["cash_balance"])
            symbol = str(row["symbol"])
            btype = str(row["btype"])
            volume = float(row["volume"])
        except (ValueError, TypeError):
            continue

        if btype == "银证转入":
            last_cash = cash
            continue

        if pd.isna(symbol):
            continue

        rqcode = to_rqdata(symbol)
        if rqcode not in stock_prices.columns:
            continue

        # Update position
        old_shares = positions.get(rqcode, 0.0)
        new_shares = max(0.0, old_shares + volume)
        if new_shares > 0:
            positions[rqcode] = new_shares
        elif rqcode in positions:
            del positions[rqcode]

        last_cash = cash
        date_key = row["trade_time"].date()
        daily_states[date_key] = (cash, dict(positions))

    if not daily_states:
        return None

    trade_dates = set(daily_states.keys())

    # Build day-by-day NAV from trade dates to end, filling gaps with MTM
    all_dates = sorted(stock_prices.index)
    first_date = pd.Timestamp(min(trade_dates))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(first_date.date())]

    nav_values = {}
    last_positions: dict[str, float] = {}
    last_cash = initial_capital

    for date in all_dates:
        date_key = date.date()
        if date_key in daily_states:
            last_cash, last_positions = daily_states[date_key]

        # Compute MTM NAV
        mv = 0.0
        for rqcode, shares in last_positions.items():
            if rqcode in stock_prices.columns:
                price = stock_prices.loc[date, rqcode]
                if pd.notna(price):
                    mv += shares * price
        nav = last_cash + mv
        nav_values[date] = nav

    nav_series = pd.Series(nav_values, name="nav")
    nav_series.index = pd.to_datetime(nav_series.index)

    result = nav_series.to_frame("nav").reset_index()
    result.columns = ["trade_date", "nav"]
    result["daily_return"] = result["nav"].pct_change()

    return result


def main():
    stock_prices = pd.read_parquet(STOCK_PRICES_PATH)
    stock_prices.index = pd.to_datetime(stock_prices.index)
    print(f"Loaded stock prices: {len(stock_prices)} days, {len(stock_prices.columns)} stocks")

    # Find all strategy CSVs
    csvs = sorted(PROCESSED_DIR.glob("交易记录*.csv"))
    print(f"Found {len(csvs)} strategy CSVs")

    # Read existing registry
    existing = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()

    results = []
    for csv_path in csvs:
        key = csv_path.stem  # filename without .csv
        print(f"  [{key}] ", end="")

        nav_df = rebuild_mtm_nav(key, stock_prices)
        if nav_df is None or len(nav_df) < MIN_DAYS:
            print(f"SKIP (MTM failed or < {MIN_DAYS} days)")
            continue

        # ALWAYS save the corrected NAV (overwrite old broken files)
        out_path = NAV_DIR / f"{key}_nav.csv"
        nav_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Quality filter (registry only, NAV files already saved)
        ret = nav_df["daily_return"].dropna()
        extreme_rate = (ret.abs() > 0.2).mean()
        ann_vol = ret.std() * np.sqrt(252)
        if extreme_rate > 0.03 or ann_vol > 5.0:
            print(f"SKIP (extreme={extreme_rate:.1%}, vol={ann_vol:.0%})")
            continue

        total_return = float(nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1)
        max_dd = float((1 - nav_df["nav"] / nav_df["nav"].cummax()).min())

        results.append({
            "strategy_key": key,
            "start_date": str(nav_df["trade_date"].min().date()),
            "end_date": str(nav_df["trade_date"].max().date()),
            "trading_days": len(nav_df),
            "total_return": total_return,
            "annual_vol": ann_vol,
            "max_drawdown": max_dd,
            "source_files": 0,
        })

        print(f"OK  {len(nav_df)}d  vol={ann_vol:.0%}  extreme={extreme_rate:.1%}  tot={total_return:+.2%}")

    # Create MTM-only strategy definitions
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(DEFINITIONS_PATH, index=False, encoding="utf-8-sig")
        result_df.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
        print(f"\n[DONE] {len(results)} strategies passed quality filter")
        print(f"  Registry -> {REGISTRY_PATH}")
        print(f"  Definitions -> {DEFINITIONS_PATH}")
    else:
        print(f"\n[FAIL] No strategies passed the quality filter")


if __name__ == "__main__":
    main()
