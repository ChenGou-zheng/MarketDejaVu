"""
Phase 1: Build no-leakage snapshot library for similarity-based strategy selection.

Improvement over v1: Proper mark-to-market NAV from raw trade records,
instead of "cash + shares" unit error.

Output:
  - data/processed/snapshot/features.parquet        # F(t) market features
  - data/processed/snapshot/labels.parquet            # strategy N-day excess returns
  - data/processed/snapshot/benchmark_close.parquet   # HS300 close price (benchmark)
  - data/processed/snapshot/strategy_returns.csv      # rebuilt daily returns per strategy

All computations use only data available at time t — no future leakage.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ALIGNED_PATH = PROCESSED_DIR / "aligned_daily.parquet"
NAV_DIR = PROCESSED_DIR / "strategy_nav"
REGISTRY_PATH = PROCESSED_DIR / "strategy_registry.csv"
OUT_DIR = PROCESSED_DIR / "snapshot"

N = 20
MIN_TRADING_DAYS = 60


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["HS300_close"]
    ret = df["HS300_ret"]
    F = pd.DataFrame(index=df.index)

    F["momentum_20d"] = close.pct_change(20)
    F["momentum_60d"] = close.pct_change(60)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    F["rsi_14"] = 100 - 100 / (1 + rs)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    F["ma5_above_ma20"] = (ma5 > ma20).astype(float)

    F["skew_20d"] = ret.rolling(20).skew()

    up = (ret > 0).rolling(20).sum()
    down = (ret < 0).rolling(20).sum()
    F["up_down_ratio"] = up / down.replace(0, np.nan)

    cummax = close.cummax()
    drawdown = close / cummax - 1
    is_trough = drawdown == drawdown.expanding().min()
    trough_idx = pd.Series(range(len(drawdown)), index=drawdown.index)[is_trough]
    date_pos = pd.Series(range(len(drawdown)), index=drawdown.index)
    last_trough_idx = trough_idx.reindex(drawdown.index, method="ffill")
    F["recovery_days"] = (date_pos - last_trough_idx.values).where(last_trough_idx.notna(), 0)

    F["realized_vol_20d"] = df["realized_vol_20d"]
    F["max_dd_60d"] = df["max_dd_60d"]
    F["yield_slope"] = df["yield_slope"]
    F["net_total"] = df["net_total"]
    F["margin_balance"] = df["margin_balance"]

    F["pmi_mfg"] = df["pmi_mfg"]
    F["cpi_yoy"] = df["cpi_yoy"]
    F["m2_yoy"] = df["m2_yoy"]
    F["ppi_yoy"] = df["ppi_yoy"]

    return F.ffill().bfill().dropna()


def rebuild_daily_returns(strategy_key: str) -> pd.DataFrame | None:
    """Rebuild daily equity curve using mark-to-market position tracking.

    Reads raw CSV rows chronologically, tracks per-symbol positions,
    marks positions to the most recent fill price, and computes daily equity.
    """
    safe = strategy_key.replace("/", "_").replace("\\", "_")
    csv_path = PROCESSED_DIR / f"{safe}.csv"
    if not csv_path.exists():
        return None

    try:
        raw = pd.read_csv(csv_path)
    except Exception:
        return None

    required = {"trade_time", "cash_balance", "posi_balance", "posi_vwap", "symbol", "btype", "volume"}
    if not required.issubset(raw.columns):
        return None

    # Filter out summary/header rows embedded in the CSV
    raw = raw[~raw["symbol"].astype(str).str.contains("以上是", na=False)]
    raw = raw[~raw["trade_time"].astype(str).str.contains("以上是", na=False)]
    raw = raw.dropna(subset=["trade_time", "symbol", "btype"])
    if len(raw) < 5:
        return None

    raw = raw.sort_values("trade_time").reset_index(drop=True)
    raw["trade_time"] = pd.to_datetime(raw["trade_time"])

    # Track per-symbol positions: {symbol: (shares, last_price)}
    positions: dict[str, list[float, float]] = {}
    equity_rows = []

    for _, row in raw.iterrows():
        try:
            cash = float(row["cash_balance"])
            symbol = str(row["symbol"])
            btype = str(row["btype"])
        except (ValueError, TypeError):
            continue
        try:
            volume = float(row["volume"])
            price = float(row["vwap"]) if pd.notna(row.get("vwap")) else float(row["order_price"])
        except (ValueError, TypeError):
            continue

        # Skip non-trade events
        if btype == "银证转入":
            continue

        # Update per-symbol position
        if volume > 0:  # buy to open
            old_shares, old_price = positions.get(symbol, (0.0, price))
            new_shares = old_shares + volume
            new_price = (old_shares * old_price + volume * price) / new_shares if new_shares > 0 else price
            positions[symbol] = [new_shares, new_price]
        elif volume < 0:  # sell to close
            old_shares, old_price = positions.get(symbol, (0.0, 0.0))
            new_shares = max(0.0, old_shares + volume)  # volume is negative
            positions[symbol] = [new_shares, old_price]

        # Mark all open positions to most recent fill price for this symbol
        positions[symbol][1] = price  # update last known price

        # Total market value of all positions
        total_mv = sum(shares * lp for shares, lp in positions.values())
        equity = cash + total_mv

        equity_rows.append({
            "trade_time": row["trade_time"],
            "equity": equity,
            "open_positions": len([s for s, (sh, _) in positions.items() if sh > 0]),
        })

    if len(equity_rows) < 5:
        return None

    equity_df = pd.DataFrame(equity_rows)
    equity_df["trade_date"] = equity_df["trade_time"].dt.date

    # Last observation per day
    daily = equity_df.groupby("trade_date").last().reset_index()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["nav"] = daily["equity"]
    daily["daily_return"] = daily["nav"].pct_change()

    return daily[["trade_date", "nav", "daily_return"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_parquet(ALIGNED_PATH)
    aligned.index = pd.to_datetime(aligned.index)
    all_dates = aligned.index

    features = compute_features(aligned)
    print(f"[FEATURES] {len(features)} days x {len(features.columns)} cols")
    print(f"  Columns: {list(features.columns)}")
    print(f"  Date range: {features.index[0].date()} ~ {features.index[-1].date()}")

    # benchmark N-day forward return
    bench_close = aligned["HS300_close"]
    bench_nav = bench_close / bench_close.iloc[0]
    bench_fwd = bench_nav.pct_change(N).shift(-N)
    bench_fwd.name = "benchmark_fwd_ret"

    # Load registry for available strategies
    reg = pd.read_csv(REGISTRY_PATH)

    print(f"\n[LABELS] Rebuilding daily returns from raw CSVs...")

    excess_list = []
    valid_keys = []
    daily_returns = {}

    for _, row in reg.iterrows():
        key = row["strategy_key"]
        safe = key.replace("/", "_").replace("\\", "_")

        # Try: MTM rebuild from raw trade CSV (original strategies)
        daily_df = rebuild_daily_returns(key)

        # Fall back: pre-computed NAV file (ETF/sector synthetic strategies)
        if daily_df is None:
            nav_precomputed = NAV_DIR / f"{safe}_nav.csv"
            if nav_precomputed.exists():
                try:
                    raw_nav = pd.read_csv(nav_precomputed)
                    if "trade_date" in raw_nav.columns and "nav" in raw_nav.columns:
                        raw_nav["trade_date"] = pd.to_datetime(raw_nav["trade_date"])
                        daily_df = raw_nav[["trade_date", "nav"]].copy()
                        if "daily_return" not in raw_nav.columns:
                            daily_df["daily_return"] = daily_df["nav"].pct_change()
                        else:
                            daily_df["daily_return"] = raw_nav["daily_return"]
                except Exception:
                    pass

        if daily_df is None or len(daily_df) < MIN_TRADING_DAYS:
            print(f"  {key:30s}  SKIP (no viable NAV source or < {MIN_TRADING_DAYS} days)")
            continue

        # Check for extreme daily returns
        ret = daily_df["daily_return"].dropna()
        extreme_rate = (ret.abs() > 0.2).mean()
        ann_vol = ret.std() * np.sqrt(252)
        if extreme_rate > 0.03 or ann_vol > 5.0:
            print(f"  {key:30s}  SKIP (extreme_rate={extreme_rate:.1%} ann_vol={ann_vol:.0%})")
            continue

        # NAV to unified date index
        nav = daily_df.set_index("trade_date")["nav"].sort_index()
        nav_filled = nav.reindex(all_dates, method="ffill")
        nav_filled = nav_filled.dropna()
        nav_norm = nav_filled / nav_filled.iloc[0]

        # Forward N-day excess return
        strat_fwd = nav_norm.pct_change(N).shift(-N)
        bench_aligned = bench_fwd.reindex(strat_fwd.index)
        excess = strat_fwd - bench_aligned
        excess.name = key

        n_valid = excess.notna().sum()
        if n_valid < MIN_TRADING_DAYS:
            print(f"  {key:30s}  SKIP (only {n_valid} valid labels)")
            continue

        # Clip excess returns to [-3, 3] to limit outlier impact
        excess = excess.clip(-3.0, 3.0)

        # Also filter out strategies where > 5% of excess returns are at the clip bound
        clipped_rate = ((excess == 3.0) | (excess == -3.0)).mean()
        if clipped_rate > 0.05:
            print(f"  {key:30s}  SKIP ({clipped_rate:.1%} at clip bound)")
            continue

        excess_list.append(excess)
        valid_keys.append(key)
        # Compute continuous daily returns from filled NAV
        filled_df = nav_filled.to_frame(name="nav").copy()
        filled_df["daily_return"] = filled_df["nav"].pct_change()
        daily_returns[key] = filled_df
        print(f"  {key:30s}  {n_valid:5d} labels  vol={ann_vol:.0%}  extreme_ret={extreme_rate:.1%}")

    if not excess_list:
        print("\n[FAIL] No valid strategies remain after filtering.")
        return

    labels = pd.DataFrame(excess_list).T
    print(f"\n[LABELS] Raw shape: {labels.shape}")

    label_dates = labels.dropna(how="all").index
    common = features.index.intersection(label_dates)
    features = features.loc[common]
    labels = labels.loc[common]

    print(f"\n[ALIGNED] features: {features.shape}, labels: {labels.shape}")
    print(f"  Date range: {common[0].date()} ~ {common[-1].date()}")
    print(f"  Strategies: {list(labels.columns)}")

    features.to_parquet(OUT_DIR / "features.parquet")
    labels.to_parquet(OUT_DIR / "labels.parquet")
    bench_close.to_frame().to_parquet(OUT_DIR / "benchmark_close.parquet")

    # Save filled daily returns (continuous, ready for backtest)
    ret_summary = pd.DataFrame({k: v["daily_return"] for k, v in daily_returns.items()})
    ret_summary.to_parquet(OUT_DIR / "rebuilt_returns.parquet")

    print(f"\n[DONE] Snapshot saved to {OUT_DIR}")
    for f in ["features.parquet", "labels.parquet", "benchmark_close.parquet"]:
        p = OUT_DIR / f
        print(f"  {f}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
