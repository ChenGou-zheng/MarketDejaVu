"""
Strategy rank selector: pick top-N strategies by recent performance momentum.

At each decision date t:
  1. Compute each strategy's cumulative daily return over [t-lookback, t-1]
  2. Rank by cumulative return (excess over HS300)
  3. Pick top-1 strategy

No future information leakage: uses only daily returns known at t.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "processed" / "snapshot"
OUT_DIR = PROJECT_ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 60


def main():
    daily_ret = pd.read_parquet(SNAPSHOT_DIR / "rebuilt_returns.parquet")
    daily_ret.index = pd.to_datetime(daily_ret.index)

    aligned = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "aligned_daily.parquet")
    aligned.index = pd.to_datetime(aligned.index)
    hs300_ret = aligned["HS300_ret"]

    # Convert strategy returns to excess returns (subtract HS300)
    hs300_aligned = hs300_ret.reindex(daily_ret.index).ffill()
    excess_ret = daily_ret.sub(hs300_aligned, axis=0)

    strategies = list(excess_ret.columns)
    print(f"Rank selector: {len(strategies)} strategies, LOOKBACK={LOOKBACK}")

    decision_rows = []
    for i, t in enumerate(excess_ret.index):
        if i < LOOKBACK:
            continue

        window = excess_ret.iloc[i - LOOKBACK : i]
        cumulative = window.sum()

        best_strat = cumulative.idxmax()
        best_excess = cumulative.max()

        decision_rows.append({
            "decision_date": t,
            "strategy": best_strat,
            "best_cum_excess": best_excess,
            "n_valid": window[best_strat].notna().sum(),
        })

    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(OUT_DIR / "rank_decisions.csv", index=False, encoding="utf-8-sig")

    print(f"[DONE] rank_decisions.csv: {len(decisions)} rows")
    print(f"  Date range: {decisions['decision_date'].min()} ~ {decisions['decision_date'].max()}")

    top_n = decisions["strategy"].value_counts().head(10)
    print(f"  Top-10 strategies by selection count:")
    for s, c in top_n.items():
        print(f"    {s}: {c} ({c/len(decisions):.1%})")


if __name__ == "__main__":
    main()
