"""
Dynamic backtest: KNN + walk-forward strategy quality filter.

For each block:
  1. Get all significant strategies from similarity engine
  2. Filter to strategies with WR >= MIN_WR over last MIN_SELECTIONS blocks
  3. Pick best by predicted mean_excess
  4. Hold for N days, record outcome, update track

The quality filter prevents KNN from selecting strategies that failed historically.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "processed" / "snapshot"
ALIGNED_PATH = PROJECT_ROOT / "data" / "processed" / "aligned_daily.parquet"
OUT_DIR = PROJECT_ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 20
MIN_SELECTIONS = 3
MIN_WR = 0.50


def compute_dsr(excess_ret: pd.Series, n_trials: int) -> dict:
    ret = excess_ret.dropna()
    T = len(ret)
    if T < 10:
        return {"sharpe": 0, "deflated_p": 1.0}
    sr = ret.mean() / ret.std() * np.sqrt(252)
    skew = ret.skew()
    kurt = ret.kurtosis()
    se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4 * sr**2) / T)
    z_sr = sr / se_sr if se_sr > 0 else 0.0
    gamma = 0.5772156649
    expected_max_z = (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    dz = z_sr - expected_max_z
    return {"sharpe_annual": sr, "deflated_z": dz, "deflated_p": 1.0 - norm.cdf(dz), "n_obs": T}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-filter", action="store_true")
    args = parser.parse_args()
    qf = args.quality_filter

    print(f"=== Backtest with{'' if qf else 'out'} quality filter ===")

    # Load all significant decisions (not just best)
    dec = pd.read_csv(OUT_DIR / "similarity_decisions.csv", parse_dates=["decision_date"])
    sig_dec = dec[dec["is_significant"]].copy()

    daily_ret = pd.read_parquet(SNAPSHOT_DIR / "rebuilt_returns.parquet")
    daily_ret.index = pd.to_datetime(daily_ret.index)

    aligned = pd.read_parquet(ALIGNED_PATH)
    aligned.index = pd.to_datetime(aligned.index)
    hs300_ret = aligned["HS300_ret"].dropna()

    strategies = [c for c in daily_ret.columns]
    all_dates = daily_ret.index.intersection(hs300_ret.index)
    all_dates = all_dates[all_dates >= sig_dec["decision_date"].min()]

    # Walk-forward strategy track record
    track = defaultdict(lambda: {"selections": 0, "wins": 0})

    portfolio_nav = [1.0]
    portfolio_ret, excess_ret_list = [], []
    daily_records = []
    current_strategy = None
    days_since = 0

    for d_idx, date in enumerate(all_dates):
        if days_since == 0:
            # Find at most recent decision
            recent = sig_dec[sig_dec["decision_date"] <= date]
            if recent.empty:
                current_strategy = None
            else:
                latest_dt = recent["decision_date"].max()
                candidates = recent[recent["decision_date"] == latest_dt]

                if qf:
                    # Filter by track record: WR >= MIN_WR after MIN_SELECTIONS
                    def _eligible(s):
                        t = track.get(s, {"selections": 0, "wins": 0})
                        return t["selections"] < MIN_SELECTIONS or t["wins"] / t["selections"] >= MIN_WR

                    eligible = candidates[candidates["strategy"].apply(_eligible)]
                    if not eligible.empty:
                        best_row = eligible.loc[eligible["mean_excess"].idxmax()]
                        current_strategy = best_row["strategy"]
                    else:
                        # Fallback: pick best overall even if ineligible
                        best_row = candidates.loc[candidates["mean_excess"].idxmax()]
                        current_strategy = best_row["strategy"]
                else:
                    best_row = candidates.loc[candidates["mean_excess"].idxmax()]
                    current_strategy = best_row["strategy"]

        # Compute returns
        if current_strategy is not None and current_strategy in daily_ret.columns:
            sr = daily_ret.loc[date, current_strategy]
            daily_ret_val = sr if pd.notna(sr) else hs300_ret.loc[date]
        else:
            daily_ret_val = hs300_ret.loc[date]

        bench_ret = hs300_ret.loc[date]
        excess = daily_ret_val - bench_ret
        portfolio_ret.append(daily_ret_val)
        excess_ret_list.append(excess)
        portfolio_nav.append(portfolio_nav[-1] * (1 + daily_ret_val))
        daily_records.append({
            "trade_date": date,
            "portfolio_return": daily_ret_val,
            "benchmark_return": bench_ret,
            "excess_return": excess,
            "selected_strategy": current_strategy or "BENCH",
            "portfolio_nav": portfolio_nav[-1],
        })
        days_since = (days_since + 1) % N

        # At end of block, update track record
        if days_since == 0 and current_strategy is not None:
            block_excess = pd.Series(excess_ret_list[-N:]).sum()
            track[current_strategy]["selections"] += 1
            if block_excess > 0:
                track[current_strategy]["wins"] += 1

    result = pd.DataFrame(daily_records)

    # — metrics —
    es = pd.Series(excess_ret_list)
    total = (1 + es).prod() - 1
    ann = (1 + total) ** (252 / len(es)) - 1
    te = es.std() * np.sqrt(252)
    ir = ann / te if te > 0 else 0
    wr = (es > 0).mean()
    dd = (1 - pd.Series(portfolio_nav) / pd.Series(portfolio_nav).cummax()).max()
    dsr = compute_dsr(es, len(strategies))

    result["block_id"] = result.index // N
    be = result.groupby("block_id")["excess_return"].apply(lambda x: (1 + x).prod() - 1)
    bw = (be > 0).mean()

    split = pd.Timestamp("2024-01-01")

    print(f"\n{'='*65}")
    print(f"  Backtest (quality_filter={'ON' if qf else 'OFF'})")
    print(f"{'='*65}")
    print(f"  Days:      {len(result)}")
    print(f"  Ann ex:    {ann:+.2%}")
    print(f"  IR:        {ir:.3f}")
    print(f"  Daily WR:  {wr:.1%}")
    print(f"  Block WR:  {bw:.1%}  (mean={be.mean():+.2%})")
    print(f"  Max DD:    {dd:.2%}")
    print(f"  DSR p:     {dsr['deflated_p']:.4f}")
    print(f"{'─'*65}")
    for label, sub in [("In-sample", result[result["trade_date"] < split]),
                        ("Out-of-sample", result[result["trade_date"] >= split])]:
        if len(sub) < 20:
            continue
        e = sub["excess_return"]
        t = (1 + e).prod() - 1
        a = (1 + t) ** (252 / len(e)) - 1
        sub["bid"] = sub.index // N
        bw2 = (sub.groupby("bid")["excess_return"].apply(lambda x: (1 + x).prod() - 1) > 0).mean()
        print(f"  {label:18s}  ann={a:+.2%}  daily_WR={(e>0).mean():.1%}  block_WR={bw2:.1%}")

    tag = f"knn_qf_{'on' if qf else 'off'}"
    result.to_csv(OUT_DIR / f"backtest_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[DONE] backtest_{tag}.csv")

    # Track record summary
    print(f"\n  Strategy track records ({sum(1 for t in track.values() if t['selections']>=MIN_SELECTIONS and t['wins']/t['selections']>=MIN_WR)}/{len(track)} eligible):")
    for s, t in sorted(track.items(), key=lambda x: -x[1]["selections"]):
        w = t["wins"] / t["selections"] if t["selections"] > 0 else 0
        el = "E" if (t["selections"] >= MIN_SELECTIONS and w >= MIN_WR) else ("-" if t["selections"] >= MIN_SELECTIONS else "?")
        print(f"    {el} {s:30s}  sel={t['selections']:3d}  WR={w:.0%}")


if __name__ == "__main__":
    main()
