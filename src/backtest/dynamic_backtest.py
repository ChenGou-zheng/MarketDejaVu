"""
Dynamic backtest with top-K diversification.

Selects top-K significant strategies (by mean_excess) and equal-weights them.
Reduces volatility vs single-strategy approach.
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
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--quality-filter", action="store_true")
    args = parser.parse_args()
    top_k = args.top_k
    qf = args.quality_filter

    print(f"=== Backtest top_k={top_k} quality_filter={'ON' if qf else 'OFF'} ===")

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

    track = defaultdict(lambda: {"selections": 0, "wins": 0})

    portfolio_nav = [1.0]
    portfolio_ret, excess_ret_list = [], []
    daily_records = []
    current_strategies: list[str] = []
    days_since = 0

    def _eligible(s):
        t = track.get(s, {"selections": 0, "wins": 0})
        return t["selections"] < MIN_SELECTIONS or t["wins"] / t["selections"] >= MIN_WR

    for d_idx, date in enumerate(all_dates):
        if days_since == 0:
            recent = sig_dec[sig_dec["decision_date"] <= date]
            if recent.empty:
                current_strategies = []
            else:
                latest_dt = recent["decision_date"].max()
                candidates = recent[recent["decision_date"] == latest_dt].copy()

                if qf:
                    eligible = candidates[candidates["strategy"].apply(_eligible)]
                    if eligible.empty:
                        eligible = candidates
                else:
                    eligible = candidates

                eligible = eligible.sort_values("mean_excess", ascending=False)
                current_strategies = eligible["strategy"].head(top_k).tolist()

        # Compute portfolio return: equal-weighted across selected strategies
        if current_strategies:
            rets = []
            for s in current_strategies:
                if s in daily_ret.columns:
                    sr = daily_ret.loc[date, s]
                    rets.append(sr if pd.notna(sr) else hs300_ret.loc[date])
            daily_ret_val = np.mean(rets) if rets else hs300_ret.loc[date]
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
            "n_strategies": len(current_strategies),
            "strategies": "|".join(current_strategies) if current_strategies else "BENCH",
            "portfolio_nav": portfolio_nav[-1],
        })
        days_since = (days_since + 1) % N

        if days_since == 0 and current_strategies:
            block_excess = pd.Series(excess_ret_list[-N:]).sum()
            for s in current_strategies:
                track[s]["selections"] += 1
                if block_excess > 0:
                    track[s]["wins"] += 1

    result = pd.DataFrame(daily_records)

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
    print(f"  Backtest top_k={top_k} qf={'ON' if qf else 'OFF'}")
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
        sub2 = sub.copy()
        sub2["bid"] = sub2.index // N
        bw2 = (sub2.groupby("bid")["excess_return"].apply(lambda x: (1 + x).prod() - 1) > 0).mean()
        print(f"  {label:18s}  ann={a:+.2%}  daily_WR={(e>0).mean():.1%}  block_WR={bw2:.1%}")

    tag = f"knn_k{top_k}{'_qf' if qf else ''}"
    result.to_csv(OUT_DIR / f"backtest_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[DONE] backtest_{tag}.csv")
    print(f"\n  Top strategies by block WR:")
    for s, t in sorted(track.items(), key=lambda x: -x[1]["selections"])[:15]:
        w = t["wins"] / t["selections"] if t["selections"] > 0 else 0
        el = "E" if (t["selections"] >= MIN_SELECTIONS and w >= MIN_WR) else ("-" if t["selections"] >= MIN_SELECTIONS else "?")
        print(f"    {el} {s:35s}  sel={t['selections']:3d}  WR={w:.0%}")


if __name__ == "__main__":
    main()
