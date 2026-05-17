"""
Ablation v3: KNN + Rank (no FDR) + N=10 + 综合评分排序.

Changes from v2:
  - Filter threshold restored to MAX_ANN_VOL=5.0
  - 方案F: ranking score = mean_excess × (1 - max_dd) × block_wr_1y
  - All 4 pools × 4 time periods
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.similarity_engine import (
    weighted_mean,
    FEATURE_WEIGHTS,
    K,
    MIN_K,
    HALFLIFE_YEARS,
)

NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "processed" / "snapshot"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 10
TOP_K = 3

TIME_PERIODS = {
    "T0(2020+)": "2020-01-01",
    "T1(2024+)": "2024-01-01",
    "T2(2025+)": "2025-01-01",
    "T3(2026+)": "2026-01-01",
}


def passes_qf(key: str) -> bool:
    df = pd.read_csv(NAV_DIR / f"{key}_nav.csv")
    ret = df["daily_return"].dropna()
    return (ret.abs() > 0.2).mean() <= 0.03 and ret.std() * np.sqrt(252) <= 5.0


def load_nav(key: str) -> pd.Series | None:
    safe = key.replace("/", "_").replace("\\", "_")
    path = NAV_DIR / f"{safe}_nav.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "trade_date" not in df.columns or "nav" not in df.columns:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    nav = df.set_index("trade_date")["nav"].sort_index()
    return nav


def compute_quality(
    strategy_keys: list[str], all_dates, bench_nav
) -> dict[str, dict]:
    """Pre-compute max_dd and 1y block WR for each strategy."""
    bc = pd.read_parquet(SNAPSHOT_DIR / "benchmark_close.parquet")
    hs300 = bc["HS300_close"]
    hs300_nav = hs300 / hs300.iloc[0]
    bench_fwd = hs300_nav.pct_change(N_DAYS).shift(-N_DAYS)

    quality = {}
    for key in strategy_keys:
        nav = load_nav(key)
        if nav is None:
            continue
        nav_filled = nav.reindex(all_dates, method="ffill").dropna()
        if len(nav_filled) < 120:
            continue
        nav_norm = nav_filled / nav_filled.iloc[0]

        # max drawdown (full history)
        cummax = nav_filled.expanding().max()  # using absolute NAV for DD
        dd_series = nav_filled / cummax - 1
        max_dd = abs(dd_series.min())

        # 1y block WR: last ~252 trading days
        last_year_start = nav_filled.index[-1] - pd.Timedelta(days=400)
        ly = nav_filled[nav_filled.index >= last_year_start]
        if len(ly) < 60:
            block_wr = 0.5
        else:
            ly_norm = ly / ly.iloc[0]
            ly_fwd = ly_norm.pct_change(N_DAYS).shift(-N_DAYS)
            ly_bench = bench_fwd.reindex(ly_fwd.index)
            ly_excess = (ly_fwd - ly_bench).dropna()
            # block WR: each N_DAYS block
            n_blocks = len(ly_excess) // N_DAYS
            if n_blocks >= 2:
                blocks = ly_excess.iloc[:n_blocks * N_DAYS].values.reshape(-1, N_DAYS)
                block_rets = (1 + blocks).prod(axis=1) - 1
                block_wr = (block_rets > 0).mean()
            else:
                block_wr = 0.5

        quality[key] = {"max_dd": max_dd, "block_wr_1y": block_wr}

    return quality


def make_labels(strategy_keys: list[str], all_dates, bench_fwd) -> pd.DataFrame:
    excess_list, valid_keys = [], []
    for key in strategy_keys:
        nav = load_nav(key)
        if nav is None:
            continue
        nav_filled = nav.reindex(all_dates, method="ffill").dropna()
        nav_norm = nav_filled / nav_filled.iloc[0]
        strat_fwd = nav_norm.pct_change(N_DAYS).shift(-N_DAYS)
        bench_aligned = bench_fwd.reindex(strat_fwd.index)
        excess = (strat_fwd - bench_aligned).clip(-3.0, 3.0)
        if excess.notna().sum() < 60:
            continue
        excess.name = key
        excess_list.append(excess)
        valid_keys.append(key)
    if not excess_list:
        return pd.DataFrame()
    return pd.DataFrame(excess_list).T


def run_similarity(
    features: pd.DataFrame, labels: pd.DataFrame, strategy_quality: dict
) -> pd.DataFrame:
    """KNN + rank by comprehensive score (方案F)."""
    pos = pd.Series(range(len(features)), index=features.index)
    weights = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in features.columns])
    halflife_days = int(HALFLIFE_YEARS * 252)
    lambda_decay = np.log(2) / halflife_days
    strategy_names = list(labels.columns)
    decision_rows = []

    for i, t in enumerate(features.index):
        if pos[t] < N_DAYS:
            continue
        avail_mask = pos < (pos[t] - N_DAYS)
        avail_idx = features.index[avail_mask.values]
        n_avail = len(avail_idx)
        if n_avail < MIN_K:
            continue

        F_avail = features.loc[avail_idx]
        mu, sigma = F_avail.mean(), F_avail.std().replace(0, np.nan)
        F_std = ((F_avail - mu) / sigma).values
        f_t = ((features.loc[t] - mu) / sigma).values

        diff = F_std - f_t
        dist = np.sqrt((diff ** 2 * weights).sum(axis=1))

        k_actual = min(K, n_avail)
        nearest_idx = np.argpartition(dist, k_actual - 1)[:k_actual]
        nearest_idx = nearest_idx[dist[nearest_idx].argsort()]
        neigh_dates = avail_idx[nearest_idx]

        time_decay_w = np.exp(-lambda_decay * (pos[t] - pos[neigh_dates].values))
        time_decay_w = time_decay_w / time_decay_w.sum() * k_actual

        label_values = labels.loc[neigh_dates, strategy_names].values
        strategy_results = []
        for s_idx, s_name in enumerate(strategy_names):
            x = label_values[:, s_idx]
            valid = ~np.isnan(x)
            if valid.sum() < 3:
                continue
            x_v, w_v = x[valid], time_decay_w[valid]
            w_v = w_v / w_v.sum() * len(x_v)
            mu_excess = weighted_mean(x_v, w_v)

            # 方案F: comprehensive score
            # Score = mean_excess × (1 + block_wr_1y - max_dd)
            # block_wr > 0.5 → bonus, < 0.5 → penalty
            # (1 - max_dd) alone is too harsh; this additive form is gentler
            q = strategy_quality.get(s_name, {"max_dd": 0.5, "block_wr_1y": 0.5})
            max_dd = q["max_dd"]
            block_wr = q["block_wr_1y"]
            score = mu_excess * (1.0 + block_wr - max_dd * 0.5)

            strategy_results.append({
                "strategy": s_name,
                "mean_excess": mu_excess,
                "score": score,
                "max_dd": max_dd,
                "block_wr_1y": block_wr,
                "is_significant": True,
                "decision_date": t,
            })

        if strategy_results:
            decision_rows.append(pd.DataFrame(strategy_results))

    return pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()


def run_backtest(decisions: pd.DataFrame, returns: pd.DataFrame,
                 hs300_ret: pd.Series) -> pd.DataFrame:
    strategies = [c for c in returns.columns]
    all_dates = returns.index.intersection(hs300_ret.index)
    all_dates = all_dates[all_dates >= decisions["decision_date"].min()]

    from collections import defaultdict
    track = defaultdict(lambda: {"selections": 0, "wins": 0})
    daily_records = []
    current_strategies = []
    days_since = 0
    MIN_SEL, MIN_WR = 3, 0.50

    def _eligible(s):
        t = track.get(s, {"selections": 0, "wins": 0})
        return t["selections"] < MIN_SEL or t["wins"] / max(t["selections"], 1) >= MIN_WR

    for date in all_dates:
        if days_since == 0:
            recent = decisions[decisions["decision_date"] <= date]
            if recent.empty:
                current_strategies = []
            else:
                latest_dt = recent["decision_date"].max()
                candidates = recent[recent["decision_date"] == latest_dt].copy()
                eligible = candidates[candidates["strategy"].apply(_eligible)]
                if eligible.empty:
                    eligible = candidates
                current_strategies = eligible.sort_values("score", ascending=False)["strategy"].head(TOP_K).tolist()

        if current_strategies:
            rets = []
            for s in current_strategies:
                if s in returns.columns:
                    sr = returns.loc[date, s]
                    rets.append(sr if pd.notna(sr) else hs300_ret.loc[date])
            daily_ret = np.mean(rets) if rets else hs300_ret.loc[date]
        else:
            daily_ret = hs300_ret.loc[date]

        excess = daily_ret - hs300_ret.loc[date]
        daily_records.append({"trade_date": date, "excess_return": excess})
        days_since = (days_since + 1) % N_DAYS

        if days_since == 0 and current_strategies:
            block_ex = sum(r["excess_return"] for r in daily_records[-N_DAYS:])
            for s in current_strategies:
                track[s]["selections"] += 1
                if block_ex > 0:
                    track[s]["wins"] += 1

    return pd.DataFrame(daily_records)


def metrics(bt: pd.DataFrame, cutoff: str):
    sub = bt[bt["trade_date"] >= cutoff]
    if len(sub) < 10:
        return {"ann_excess": "-", "ir": "-", "block_wr": "-", "max_dd": "-"}
    es = sub["excess_return"].dropna()
    total = (1 + es).prod() - 1
    ann = (1 + total) ** (252 / len(es)) - 1
    te = es.std() * np.sqrt(252)
    ir = ann / te if te > 0 else 0.0
    n_b = len(es) // N_DAYS
    be = (1 + es.iloc[:n_b * N_DAYS].values.reshape(-1, N_DAYS)).prod(axis=1) - 1
    bw = (be > 0).mean()
    dd = (1 - (1 + es).cumprod() / (1 + es).cumprod().cummax()).max()
    return {"ann_excess": f"{ann:+.2%}", "ir": f"{ir:.3f}", "block_wr": f"{bw:.1%}", "max_dd": f"{dd:.2%}"}


def main():
    print("=" * 70)
    print("  Ablation v3: KNN+Rank + N=10 + 综合评分(方案F)")
    print("=" * 70)

    features = pd.read_parquet(SNAPSHOT_DIR / "features.parquet")
    bc = pd.read_parquet(SNAPSHOT_DIR / "benchmark_close.parquet")
    hs300 = bc["HS300_close"]
    hs300_nav = hs300 / hs300.iloc[0]
    bench_fwd = hs300_nav.pct_change(N_DAYS).shift(-N_DAYS)
    all_dates = features.index

    mtm_all = sorted(f.stem.replace("_nav", "") for f in NAV_DIR.glob("交易记录*_nav.csv"))
    mtm_good = [k for k in mtm_all if passes_qf(k)]
    etf_all = sorted(f.stem.replace("_nav", "") for f in NAV_DIR.glob("ETF_*_nav.csv"))

    print(f"\nMTM all={len(mtm_all)}  good={len(mtm_good)}  ETF={len(etf_all)}")

    pools = {
        "MTM-unfiltered": mtm_all,
        "MTM-filtered": mtm_good,
        "MTM-unfiltered+ETF": mtm_all + etf_all,
        "MTM-filtered+ETF": mtm_good + etf_all,
    }

    all_keys = mtm_all + etf_all

    # Pre-compute quality metrics
    print(f"\nQuality metrics for {len(all_keys)} strategies...")
    strategy_quality = compute_quality(all_keys, all_dates, hs300_nav)
    print(f"  Loaded quality for {len(strategy_quality)} strategies")
    for k, v in sorted(strategy_quality.items(), key=lambda x: -x[1]["score"] if "score" in x[1] else 0)[:5]:
        print(f"    {k:25s}  max_dd={v['max_dd']:.2%}  block_wr_1y={v['block_wr_1y']:.1%}")

    print(f"\nLabels for {len(all_keys)} strategies...")
    full_labels = make_labels(all_keys, all_dates, bench_fwd)
    print(f"  Shape: {full_labels.shape}")

    rr = pd.read_parquet(SNAPSHOT_DIR / "rebuilt_returns.parquet")
    hs300_ret = hs300.pct_change().dropna()

    all_results = {}
    for pool_name, pool_keys in pools.items():
        pool_set = set(pool_keys)
        cols = [c for c in full_labels.columns if c in pool_set]
        print(f"\n{'='*55}\n  POOL: {pool_name} ({len(cols)} strategies)\n{'='*55}")
        if len(cols) < 3:
            continue

        pl = full_labels[cols].dropna(how="all", axis=1)
        if pl.shape[1] < 3:
            continue
        common = features.index.intersection(pl.index)
        pl, pf = pl.loc[common], features.loc[common]

        decisions = run_similarity(pf, pl, strategy_quality)
        if decisions.empty:
            continue

        pool_rets = rr[[c for c in rr.columns if c in pool_set]]
        bt = run_backtest(decisions, pool_rets, hs300_ret)

        res = {}
        for label, cutoff in TIME_PERIODS.items():
            res[label] = metrics(bt, cutoff)
            m = res[label]
            print(f"  {label}: ann={m['ann_excess']}  IR={m['ir']}  BW={m['block_wr']}  DD={m['max_dd']}")

        all_results[pool_name] = res
        bt.to_csv(TABLES_DIR / f"ablation_v3_{pool_name.replace('+','_')}.csv", index=False)

    # Print table
    print("\n\n" + "=" * 90)
    print("  KNN+Rank+N=10+方案F  |  Ablation × Time Period")
    print("=" * 90)
    tl = list(TIME_PERIODS.keys())
    header = f"{'Pool':30s}" + "".join(f"{'| ' + t:>22s}" for t in tl)
    print("\n" + header)
    print("-" * len(header))
    for pn in pools:
        if pn not in all_results:
            continue
        r = all_results[pn]
        row = f"{pn:30s}"
        for t in tl:
            m = r.get(t, {})
            cell = f"{m.get('ann_excess','-'):>7s}/{m.get('block_wr','-'):>5s}/{m.get('max_dd','-'):>7s}"
            row += f"{'| ' + cell:>22s}"
        print(row)

    rows = []
    for pn, pres in all_results.items():
        for t, m in pres.items():
            rows.append({"pool": pn, "period": t, **m})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "ablation_v3_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Saved to output/tables/ablation_v3_results.csv")


if __name__ == "__main__":
    main()
