"""
Compute equal-weight baselines for each ablation pool × time period.

Output: console table + appended to ablation_results.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "processed" / "snapshot"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

N_DAYS = 20

TIME_PERIODS = {
    "T0(2020+)": "2020-01-01",
    "T1(2024+)": "2024-01-01",
    "T2(2025+)": "2025-01-01",
    "T3(2026+)": "2026-01-01",
}


def passes_qf(key: str) -> bool:
    df = pd.read_csv(NAV_DIR / f"{key}_nav.csv")
    ret = df["daily_return"].dropna()
    extreme = (ret.abs() > 0.2).mean()
    vol = ret.std() * np.sqrt(252)
    return extreme <= 0.03 and vol <= 2.0


def metrics_from_excess(es: pd.Series) -> dict:
    es = es.dropna()
    if len(es) < 10:
        return {"ann_excess": "-", "block_wr": "-", "max_dd": "-"}
    total = (1 + es).prod() - 1
    ann = (1 + total) ** (252 / len(es)) - 1
    n_b = len(es) // N_DAYS
    be = (1 + es.iloc[:n_b * N_DAYS].values.reshape(-1, N_DAYS)).prod(axis=1) - 1
    bw = (be > 0).mean()
    cum = (1 + es).cumprod()
    dd = (1 - cum / cum.cummax()).max()
    return {"ann_excess": f"{ann:+.2%}", "block_wr": f"{bw:.1%}", "max_dd": f"{dd:.2%}"}


def main():
    # Universe
    mtm_all = sorted(f.stem.replace("_nav", "") for f in NAV_DIR.glob("交易记录*_nav.csv"))
    mtm_good = [k for k in mtm_all if passes_qf(k)]
    etf_all = sorted(f.stem.replace("_nav", "") for f in NAV_DIR.glob("ETF_*_nav.csv"))

    pools = {
        "MTM-unfiltered": mtm_all,
        "MTM-filtered": mtm_good,
        "MTM-unfiltered+ETF": mtm_all + etf_all,
        "MTM-filtered+ETF": mtm_good + etf_all,
    }

    # Load returns and benchmark
    rr = pd.read_parquet(SNAPSHOT_DIR / "rebuilt_returns.parquet")
    bc = pd.read_parquet(SNAPSHOT_DIR / "benchmark_close.parquet")
    hs300_ret = bc["HS300_close"].pct_change().dropna()
    common_dates = rr.index.intersection(hs300_ret.index)

    all_rows = []
    print(f"\n{'='*70}")
    print("  Equal-Weight Baselines")
    print(f"{'='*70}\n")

    for pool_name, pool_keys in pools.items():
        pool_set = set(pool_keys)
        avail = [c for c in rr.columns if c in pool_set]
        if len(avail) == 0:
            continue

        # equal-weight portfolio return
        ew_ret = rr[avail].reindex(common_dates).mean(axis=1)
        ew_excess = ew_ret - hs300_ret.reindex(common_dates)

        print(f"  {pool_name} ({len(avail)} strategies):")

        row_data = {"pool": f"Equal-weight ({pool_name})"}
        for label, cutoff in TIME_PERIODS.items():
            sub = ew_excess[ew_excess.index >= cutoff]
            m = metrics_from_excess(sub)
            for k, v in m.items():
                row_data[f"{label}_{k}"] = v
            print(f"    {label}: ann={m['ann_excess']}  BW={m['block_wr']}  DD={m['max_dd']}")

        all_rows.append(row_data)

    # Save baseline results
    out_path = TABLES_DIR / "ablation_baselines.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Baselines saved to {out_path}")

    # ── Print combined comparison table ──
    print("\n\n" + "=" * 100)
    print("  COMBINED: KNN vs Equal-Weight × Time Period")
    print("=" * 100)

    tl = list(TIME_PERIODS.keys())
    header = f"{'':30s}" + "".join(f"{'| ' + t:>22s}" for t in tl)
    print(f"\n{header}")
    print("-" * len(header))

    # Load KNN results
    knn_df = pd.read_csv(TABLES_DIR / "ablation_results.csv")

    for pool_name in pools:
        # KNN row
        knn_rows = knn_df[knn_df["pool"] == pool_name]
        if not knn_rows.empty:
            row = f"KNN ({pool_name:20s})"
            for t in tl:
                r = knn_rows[knn_rows["period"] == t]
                if not r.empty:
                    cell = f"{r.iloc[0]['ann_excess']:>7s}/{r.iloc[0]['block_wr']:>5s}/{r.iloc[0]['max_dd']:>7s}"
                else:
                    cell = "   -/  -/     -"
                row += f"{'| ' + cell:>22s}"
            print(row)

        # Equal-weight row
        ew_label = f"Equal-weight ({pool_name})"
        ew_rows = [r for r in all_rows if r["pool"] == ew_label]
        if ew_rows:
            ew = ew_rows[0]
            row = f"EW    ({pool_name:20s})"
            for t in tl:
                ann = ew.get(f"{t}_ann_excess", "-")
                bw = ew.get(f"{t}_block_wr", "-")
                dd = ew.get(f"{t}_max_dd", "-")
                cell = f"{ann:>7s}/{bw:>5s}/{dd:>7s}"
                row += f"{'| ' + cell:>22s}"
            print(row)

        print("-" * len(header))


if __name__ == "__main__":
    main()
