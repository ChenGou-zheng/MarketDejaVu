"""
MTM 净值重建正确性检验脚本

检查层级:
  L1: 自动化完整性检查 (NAV>0, 收益边界, 现金非负, 数据间隙, 首日校对)
  L2: 交易级抽样审计 (随机抽 10 个交易日逐日验证 NAV 公式)
  L3: 外部交叉验证 (ETF 价格比对)
  L4: 统计合理性检验 (极端收益溯源, 分布分析)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NAV_DIR = PROCESSED_DIR / "strategy_nav"
STOCK_PRICES_PATH = PROJECT_ROOT / "data" / "external" / "stock_prices.parquet"
ETF_PRICES_PATH = PROJECT_ROOT / "data" / "external" / "etf_prices.parquet"
OUT_DIR = PROJECT_ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPOT_CHECKS = 10
MAX_DAILY_RETURN_STOCK = 0.105  # stock limit + small tolerance
MAX_DAILY_RETURN_ETF = 0.21     # ETF limit + small tolerance
CASH_TOLERANCE = 1.0            # 1 yuan tolerance for NAV recompute

# ── Helper ──


def to_rqdata(symbol: str) -> str:
    if symbol.startswith("SHSE."):
        return symbol.replace("SHSE.", "") + ".XSHG"
    elif symbol.startswith("SZSE."):
        return symbol.replace("SZSE.", "") + ".XSHE"
    return symbol


# ═══════════════════════════════════════════
#  L1: 自动化完整性检查
# ═══════════════════════════════════════════


def check_nav_positive(nav: pd.DataFrame) -> tuple[bool, str]:
    invalid = (nav["nav"] <= 0).sum()
    if invalid > 0:
        return False, f"{invalid} days with NAV <= 0"
    return True, ""


def check_return_bounds(nav: pd.DataFrame, is_etf_heavy: bool) -> tuple[bool, str]:
    ret = nav["daily_return"].dropna()
    limit = MAX_DAILY_RETURN_ETF if is_etf_heavy else MAX_DAILY_RETURN_STOCK
    violations = (ret.abs() > limit).sum()
    if violations > 0:
        pct = violations / len(ret) * 100
        return False, f"{violations} days ({pct:.2f}%) exceeding ±{limit:.0%}"
    return True, ""


def check_cash_non_negative(csv: pd.DataFrame) -> tuple[bool, str]:
    neg = (csv["cash_balance"] < 0).sum()
    if neg > 0:
        return False, f"{neg} rows with negative cash_balance"
    return True, ""


def check_position_consistency(csv: pd.DataFrame) -> tuple[bool, str]:
    errors = []
    for _, row in csv.iterrows():
        btype = str(row.get("btype", ""))
        try:
            vol = float(row.get("volume", 0))
            posi = float(row.get("posi_balance", 0))
        except (ValueError, TypeError):
            continue
        if btype == "卖出平仓" and vol < 0 and posi < 0:
            errors.append(f"{row['trade_time']}: sold but posi={posi}")
        if btype == "买入开仓" and vol > 0 and posi <= 0:
            errors.append(f"{row['trade_time']}: bought but posi={posi}")
    if errors:
        return False, f"{len(errors)} position inconsistencies (e.g., {errors[0]})"
    return True, ""


def check_data_gaps(nav: pd.DataFrame) -> tuple[bool, str]:
    dates = pd.to_datetime(nav["trade_date"])
    gaps = dates.diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta(days=5)]
    if len(large_gaps) > 0:
        gap_details = []
        for idx in large_gaps.index[:3]:
            gap_date = dates.loc[idx]
            gap_days = large_gaps.loc[idx].days
            gap_details.append(f"{gap_date.date()}: {gap_days}d")
        return False, f"{len(large_gaps)} gaps >5d ({'; '.join(gap_details)})"
    return True, ""


def rebuild_state_at(csv: pd.DataFrame, target_date_str: str) -> tuple[float, dict[str, float]]:
    """Replay all CSV trades up to (and including) target_date, return (cash, positions).
    
    This mirrors rebuild_mtm_nav.py's logic.
    """
    initial_capital = None
    for _, row in csv.iterrows():
        if str(row.get("btype", "")) == "银证转入":
            try:
                initial_capital = float(row["cash_balance"])
            except (ValueError, TypeError):
                initial_capital = 0
            break
    if initial_capital is None:
        return 0.0, {}

    target_ts = pd.Timestamp(target_date_str)
    csv_sorted = csv.sort_values("trade_time")

    last_cash = initial_capital
    positions: dict[str, float] = {}
    for _, row in csv_sorted.iterrows():
        trade_ts = pd.to_datetime(row["trade_time"], errors="coerce")
        if pd.isna(trade_ts):
            continue
        if trade_ts.date() > target_ts.date():
            break
        try:
            cash = float(row["cash_balance"])
            btype = str(row.get("btype", ""))
            sym = str(row.get("symbol", ""))
            volume = float(row.get("volume", 0))
        except (ValueError, TypeError):
            continue

        if btype == "银证转入":
            last_cash = cash
            continue
        if pd.isna(sym) or sym == "nan":
            continue

        rq = to_rqdata(sym)
        old_shares = positions.get(rq, 0.0)
        new_shares = max(0.0, old_shares + volume)
        if new_shares > 0:
            positions[rq] = new_shares
        elif rq in positions:
            del positions[rq]
        last_cash = cash

    return last_cash, positions


def compute_mv(positions: dict[str, float], date: pd.Timestamp, stock_prices: pd.DataFrame) -> float:
    mv = 0.0
    for rq, shares in positions.items():
        if rq in stock_prices.columns and date in stock_prices.index:
            price = stock_prices.loc[date, rq]
            if pd.notna(price):
                mv += shares * price
    return mv


def check_first_day_nav(
    csv: pd.DataFrame, nav: pd.DataFrame, stock_prices: pd.DataFrame
) -> tuple[bool, str]:
    """Verify NAV[0] = cash + Σ(shares × close_price)."""
    first_date = str(pd.to_datetime(nav["trade_date"].iloc[0]).date())[:10]
    first_nav_ts = pd.Timestamp(first_date)

    cash, positions = rebuild_state_at(csv, first_date)

    mv = compute_mv(positions, first_nav_ts, stock_prices)
    expected = cash + mv
    actual = nav.iloc[0]["nav"]
    diff = abs(expected - actual)
    if diff > CASH_TOLERANCE:
        return False, f"NAV mismatch: expected={expected:.2f} actual={actual:.2f} diff={diff:.2f}"
    return True, f"OK (expected={expected:.2f}, actual={actual:.2f})"


# ═══════════════════════════════════════════
#  L2: 交易级抽样审计
# ═══════════════════════════════════════════


def spot_check_trades(
    csv: pd.DataFrame, nav: pd.DataFrame, stock_prices: pd.DataFrame, n: int = N_SPOT_CHECKS
) -> dict:
    """Randomly sample n trade days, verify NAV formula using cumulative state replay."""
    rng = np.random.default_rng(42)
    trade_mask = (csv["btype"] != "银证转入") & csv["symbol"].notna()
    trade_times = pd.to_datetime(csv.loc[trade_mask, "trade_time"], errors="coerce").dropna()
    trade_dates = sorted(set(trade_times.dt.date))

    if len(trade_dates) < n:
        n = len(trade_dates)

    sampled = rng.choice(list(trade_dates), size=n, replace=False)
    results = []
    nav_map = {}
    nav_dates = pd.to_datetime(nav["trade_date"])
    for i in range(len(nav)):
        nav_map[nav_dates.iloc[i].date()] = nav.iloc[i]["nav"]

    for date in sampled:
        date_str = str(date)
        ts = pd.Timestamp(date)
        cash, positions = rebuild_state_at(csv, date_str)
        mv = compute_mv(positions, ts, stock_prices)
        expected = cash + mv

        if date in nav_map:
            actual = nav_map[date]
        else:
            results.append({
                "date": str(date), "expected": round(expected, 2),
                "actual": None, "diff": None,
                "missing_prices": 0, "pass": False,
                "note": "date not in NAV",
            })
            continue

        diff = expected - actual
        passed = abs(diff) <= CASH_TOLERANCE
        results.append({
            "date": str(date),
            "expected": round(expected, 2),
            "actual": round(actual, 2),
            "diff": round(diff, 2),
            "missing_prices": sum(1 for rq in positions if rq not in stock_prices.columns),
            "pass": passed,
            "note": "" if passed else "NAV mismatch",
        })

    n_pass = sum(r["pass"] for r in results)
    return {
        "sampled": n,
        "passed": n_pass,
        "failed": n - n_pass,
        "pass_rate": f"{n_pass}/{n}",
        "details": results,
    }


# ═══════════════════════════════════════════
#  L3: ETF 价格交叉验证
# ═══════════════════════════════════════════


def check_etf_correlation(
    key: str, nav: pd.DataFrame, etf_prices: pd.DataFrame
) -> tuple[bool, str]:
    """If strategy key contains known ETF names, compare with ETF price."""
    known_etfs = {
        "511260": "国债ETF",
        "518880": "黄金ETF",
        "512890": "红利低波ETF",
        "515220": "煤炭ETF",
        "510050": "上证50ETF",
        "510300": "沪深300ETF",
        "159915": "创业板ETF",
    }

    csv_path = PROCESSED_DIR / f"{key.replace('/', '_')}.csv"
    if not csv_path.exists():
        return True, "no CSV for ETF check"

    csv = pd.read_csv(csv_path)
    # find most held ETF symbol
    etf_holdings = {}
    for _, row in csv.iterrows():
        sym = str(row.get("symbol", ""))
        if pd.isna(sym):
            continue
        rq = to_rqdata(sym)
        code = sym.split(".")[-1] if "." in sym else sym
        if code in known_etfs:
            etf_holdings[rq] = etf_holdings.get(rq, 0) + abs(row.get("volume", 0))

    if not etf_holdings:
        return True, "no known ETFs in this strategy"

    best_etf = max(etf_holdings, key=etf_holdings.get)
    if best_etf not in etf_prices.columns:
        return True, f"{best_etf} not in etf_prices"

    # compute NAV daily return and ETF daily return
    nav_ret = nav.set_index("trade_date")["daily_return"].dropna()
    etf_close = etf_prices[best_etf].dropna()
    etf_ret = etf_close.pct_change().dropna()

    common = nav_ret.index.intersection(etf_ret.index)
    if len(common) < 20:
        return True, f"only {len(common)} overlapping days"

    corr = nav_ret.loc[common].corr(etf_ret.loc[common])
    if corr < 0.1:
        return False, f"low correlation with {best_etf}: r={corr:.3f}"
    return True, f"correlation with {best_etf}: r={corr:.3f}"


# ═══════════════════════════════════════════
#  L4: 统计合理性检验
# ═══════════════════════════════════════════


def analyze_extreme_returns(
    key: str, nav: pd.DataFrame, csv: pd.DataFrame, stock_prices: pd.DataFrame
) -> dict:
    """For extreme return days, identify cause."""
    ret = nav["daily_return"].dropna()
    threshold = 0.10
    extreme_mask = ret.abs() > threshold
    extreme_days = ret[extreme_mask].index

    report = {
        "n_extreme": len(extreme_days),
        "extreme_rate": f"{len(extreme_days) / len(ret):.2%}",
        "max_daily": f"{ret.max():+.2%}",
        "min_daily": f"{ret.min():+.2%}",
        "top_5": [],
    }

    # Trace top 5 extreme days
    top_extreme = ret.abs().sort_values(ascending=False).head(5)
    for idx in top_extreme.index:
        date = nav.iloc[idx]["trade_date"]
        date_str = str(pd.to_datetime(date).date())[:10]

        # check if there were trades on this day
        day_trades = csv[csv["trade_time"].str.startswith(date_str, na=False)]
        trade_summary = ""
        if not day_trades.empty:
            btypes = day_trades["btype"].value_counts().to_dict()
            trade_summary = "; ".join(f"{k}:{v}" for k, v in btypes.items())

        # check if price change explains the return
        nav_change = nav.iloc[idx]["daily_return"]

        report["top_5"].append({
            "date": date_str,
            "return": f"{nav_change:+.2%}",
            "trades": trade_summary or "none",
        })

    return report


def return_distribution_stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    return {
        "n_days": len(ret),
        "mean": f"{ret.mean():+.4%}",
        "std": f"{ret.std():.2%}",
        "ann_vol": f"{ret.std() * np.sqrt(252):.0%}",
        "skew": f"{ret.skew():.2f}",
        "kurtosis": f"{ret.kurtosis():.2f}",
        "p1": f"{ret.quantile(0.01):+.2%}",
        "p99": f"{ret.quantile(0.99):+.2%}",
        "extreme_gt_20pct": f"{(ret.abs() > 0.2).mean():.2%}",
    }


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════


def validate_strategy(
    key: str, stock_prices: pd.DataFrame, etf_prices: pd.DataFrame
) -> dict:
    safe = key.replace("/", "_")
    csv_path = PROCESSED_DIR / f"{safe}.csv"
    nav_path = NAV_DIR / f"{safe}_nav.csv"

    report = {"strategy_key": key, "errors": [], "warnings": [], "n_errors": 0, "n_warnings": 0}

    if not csv_path.exists():
        report["errors"].append("CSV not found")
        report["n_errors"] = 1
        report["overall"] = "SKIP"
        return report
    if not nav_path.exists():
        report["errors"].append("NAV file not found (filtered out by gap/quality check)")
        report["n_errors"] = 1
        report["overall"] = "SKIP"
        return report

    csv = pd.read_csv(csv_path)
    nav = pd.read_csv(nav_path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])

    # Detect if strategy is ETF-heavy (holds mostly ETFs)
    symbols = csv["symbol"].dropna().unique()
    etf_count = sum(1 for s in symbols if str(s).startswith(("SHSE.51", "SHSE.58", "SZSE.15", "SZSE.16")))
    is_etf_heavy = etf_count / max(len(symbols), 1) > 0.5

    # ── L1 ──
    ok, msg = check_nav_positive(nav)
    if not ok:
        report["errors"].append(f"L1_NAV_POSITIVE: {msg}")
    report["nav_positive"] = "PASS" if ok else f"FAIL ({msg})"

    ok, msg = check_return_bounds(nav, is_etf_heavy)
    if not ok:
        report["errors"].append(f"L1_RETURN_BOUND: {msg}")
    report["return_bounds"] = "PASS" if ok else f"FAIL ({msg})"

    ok, msg = check_cash_non_negative(csv)
    if not ok:
        report["errors"].append(f"L1_CASH_NEGATIVE: {msg}")
    report["cash_non_negative"] = "PASS" if ok else f"FAIL ({msg})"

    ok, msg = check_position_consistency(csv)
    if not ok:
        report["warnings"].append(f"L1_POSITION: {msg}")
    report["position_consistency"] = "PASS" if ok else f"WARN ({msg})"

    ok, msg = check_data_gaps(nav)
    if not ok:
        report["warnings"].append(f"L1_DATA_GAP: {msg}")
    report["data_gaps"] = "PASS" if ok else f"WARN ({msg})"

    ok, msg = check_first_day_nav(csv, nav, stock_prices)
    if not ok:
        report["warnings"].append(f"L1_FIRST_DAY: {msg}")
    report["first_day_nav"] = "PASS" if ok else f"WARN ({msg})"

    # ── L2 ──
    spot = spot_check_trades(csv, nav, stock_prices, n=N_SPOT_CHECKS)
    if spot["failed"] > 0:
        report["errors"].append(f"L2_SPOT_CHECK: {spot['failed']}/{spot['sampled']} failed")
    report["spot_check"] = spot

    # ── L3 ──
    ok, msg = check_etf_correlation(key, nav, etf_prices)
    if not ok:
        report["warnings"].append(f"L3_ETF_CORR: {msg}")
    report["etf_correlation"] = "PASS" if ok else f"WARN ({msg})"

    # ── L4 ──
    ret = nav["daily_return"]
    report["return_stats"] = return_distribution_stats(ret)
    report["extreme_analysis"] = analyze_extreme_returns(key, nav, csv, stock_prices)

    # ── Overall ──
    report["n_errors"] = len(report["errors"])
    report["n_warnings"] = len(report["warnings"])
    report["overall"] = (
        "PASS"
        if report["n_errors"] == 0 and report["n_warnings"] <= 2
        else (
            "WARN"
            if report["n_errors"] == 0
            else "FAIL"
        )
    )

    return report


def main():
    print("=" * 70)
    print("  MTM NAV 净值重建正确性检验")
    print("=" * 70)

    # Load data
    print("\n[LOAD] stock prices...")
    stock_prices = pd.read_parquet(STOCK_PRICES_PATH)
    stock_prices.index = pd.to_datetime(stock_prices.index)
    print(f"       {len(stock_prices)} days x {len(stock_prices.columns)} stocks")

    print("[LOAD] ETF prices...")
    try:
        etf_prices = pd.read_parquet(ETF_PRICES_PATH)
        etf_prices.index = pd.to_datetime(etf_prices.index)
        print(f"       {len(etf_prices)} days x {len(etf_prices.columns)} ETFs")
    except FileNotFoundError:
        etf_prices = pd.DataFrame()
        print("       (not found, L3 skipped)")

    # Find all MTM strategies
    csvs = sorted(PROCESSED_DIR.glob("交易记录*.csv"))
    print(f"\n[SCAN] {len(csvs)} MTM strategy CSVs found\n")

    all_reports = []
    for csv_path in csvs:
        key = csv_path.stem
        print(f"  [{key}] ", end="", flush=True)
        report = validate_strategy(key, stock_prices, etf_prices)
        all_reports.append(report)

        status = report["overall"]
        n_err = report["n_errors"]
        n_warn = report["n_warnings"]

        if status == "PASS":
            print(f"✅ PASS  (warnings={n_warn})")
        elif status == "WARN":
            print(f"⚠️  WARN  (errors={n_err}, warnings={n_warn})")
        else:
            print(f"❌ FAIL  (errors={n_err}, warnings={n_warn})")
            for e in report["errors"]:
                print(f"       Error: {e}")
        for w in report["warnings"]:
            print(f"       Warn:  {w}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    n_pass = sum(1 for r in all_reports if r.get("overall") == "PASS")
    n_warn = sum(1 for r in all_reports if r.get("overall") == "WARN")
    n_fail = sum(1 for r in all_reports if r.get("overall") == "FAIL")
    n_skip = sum(1 for r in all_reports if r.get("overall") == "SKIP")
    print(f"  Total: {len(all_reports)}  PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIP={n_skip}")

    if n_fail > 0:
        print(f"\n  FAILED strategies:")
        for r in all_reports:
            if r.get("overall") == "FAIL":
                print(f"    {r['strategy_key']}: {r['errors']}")

    # ── Save report ──
    rows = []
    for r in all_reports:
        rows.append({
            "strategy_key": r["strategy_key"],
            "overall": r.get("overall", "?"),
            "n_errors": r.get("n_errors", 0),
            "n_warnings": r.get("n_warnings", 0),
            "nav_positive": r.get("nav_positive", ""),
            "return_bounds": r.get("return_bounds", ""),
            "cash_non_negative": r.get("cash_non_negative", ""),
            "position_consistency": r.get("position_consistency", ""),
            "data_gaps": r.get("data_gaps", ""),
            "first_day_nav": r.get("first_day_nav", ""),
            "etf_correlation": r.get("etf_correlation", ""),
            "extreme_rate": r.get("return_stats", {}).get("extreme_gt_20pct", ""),
            "ann_vol": r.get("return_stats", {}).get("ann_vol", ""),
            "max_daily": r.get("extreme_analysis", {}).get("max_daily", ""),
            "min_daily": r.get("extreme_analysis", {}).get("min_daily", ""),
        })

    # Spot check details (only for non-SKIP strategies)
    for r in all_reports:
        if r.get("overall") == "SKIP":
            continue
        sp = r.get("spot_check", {})
        if sp.get("details"):
            for d in sp["details"]:
                rows.append({
                    "strategy_key": f"{r['strategy_key']} (spot:{d['date']})",
                    "overall": "SPOT",
                    "n_errors": 0,
                    "n_warnings": 0,
                    "nav_positive": "",
                    "return_bounds": "",
                    "cash_non_negative": "",
                    "position_consistency": "",
                    "data_gaps": "",
                    "first_day_nav": "",
                    "etf_correlation": f"expected={d['expected']} actual={d['actual']} diff={d['diff']} pass={d['pass']}",
                    "extreme_rate": "",
                    "ann_vol": "",
                    "max_daily": "",
                    "min_daily": "",
                })

    summary_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "mtm_validation_report.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Report saved to {out_path}")

    # ── Print top 5 extreme events for each FAIL/WARN strategy ──
    print(f"\n{'='*70}")
    print("  EXTREME EVENTS (top 5 per strategy with high extreme rate)")
    print(f"{'='*70}")
    for r in all_reports:
        if r.get("overall") == "SKIP":
            continue
        ext = r.get("extreme_analysis", {})
        ext_rate_str = ext.get("extreme_rate", "0%")
        ext_rate_val = float(ext_rate_str.replace("%", "")) / 100 if "%" in ext_rate_str else 0
        if ext_rate_val > 0.03 or r.get("overall", "") != "PASS":
            print(f"\n  {r['strategy_key']}  (extreme_rate={ext_rate_str})")
            for ev in ext.get("top_5", []):
                print(f"    {ev['date']}  {ev['return']}  trades=[{ev['trades']}]")


if __name__ == "__main__":
    main()
