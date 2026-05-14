"""
全量数据拉取模块（rqdatac + AKShare）。

输出: data/external/ 下各 CSV 文件
用法:
    python src/data/fetcher.py              # 拉取全部
    python src/data/fetcher.py --force      # 强制重新拉取
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2016-01-08"
END_DATE = "2026-04-30"

# ── 缓存 ──
def _path(name: str) -> Path:
    return EXTERNAL_DIR / f"{name}.parquet"

def _load(name: str) -> pd.DataFrame | None:
    p = _path(name)
    if p.exists():
        return pd.read_parquet(p)
    return None

def _save(name: str, df: pd.DataFrame) -> None:
    df.to_parquet(_path(name))

# ═══════════════════════════════════════════
#  B1: 宽基指数 OHLCV (rqdatac)
# ═══════════════════════════════════════════
def fetch_index_prices(force: bool = False) -> pd.DataFrame:
    name = "index_prices"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    import rqdatac
    rqdatac.init()

    symbols = {"000300.XSHG": "HS300", "000905.XSHG": "ZZ500", "399006.XSHE": "CYB"}
    dfs = []
    for code, label in symbols.items():
        df = rqdatac.get_price(code, start_date=START_DATE, end_date=END_DATE,
                               fields=["open", "high", "low", "close", "volume"])
        df = df.rename(columns={c: f"{label}_{c}" for c in df.columns})
        # Normalize index
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(-1)
        df.index = pd.to_datetime(df.index)
        df.index.name = None
        dfs.append(df)
    merged = pd.concat(dfs, axis=1).sort_index()
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = None
    _save(name, merged)
    print(f"[OK] {name}: {len(merged)} rows, cols={list(merged.columns)}")
    return merged


# ═══════════════════════════════════════════
#  B5: 收益率曲线 (rqdatac)
# ═══════════════════════════════════════════
def fetch_yield_curve(force: bool = False) -> pd.DataFrame:
    name = "yield_curve"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    import rqdatac
    rqdatac.init()
    df = rqdatac.get_yield_curve(START_DATE, END_DATE)
    keep = [c for c in ["1Y", "10Y"] if c in df.columns]
    df = df[keep].copy()
    if "1Y" in df and "10Y" in df:
        df["yield_slope"] = df["10Y"] - df["1Y"]
    df = df.sort_index()
    _save(name, df)
    print(f"[OK] {name}: {len(df)} rows, cols={list(df.columns)}")
    return df


# ═══════════════════════════════════════════
#  B2: 北向资金 (AKShare)
# ═══════════════════════════════════════════
def fetch_northbound_flow(force: bool = False) -> pd.DataFrame:
    name = "northbound_flow"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    try:
        import akshare as ak
        # 沪股通 + 深股通 分别拉取后合并
        parts = []
        for sym in ["沪股通", "深股通"]:
            df = ak.stock_hsgt_hist_em(symbol=sym)
            df = df.rename(columns={"日期": "trade_date", "当日成交净买额": f"net_{sym}"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            parts.append(df.set_index("trade_date")[[f"net_{sym}"]])
        merged = pd.concat(parts, axis=1)
        merged.columns = ["net_sh", "net_sz"]  # standardize names
        merged["net_total"] = merged.sum(axis=1)
        merged = merged.loc[START_DATE:END_DATE].sort_index()
        _save(name, merged)
        print(f"[OK] {name}: {len(merged)} rows (AKShare)")
        return merged
    except Exception as e:
        print(f"[WARN] {name}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
#  B2: 融资融券 (AKShare)
# ═══════════════════════════════════════════
def fetch_margin(force: bool = False) -> pd.DataFrame:
    name = "margin"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    try:
        import akshare as ak
        df = ak.stock_margin_sse(start_date="20160104", end_date="20260430")
        df = df.rename(columns={"信用交易日期": "trade_date", "融资余额": "margin_balance"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")[["margin_balance"]].sort_index()
        _save(name, df)
        print(f"[OK] {name}: {len(df)} rows (AKShare)")
        return df
    except Exception as e:
        print(f"[WARN] {name}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
#  B5: 宏观经济 (AKShare)
# ═══════════════════════════════════════════
def fetch_macro(force: bool = False) -> pd.DataFrame:
    name = "macro"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    import akshare as ak
    parts = {}

    # PMI — 月度
    try:
        df = ak.macro_china_pmi()
        # columns: 月份, 制造业-指数, 制造业-同比增长, 非制造业-指数, 非制造业-同比增长
        df = df.rename(columns={"月份": "trade_date", "制造业-指数": "pmi_mfg"})
        df["trade_date"] = df["trade_date"].str.replace("年|月份", "", regex=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m", errors="coerce")
        df = df.dropna(subset=["trade_date"])
        parts["pmi"] = df.set_index("trade_date")[["pmi_mfg"]]
        print(f"  [OK] PMI: {len(parts['pmi'])} rows")
    except Exception as e:
        print(f"  [WARN] PMI: {e}")

    # CPI — 月度
    try:
        df = ak.macro_china_cpi_monthly()
        # columns: 商品, 日期, 今值, 预测值, 前值
        df = df.rename(columns={"日期": "trade_date", "今值": "cpi_yoy"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        parts["cpi"] = df.set_index("trade_date")[["cpi_yoy"]]
        print(f"  [OK] CPI: {len(parts['cpi'])} rows")
    except Exception as e:
        print(f"  [WARN] CPI: {e}")

    # M2
    try:
        df = ak.macro_china_money_supply()
        # 月份, 货币和准货币(M2)-数量(亿元), 货币和准货币(M2)-同比增长, ...
        m2_col = [c for c in df.columns if "M2" in c and "同比增长" in c][0]
        df = df.rename(columns={"月份": "trade_date", m2_col: "m2_yoy"})
        df["trade_date"] = df["trade_date"].str.replace("年|月份", "", regex=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m", errors="coerce")
        df = df.dropna(subset=["trade_date"])
        parts["m2"] = df.set_index("trade_date")[["m2_yoy"]]
        print(f"  [OK] M2: {len(parts['m2'])} rows")
    except Exception as e:
        print(f"  [WARN] M2: {e}")

    # PPI
    try:
        df = ak.macro_china_ppi_yearly()
        df = df.rename(columns={"日期": "trade_date", "今值": "ppi_yoy"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        parts["ppi"] = df.set_index("trade_date")[["ppi_yoy"]]
        print(f"  [OK] PPI: {len(parts['ppi'])} rows")
    except Exception as e:
        print(f"  [WARN] PPI: {e}")

    if parts:
        merged = pd.concat(parts.values(), axis=1).sort_index()
        _save(name, merged)
        print(f"[OK] {name}: {len(merged)} rows, cols={list(merged.columns)}")
        return merged
    return pd.DataFrame()


# ═══════════════════════════════════════════
#  B4: 申万行业指数 (AKShare)
# ═══════════════════════════════════════════
def fetch_sw_industry(force: bool = False) -> pd.DataFrame:
    name = "sw_industry"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    try:
        import akshare as ak
        # 31 个申万一级行业代码
        sw_codes = [
            "801010","801020","801030","801040","801050","801080","801110","801120",
            "801130","801140","801150","801160","801170","801180","801200","801210",
            "801230","801250","801260","801270","801280","801710","801720","801730",
            "801740","801750","801760","801770","801780","801790","801880",
        ]
        closes = {}
        for code in sw_codes:
            try:
                df = ak.index_hist_sw(symbol=code)
                df = df.rename(columns={"日期": "trade_date", "收盘": "close"})
                # 有的API用 "收盘价"
                if "close" not in df.columns:
                    df = df.rename(columns={"收盘价": "close"})
                if "close" not in df.columns:
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                s = df.set_index("trade_date")["close"]
                s = s.loc[START_DATE:END_DATE]
                if len(s) > 100:
                    closes[code] = s
            except Exception:
                continue

        if closes:
            merged = pd.DataFrame(closes).sort_index()
            _save(name, merged)
            print(f"[OK] {name}: {len(merged)} rows x {len(merged.columns)} industries")
            return merged
    except Exception as e:
        print(f"[WARN] {name}: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════
#  B3: 情绪面代理（从价格计算）
# ═══════════════════════════════════════════
def fetch_sentiment_proxy(force: bool = False) -> pd.DataFrame:
    name = "sentiment"
    if not force and (c := _load(name)) is not None:
        print(f"[SKIP] {name} (cached, {len(c)} rows)")
        return c

    idx = fetch_index_prices()
    df = pd.DataFrame(index=idx.index)

    close = idx.get("HS300_close")
    volume = idx.get("HS300_volume")
    if close is None or volume is None:
        print(f"[WARN] {name}: missing columns in index_prices, have {list(idx.columns)}")
        return pd.DataFrame()

    ret = close.pct_change()
    df["realized_vol_20d"] = ret.rolling(20).std() * np.sqrt(252)
    df["turnover_change_5d"] = volume.pct_change(5)
    df["sent_momentum"] = close.pct_change(5) - close.pct_change(20)
    df["pv_corr_20d"] = ret.rolling(20).mean() * volume.pct_change(5).rolling(20).mean()
    df["max_dd_60d"] = (close / close.rolling(60).max()) - 1

    _save(name, df)
    print(f"[OK] {name}: {len(df)} rows, cols={list(df.columns)}")
    return df


# ═══════════════════════════════════════════
#  ETF 日行情 (rqdatac)
# ═══════════════════════════════════════════
ETF_UNIVERSE = [
    ("510300.XSHG", "沪深300ETF"),
    ("510500.XSHG", "中证500ETF"),
    ("159915.XSHE", "创业板ETF"),
    ("510050.XSHG", "上证50ETF"),
    ("512100.XSHG", "中证1000ETF"),
    ("588000.XSHG", "科创50ETF"),
    ("510880.XSHG", "红利ETF"),
    ("512880.XSHG", "证券ETF"),
    ("512690.XSHG", "酒ETF"),
    ("512200.XSHG", "房地产ETF"),
    ("518880.XSHG", "黄金ETF"),
    ("513100.XSHG", "纳指ETF"),
    ("513050.XSHG", "中概互联ETF"),
]


def fetch_etf_prices(force: bool = False) -> pd.DataFrame:
    name = "etf_prices"
    p = _path(name)
    if not force and p.exists():
        df = pd.read_parquet(p)
        print(f"[SKIP] {name} (cached, {len(df)} rows, {len(df.columns)} cols)")
        return df

    import rqdatac
    rqdatac.init()

    dfs = []
    for code, label in ETF_UNIVERSE:
        try:
            df = rqdatac.get_price(code, start_date=START_DATE, end_date=END_DATE,
                                   fields=["close", "volume"], adjust_type="none")
            if isinstance(df.index, pd.MultiIndex):
                df.index = df.index.get_level_values(-1)
            df = df.rename(columns={"close": f"{label}_close", "volume": f"{label}_volume"})
            df.index = pd.to_datetime(df.index)
            df.index.name = None
            dfs.append(df)
            print(f"  [OK] {label} ({code}): {len(df)} rows")
        except Exception as e:
            print(f"  [WARN] {label} ({code}): {e}")

    if not dfs:
        print(f"[FAIL] {name}: no ETF data fetched")
        return pd.DataFrame()

    merged = pd.concat(dfs, axis=1).sort_index()
    _save(name, merged)
    print(f"[OK] {name}: {len(merged)} rows, {len(merged.columns)} cols")
    return merged


# ═══════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════
def fetch_all(force: bool = False) -> dict:
    results = {}
    steps = [
        ("index_prices", fetch_index_prices),
        ("yield_curve", fetch_yield_curve),
        ("northbound_flow", fetch_northbound_flow),
        ("margin", fetch_margin),
        ("macro", fetch_macro),
        ("sw_industry", fetch_sw_industry),
        ("sentiment", fetch_sentiment_proxy),
        ("etf_prices", fetch_etf_prices),
    ]
    results = {}
    for label, func in steps:
        print(f"\n{'='*50}\n[{label}]")
        try:
            results[label] = func(force)
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            results[label] = pd.DataFrame()

    print(f"\n{'='*50}")
    print("SUMMARY:")
    for k, v in results.items():
        s = f"{len(v)} rows" if not v.empty else "EMPTY"
        print(f"  {k:25s} -> {s}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    fetch_all(force=args.force)
