"""
数据整合：将所有拉取的数据对齐到统一日频 DataFrame。

输入:  data/external/*.csv
输出:  data/processed/aligned_daily.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "aligned_daily.parquet"

START_DATE = "2016-01-08"
END_DATE = "2026-04-30"


def load_source(name: str) -> pd.DataFrame:
    p = EXTERNAL_DIR / f"{name}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df.index.name = None  # normalize
        print(f"  Loaded {name}: {len(df)} rows, {list(df.columns)[:10]}...")
        return df
    cp = EXTERNAL_DIR / f"{name}.csv"
    if cp.exists():
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        df.index.name = None
        print(f"  Loaded {name} (csv): {len(df)} rows")
        return df
    print(f"  [WARN] {name} not found")
    return pd.DataFrame()


def main() -> pd.DataFrame:
    # 1. 宽基指数日行情
    idx = load_source("index_prices")
    if idx.empty:
        raise RuntimeError("index_prices.csv is required")

    # 2. 收益率曲线
    yc = load_source("yield_curve")

    # 3. 北向资金
    nb = load_source("northbound_flow")

    # 4. 融资融券
    mg = load_source("margin")

    # 5. 宏观（月度→日频前向填充）
    macro = load_source("macro")

    # 6. 申万行业
    sw = load_source("sw_industry")

    # 7. 情绪面代理
    sent = load_source("sentiment")

    # ── 对齐到统一日频 ──
    date_range = pd.date_range(START_DATE, END_DATE, freq="B")  # 交易日
    aligned = pd.DataFrame(index=date_range)
    aligned.index.name = "trade_date"

    # 日频数据直接 join
    for df, tag in [(idx, ""), (yc, ""), (nb, ""), (mg, ""), (sent, "")]:
        if df.empty:
            continue
        aligned = aligned.join(df, how="left")

    # 行业收益离散度（从 SW 指数计算）— 优化：采样而非全量
    if not sw.empty:
        sw_aligned = sw.reindex(date_range)
        sw_ret = sw_aligned.pct_change(fill_method=None)
        aligned["industry_dispersion"] = sw_ret.std(axis=1)
        # 行业轮动速度：用每5日的排名变化衡量，避免全量计算
        sector_rotation = []
        for i in range(0, len(sw_aligned), 5):  # every 5 days
            window = sw_aligned.iloc[i:i+5]
            if len(window) < 2:
                continue
            ranked = window.rank(axis=1)
            rot = ranked.diff().abs().sum(axis=1).mean()
            for j in range(len(window)):
                sector_rotation.append(rot)
        if sector_rotation:
            aligned["sector_rotation"] = pd.Series(sector_rotation[:len(aligned)], index=aligned.index)

    # 月度宏观 → 前向填充到日频
    if not macro.empty:
        macro_daily = macro.reindex(date_range, method="ffill")
        aligned = aligned.join(macro_daily, how="left")

    # ── 前向填充日频缺失（非交易日或缺失数据）──
    aligned = aligned.ffill()

    # ── 计算额外列 ──
    if "HS300_close" in aligned.columns:
        aligned["HS300_ret"] = aligned["HS300_close"].pct_change(fill_method=None)
    if "CYB_close" in aligned.columns:
        aligned["CYB_ret"] = aligned["CYB_close"].pct_change(fill_method=None)
    if "ZZ500_close" in aligned.columns:
        aligned["ZZ500_ret"] = aligned["ZZ500_close"].pct_change(fill_method=None)

    # 确保无剩余 NaN
    aligned = aligned.dropna(how="all")

    aligned.to_parquet(OUT_PATH)
    print(f"\n[DONE] aligned_daily.parquet: {len(aligned)} rows x {len(aligned.columns)} cols")
    print(f"  Columns: {list(aligned.columns)}")
    return aligned


if __name__ == "__main__":
    main()
