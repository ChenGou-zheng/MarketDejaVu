"""
Map total_summary strategies to real index/ETF codes.

For each strategy name:
  1. Manual override (predefined best-match code)
  2. Fallback: fuzzy match against rqdata ETF/INDX names
  3. If all fails: default to HS300

A-shares preferred over C-shares for funds.
Data cached to data/external/strategy_prices.parquet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
NAV_DIR = PROJECT_ROOT / "data" / "processed" / "strategy_nav"
REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_registry.csv"
CACHE_PATH = EXTERNAL_DIR / "strategy_prices.parquet"

START_DATE = "2016-01-08"
END_DATE = "2026-04-30"



MANUAL_OVERRIDES = {
    "煤炭周期优选动态轮动策略": ("515220.XSHG", "煤炭ETF", "ETF"),
    "沪深300增强策略": ("510300.XSHG", "沪深300ETF", "ETF"),
    "半导体优选策略": ("159516.XSHE", "半导体ETF", "ETF"),
    "成长红利量化选股": ("159708.XSHE", "红利ETF", "ETF"),
    "双创50增强": ("159782.XSHE", "双创50ETF", "ETF"),
    "计算机ETF优选策略": ("159998.XSHE", "计算机ETF", "ETF"),
    "科创参数加强板": ("159603.XSHE", "科创ETF", "ETF"),
    "机器人ETF优选策略": ("159551.XSHE", "机器人ETF", "ETF"),
    "化工ETF优选策略": ("159870.XSHE", "化工ETF", "ETF"),
    "中证2000增强": ("159531.XSHE", "中证2000ETF", "ETF"),
    "形态识别": ("000300.XSHG", "沪深300", "INDX"),
    "中证800增强": ("159707.XSHE", "中证800ETF", "ETF"),
    "军工etf增强": ("512560.XSHG", "军工ETF", "ETF"),
    "中证1000增强": ("159629.XSHE", "中证1000ETF", "ETF"),
    "旅游etf增强": ("159766.XSHE", "旅游ETF", "ETF"),
    "国企etf增强": ("159719.XSHE", "国企ETF", "ETF"),
    "游戏etf增强": ("159869.XSHE", "游戏ETF", "ETF"),
    "酒etf增强": ("512690.XSHG", "酒ETF", "ETF"),
    "动量趋势策略": ("000300.XSHG", "沪深300", "INDX"),
    "锤子": ("000300.XSHG", "沪深300", "INDX"),
    "综合全": ("000300.XSHG", "沪深300", "INDX"),
    "医疗etf增强": ("399989.XSHE", "中证医疗", "INDX"),
    "通信etf增强": ("159507.XSHE", "通信ETF", "ETF"),
    "房地产etf增强": ("159768.XSHE", "房地产ETF", "ETF"),
    "食品etf增强": ("159736.XSHE", "食品ETF", "ETF"),
    "创业板增强": ("159541.XSHE", "创业板ETF", "ETF"),
    "etf动量改": ("000300.XSHG", "沪深300", "INDX"),
    "行业etf增强": ("000300.XSHG", "沪深300", "INDX"),
    "策略ETF": ("000300.XSHG", "沪深300", "INDX"),
    "策略etf2": ("159505.XSHE", "国证2000ETF", "ETF"),
    "养殖etf增强": ("159865.XSHE", "养殖ETF", "ETF"),
    "综合拆分1": ("000300.XSHG", "沪深300", "INDX"),
    "综合拆分2": ("000300.XSHG", "沪深300", "INDX"),
    "全球etf增强": ("513100.XSHG", "纳指ETF", "ETF"),
    "百亿etf 1只": ("000300.XSHG", "沪深300", "INDX"),
    "百亿etf 2只": ("000300.XSHG", "沪深300", "INDX"),
    "桥水全天候": ("000300.XSHG", "沪深300", "INDX"),
    "申万动量": ("000300.XSHG", "沪深300", "INDX"),
    "均衡持仓": ("000300.XSHG", "沪深300", "INDX"),
    "杠铃": ("000300.XSHG", "沪深300", "INDX"),
    "etf动量轮动": ("000300.XSHG", "沪深300", "INDX"),
}

# Known ETFs from build_etf_strategies.py
KNOWN_ETFS = {
    "沪深300ETF": "510300.XSHG",
    "中证500ETF": "510500.XSHG",
    "创业板ETF": "159915.XSHE",
    "上证50ETF": "510050.XSHG",
    "中证1000ETF": "512100.XSHG",
    "科创50ETF": "588000.XSHG",
    "红利ETF": "510880.XSHG",
    "证券ETF": "512880.XSHG",
    "酒ETF": "512690.XSHG",
    "房地产ETF": "512200.XSHG",
    "黄金ETF": "518880.XSHG",
    "纳指ETF": "513100.XSHG",
    "中概互联ETF": "513050.XSHG",
}


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[etfETF\s\-_（）()]", "", s)
    s = s.replace("增强", "").replace("优选", "").replace("策略", "")
    s = s.replace("动态轮动", "").replace("量化选股", "").replace("参数加强板", "")
    return s.strip()


def fuzzy_match_strategy(strat_name: str, etfs_df, indices_df):
    """Fuzzy match a strategy name to the best ETF or index."""
    sn = normalize(strat_name)
    if not sn:
        return None

    candidates = []

    for df, dtype in [(etfs_df, "ETF"), (indices_df, "INDX")]:
        for _, row in df.iterrows():
            inst_n = normalize(row["symbol"])
            if not inst_n:
                continue

            score = 0.0
            if sn == inst_n:
                score = 1.0
            elif sn in inst_n:
                score = 0.9
            elif inst_n in sn:
                score = 0.85
            else:
                common = len(set(sn) & set(inst_n))
                total = max(len(set(sn)), len(set(inst_n)))
                if total > 0 and common / total > 0.6:
                    score = 0.7 * common / total

            if score > 0.4:
                is_a = bool(re.search(r'A$|A\b', row["symbol"]))
                is_c = bool(re.search(r'C$|C\b', row["symbol"]))
                candidates.append({
                    "code": row["order_book_id"],
                    "name": row["symbol"],
                    "type": dtype,
                    "score": score,
                    "is_a": is_a,
                    "is_c": is_c,
                })

    if not candidates:
        return None

    # Sort: score desc, A-share preferred, ETF preferred, non-C preferred
    candidates.sort(key=lambda x: (-x["score"], not x["is_a"], x["type"] != "ETF", x["is_c"]))
    return candidates[0]


def fetch_price(code: str) -> pd.Series | None:
    import rqdatac
    rqdatac.init()
    try:
        df = rqdatac.get_price(code, start_date=START_DATE, end_date=END_DATE, fields=["close"])
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(-1)
        return df["close"].rename(code)
    except Exception:
        return None


def main():
    # Load registry, identify existing entries to remove
    existing = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()
    existing_keys = set(existing["strategy_key"].tolist()) if not existing.empty else set()

    # Remove old 策略_ entries and rebuild
    old_etf_strat = existing[existing["strategy_key"].str.startswith("策略_", na=False)].copy()
    if not old_etf_strat.empty:
        print(f"Removing {len(old_etf_strat)} old strategy entries from registry")
        existing = existing[~existing["strategy_key"].str.startswith("策略_", na=False)]
        existing_keys = set(existing["strategy_key"].tolist())

    # Remove old NAV files
    for f in NAV_DIR.glob("策略_*_nav.csv"):
        f.unlink()
        print(f"  removed {f.name}")

    # Get instrument catalog from rqdata
    import rqdatac
    rqdatac.init()
    all_inst = rqdatac.all_instruments(date="2024-01-01")
    etfs_cat = all_inst[all_inst["type"] == "ETF"].copy()
    indices_cat = all_inst[all_inst["type"] == "INDX"].copy()
    print(f"Catalog: {len(etfs_cat)} ETFs, {len(indices_cat)} indices")

    # For each strategy, find best match
    matches = {}
    for strat_name, (code, label, ctype) in MANUAL_OVERRIDES.items():
        # Try: can we find a better ETF match?
        fuzzy = fuzzy_match_strategy(strat_name, etfs_cat, indices_cat)
        if fuzzy and fuzzy["score"] >= 0.85 and fuzzy["type"] == "ETF":
            # ETF match is better than manual override
            matches[strat_name] = fuzzy
            print(f"  [ETF] {strat_name:20s} -> {fuzzy['name']:20s} ({fuzzy['code']:12s}) score={fuzzy['score']:.2f}")
        else:
            # Use manual override
            matches[strat_name] = {"code": code, "name": label, "type": ctype, "score": 1.0, "is_a": False, "is_c": False}
            print(f"  [MAN] {strat_name:20s} -> {label:20s} ({code:12s})")

    # Fetch and save
    all_prices = {}
    nav_entries = []

    for strat_name, match in matches.items():
        key = f"策略_{strat_name}"
        if key in existing_keys:
            print(f"  [SKIP] {strat_name} (in registry)")
            continue

        print(f"  Fetching {strat_name} -> {match['name']} ({match['code']})...")
        close = fetch_price(match["code"])
        if close is None or len(close) < 60:
            print(f"    FAIL: no data")
            continue

            if close is None or len(close) < 60:
                print(f"    FAIL, defaulting to HS300")
                close = fetch_price("000300.XSHG")
                match["code"] = "000300.XSHG"
                match["name"] = "沪深300"

        all_prices[f"{match['code']}_{strat_name}"] = close
        nav = close / close.iloc[0]
        daily_ret = nav.pct_change()
        out_df = pd.DataFrame({"trade_date": close.index, "nav": nav.values, "daily_return": daily_ret.values})

        safe_name = f"策略_{strat_name}"
        out_path = NAV_DIR / f"{safe_name}_nav.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        total_return = float(nav.iloc[-1] - 1)
        ann_vol = float(daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0
        max_dd = float((1 - nav / nav.cummax()).min())

        nav_entries.append({
            "strategy_key": key,
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "trading_days": len(close),
            "total_return": total_return,
            "annual_vol": ann_vol,
            "max_drawdown": max_dd,
            "source_files": 0,
        })
        existing_keys.add(key)
        print(f"    {len(close)} days, {close.index[0].date()} ~ {close.index[-1].date()}, tot={total_return:+.2%} vol={ann_vol:.0%}")

    # Also ensure known ETFs are in NAV dir
    print("\n=== Known ETFs ===")
    for lbl, code in KNOWN_ETFS.items():
        key = f"ETF_长持_{lbl}"
        if key in existing_keys:
            continue
        nav_path = NAV_DIR / f"ETF_长持_{lbl}_nav.csv"
        if nav_path.exists():
            existing_keys.add(key)
            continue

        close = fetch_price(code)
        if close is None or len(close) < 60:
            continue

        nav = close / close.iloc[0]
        daily_ret = nav.pct_change()
        out_df = pd.DataFrame({"trade_date": close.index, "nav": nav.values, "daily_return": daily_ret.values})
        out_df.to_csv(nav_path, index=False, encoding="utf-8-sig")

        nav_entries.append({
            "strategy_key": key,
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "trading_days": len(close),
            "total_return": float(nav.iloc[-1] - 1),
            "annual_vol": float(daily_ret.std() * (252 ** 0.5)),
            "max_drawdown": float((1 - nav / nav.cummax()).min()),
            "source_files": 0,
        })
        existing_keys.add(key)
        print(f"  [NEW] {lbl}: {len(close)} days")

    # Save cache
    if all_prices:
        cache_df = pd.DataFrame(all_prices)
        cache_df.index = pd.to_datetime(cache_df.index)
        cache_df.to_parquet(CACHE_PATH)
        print(f"\n[OK] Cached {len(all_prices)} series -> {CACHE_PATH}")

    # Update registry
    if nav_entries:
        # Keep only MTM-clean strategies from original
        mtm_keep = ["交易记录桥水全天候", "交易记录均衡持仓", "交易记录军工", "交易记录etf动量轮动"]
        mtm = existing[existing["strategy_key"].isin(mtm_keep)].copy() if not existing.empty else pd.DataFrame()
        # Also keep known ETF entries
        etf_keep = existing[existing["strategy_key"].str.startswith("ETF_长持_", na=False)].copy() if not existing.empty else pd.DataFrame()
        # Clear old 策略_ and rebuild
        rest = existing[~existing["strategy_key"].str.startswith(("策略_", "ETF_长持_"), na=False)].copy() if not existing.empty else pd.DataFrame()

        combined = pd.concat([rest, mtm, etf_keep, pd.DataFrame(nav_entries)], ignore_index=True)
        combined = combined.sort_values("trading_days", ascending=False)
        combined.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] Registry: {len(combined)} entries (+{len(nav_entries)} new)")

    # Stats
    registry = pd.read_csv(REGISTRY_PATH)
    registry["start_date"] = pd.to_datetime(registry["start_date"])
    clean = registry[registry["source_files"] == 0]
    print(f"\n=== Stats ===")
    print(f"  Total entries: {len(registry)}")
    print(f"  Clean (ETF/策略): {len(clean)}")
    print(f"  Earliest start: {clean['start_date'].min().date()}")
    print(f"  Latest start:   {clean['start_date'].max().date()}")
    print(f"  90th percentile start: {clean['start_date'].quantile(0.9).date()}")


if __name__ == "__main__":
    main()
