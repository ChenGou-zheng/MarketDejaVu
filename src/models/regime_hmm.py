"""
市场状态推断：特征工程 + HMM 隐状态检测（Train/Test 切分）。

Train: 2016-04-01 ~ 2024-12-31 (~2283 天)
Test:  2025-01-01 ~ 2026-04-30 (~347 天)

HMM 仅拟合训练集。测试集只做 transform + predict。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_PATH = PROJECT_ROOT / "data" / "processed" / "aligned_daily.parquet"
OUT_TABLES = PROJECT_ROOT / "output" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    # B1
    feats["ret_HS300"] = df.get("HS300_ret", 0)
    feats["ret_ZZ500"] = df.get("ZZ500_ret", 0)
    feats["ret_CYB"] = df.get("CYB_ret", 0)
    feats["vol_20d"] = feats["ret_HS300"].rolling(20).std() * np.sqrt(252)
    feats["vol_60d"] = feats["ret_HS300"].rolling(60).std() * np.sqrt(252)
    feats["skew_60d"] = feats["ret_HS300"].rolling(60).skew()
    feats["kurt_60d"] = feats["ret_HS300"].rolling(60).kurt()
    close = df.get("HS300_close", df["HS300_close"])
    feats["max_dd_60d"] = (close / close.rolling(60).max()) - 1

    # B2
    feats["northbound_5d"] = df.get("net_total", 0).rolling(5).sum()
    feats["margin_chg_20d"] = df.get("margin_balance", 0).pct_change(20)
    feats["m2_yoy"] = df.get("m2_yoy", np.nan)

    # B3
    feats["realized_vol_20d"] = df.get("realized_vol_20d", 0)
    feats["turnover_chg"] = df.get("turnover_change_5d", 0)
    feats["sent_momentum"] = df.get("sent_momentum", 0)

    # B4
    feats["size_style"] = df.get("ZZ500_ret", 0) - df.get("HS300_ret", 0)
    feats["growth_style"] = df.get("CYB_ret", 0) - df.get("HS300_ret", 0)
    feats["momentum_20d"] = feats["ret_HS300"].rolling(20).mean()
    feats["momentum_60d"] = feats["ret_HS300"].rolling(60).mean()
    feats["industry_dispersion"] = df.get("industry_dispersion", 0)
    feats["sector_rotation"] = df.get("sector_rotation", 0)
    vol_300 = df.get("HS300_volume", np.nan)
    vol_500 = df.get("ZZ500_volume", np.nan)
    feats["vol_ratio_500_300"] = vol_500 / vol_300

    # B5
    feats["yield_slope"] = df.get("yield_slope", np.nan)
    feats["yield_10y"] = df.get("10Y", np.nan)
    feats["pmi_mfg"] = df.get("pmi_mfg", np.nan)
    feats["cpi_yoy"] = df.get("cpi_yoy", np.nan)
    feats["ppi_yoy"] = df.get("ppi_yoy", np.nan)

    return feats


def _fit_hmm(X: np.ndarray, n: int, n_iter: int = 1000) -> hmm.GaussianHMM | None:
    try:
        m = hmm.GaussianHMM(
            n_components=n, covariance_type="diag",
            n_iter=n_iter, random_state=42, tol=1e-4,
            init_params="stmc", params="stmc",
        )
        m.fit(X)
        return m
    except Exception as e:
        print(f"  [WARN] HMM K={n} failed: {e}")
        return None


def _state_stats(F: pd.DataFrame, states: np.ndarray, label_prefix: str) -> pd.DataFrame:
    rows = []
    for s in np.unique(states):
        mask = states == s
        avg = F.iloc[mask].mean()
        rows.append(avg)
    return pd.DataFrame(rows, index=[f"{label_prefix}_State_{i}" for i in np.unique(states)])


def _print_regime_summary(F: pd.DataFrame, states: np.ndarray, tag: str) -> None:
    print(f"\n  [{tag}] State characteristics:")
    for s in sorted(np.unique(states)):
        mask = states == s
        ret_s = F.iloc[mask]["ret_HS300"]
        n = mask.sum()
        cum = (1 + ret_s).prod() - 1
        vol = ret_s.std() * np.sqrt(252)
        sr = ret_s.mean() / ret_s.std() * np.sqrt(252) if ret_s.std() > 0 else 0
        print(f"    State {s}: {n:5d} days | cum_ret={cum:8.2%} | ann_vol={vol:.2%} | sharpe={sr:+.2f}")


def run_hmm(force_k: int | None = None) -> dict:
    # ── 加载 ──
    print("Loading data...")
    df = pd.read_parquet(ALIGNED_PATH)
    df.index = pd.to_datetime(df.index)

    print("Building features...")
    F = build_features(df)
    F = F.iloc[60:].copy()
    F = F.ffill().bfill()

    print(f"Feature matrix: {F.shape[0]} days x {F.shape[1]} features")
    print(f"Full range: {F.index[0].date()} ~ {F.index[-1].date()}")

    # ── Train / Test 切分 ──
    train_mask = F.index <= TRAIN_END
    test_mask = F.index > TRAIN_END

    F_train = F[train_mask]
    F_test = F[test_mask]

    print(f"\nTrain: {len(F_train)} days ({F_train.index[0].date()} ~ {F_train.index[-1].date()})")
    print(f"Test:  {len(F_test)} days ({F_test.index[0].date()} ~ {F_test.index[-1].date()})")

    # ── 标准化（仅在训练集上 fit）──
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(F_train.values)
    X_test = scaler.transform(F_test.values)

    # ── 选 K（仅在训练集上算 BIC）──
    if force_k is not None:
        best_k = force_k
        print(f"\nForced K = {best_k}")
    else:
        print("\nSelecting K via BIC (train set only)...")
        best_k, best_bic = None, np.inf
        for k in range(3, 7):
            m = _fit_hmm(X_train_raw, k)
            if m is None:
                continue
            bic = m.bic(X_train_raw)
            print(f"  K={k:2d}  BIC={bic:.1f}")
            if bic < best_bic:
                best_bic, best_k = bic, k
        print(f"Selected K = {best_k}  (BIC={best_bic:.1f})")

    # ── 拟合最终模型 ──
    print(f"\nFitting final HMM (K={best_k}) on train set...")
    model = _fit_hmm(X_train_raw, best_k, n_iter=2000)
    if model is None:
        raise RuntimeError("HMM fitting failed")

    # ── Predict ──
    train_states = model.predict(X_train_raw)
    test_states = model.predict(X_test)

    # ── 输出状态序列 ──
    for tag, dates, states in [
        ("train", F_train.index, train_states),
        ("test",  F_test.index,  test_states),
    ]:
        pd.DataFrame({"trade_date": dates, "state": states}).to_csv(
            PROJECT_ROOT / "data" / "processed" / f"hmm_states_{tag}.csv",
            index=False, encoding="utf-8-sig",
        )
    print("State files saved: hmm_states_{train,test}.csv")

    # ── 转移矩阵（仅训练集估计）──
    trans = pd.DataFrame(
        model.transmat_,
        index=[f"from_{i}" for i in range(best_k)],
        columns=[f"to_{i}" for i in range(best_k)],
    )
    trans.to_csv(OUT_TABLES / "transition_matrix.csv", encoding="utf-8-sig")
    print(f"\nTransition matrix (train):\n{trans.round(4)}")

    # ── 各集状态特征 ──
    train_stats = _state_stats(F_train, train_states, "Train")
    test_stats = _state_stats(F_test, test_states, "Test")
    train_stats.to_csv(OUT_TABLES / "regime_features_train.csv", encoding="utf-8-sig")
    test_stats.to_csv(OUT_TABLES / "regime_features_test.csv", encoding="utf-8-sig")

    # ── 打印总结 ──
    _print_regime_summary(F_train, train_states, "TRAIN")
    _print_regime_summary(F_test, test_states, "TEST")

    # ── 状态分布对比 ──
    print("\n--- State distribution ---")
    dist = pd.DataFrame({
        "Train": pd.Series(train_states).value_counts(normalize=True).sort_index(),
        "Test":  pd.Series(test_states).value_counts(normalize=True).sort_index(),
    }).fillna(0)
    print(dist.to_string(float_format=lambda x: f"{x:.1%}"))
    dist.to_csv(OUT_TABLES / "state_distribution.csv", encoding="utf-8-sig")

    # ── 新状态检查 ──
    train_set = set(train_states)
    test_set = set(test_states)
    print(f"\nTrain states: {sorted(train_set)}")
    print(f"Test  states: {sorted(test_set)}   (new: {sorted(test_set - train_set)})")

    return {
        "model": model, "k": best_k,
        "states": {"train": train_states, "test": test_states},
        "features": {"train": F_train, "test": F_test},
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()
    run_hmm(force_k=args.k)
