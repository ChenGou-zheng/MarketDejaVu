# Project: 中国股市历史行情与证券交易策略匹配

> **版本**: v3.1 | **截止**: 2026-05-14 | **方法**: KNN 相似度 + N=20 日块持有 + Top-3 分散

---

## 核心原则 (摘自 todo.md)

1. **绝对无未来信息泄露** — 特征/标签/筛选 100% 基于 t 时点前数据 (τ+N<t 约束)
2. **前瞻统计推断** — 加权欧氏KNN + 单边 t 检验 + BH-FDR 校正 (q=0.1)
3. **诚实评估** — Deflated Sharpe Ratio 检验系统整体显著性

---

## 最终结果

| 指标 | K=1 (单策略) | **K=3 (分散)** | K=5 |
|------|-------------|-------------|-----|
| **年化超额** | +0.31% | **+2.41%** | +1.92% |
| **信息比率** | 0.012 | **0.143** | 0.135 |
| **块胜率** | 54.1% | 52.6% | **56.4%** |
| **最大回撤** | 60.36% | **38.85%** | 40.19% |
| **DSR p-value** | < 0.001 | **< 0.001** | < 0.001 |

**选择: K=3 等权分散** — IR +12x, 回撤-35%, 年化超额+2.41%

## 样本内外 (K=3)

| | 样本内 (pre-2024) | 样本外 (2024+) |
|--|-----------------|--------------|
| 年化超额 | +2.36% | **+2.56%** |
| 块胜率 | 51.5% | **54.8%** |
| 日胜率 | 48.2% | 47.1% |
| 最大回撤 | — | 38.85% |

## 策略池 (58 个)

| 类型 | 数量 | 来源 |
|------|------|------|
| total_summary 映射 | **41** | 策略名 → rqdata ETF 名称模糊匹配 |
| 宽基 ETF (长持) | **13** | HS300/中证500/创业板/纳指/黄金等 |
| MTM 重建 (原始) | **4** | 桥水全天候/均衡持仓/军工/etf动量轮动 |

## 命令

```bash
# 完整管线
uv run python src/data/fetcher.py
uv run python src/features/build_all_strategies.py
uv run python src/features/build_snapshot.py
uv run python src/models/similarity_engine.py
uv run python src/backtest/dynamic_backtest.py --top-k 3

# 对比不同分散度
uv run python src/backtest/dynamic_backtest.py --top-k 1
uv run python src/backtest/dynamic_backtest.py --top-k 3   # 推荐
uv run python src/backtest/dynamic_backtest.py --top-k 5
```

## 数据范围分析

| 指标 | 值 |
|------|-----|
| 最早策略起始 | 2016-01-08 |
| 最晚策略起始 | **2023-09-14** (中证2000ETF) |
| 90% 策略起始 ≤ | **2023-05-11** |
| 10% 策略起始 > 2023 | 中证2000ETF (2023-09-14) |
| 总交易天数 | 2670 (对齐后) |
| 首决策日 | 2016-03-04 |

**所有新 ETF/指数策略最早数据始于 2016-01-08，与特征数据完全对齐。** 唯一较晚的是中证2000ETF (2023-09-14)，但仅占 1/58。

## 管线命令

```bash
# 1. 拉取 ETF 数据 (缓存)
uv run python src/data/fetcher.py

# 2. 构建 ETF 策略 NAV
uv run python src/features/build_etf_strategies.py

# 3. 映射 total_summary 策略→指数/ETF
uv run python src/features/map_strategies_to_funds.py

# 4. 无泄露快照库
uv run python src/features/build_snapshot.py

# 5. KNN 相似度匹配
uv run python src/models/similarity_engine.py

# 6. 动态回测
uv run python src/backtest/dynamic_backtest.py
```

## 受保护文件

`data/raw/*`, `pyproject.toml`, `uv.lock`, `docs/project-def.md`, `docs/todo.md`
