# Project: 中国股市历史行情与证券交易策略匹配

> **版本**: v2.5 | **截止**: 2026-05-14 | **方法**: KNN 相似度 + N=20 日块持有

---

## 核心原则 (摘自 todo.md)

1. **绝对无未来信息泄露** — 特征/标签/筛选 100% 基于 t 时点前数据 (τ+N<t 约束)
2. **前瞻统计推断** — 加权欧氏KNN + 单边 t 检验 + BH-FDR 校正 (q=0.1)
3. **诚实评估** — Deflated Sharpe Ratio 检验系统整体显著性

---

## 最终结果 (rqdata 真实 ETF 净值)

| 指标 | 值 |
|------|-----|
| **年化超额** | -0.72% |
| **日胜率** | 47.0% |
| **块胜率 (20日)** | **54.1%** |
| **信息比率** | -0.030 |
| **最大回撤** | 43.11% |
| **DSR p-value** | **0.0064** (显著) |
| **策略数** | 17 (13 真实 ETF + 4 MTM) |

## 样本内外对比

| | 样本内 (pre-2024) | 样本外 (2024+) |
|--|-----------------|--------------|
| 年化超额 | +2.10% | -9.60% |
| 块胜率 | **55.3%** | 48.4% |
| 日胜率 | 47.4% | 45.5% |

## 策略池

### 13 个真实 ETF (rqdata, 长持)

| ETF | 年化 Vol | 选中块数 | 块胜率 |
|-----|---------|---------|-------|
| 纳指ETF | 34% | 27 | **59%** |
| 中概互联ETF | 29% | 16 | **62%** |
| 黄金ETF | 15% | 14 | 43% |
| 动量轮动(原始) | 73% | 14 | **71%** |
| 军工(原始) | 62% | 10 | **50%** |
| 酒ETF | 37% | 9 | 44% |
| 证券ETF | 29% | 9 | 33% |
| 创业板ETF | 29% | 7 | **57%** |
| 均衡持仓(原始) | 84% | 7 | **57%** |

### 数据来源

| 类型 | 数量 | 来源 |
|------|------|------|
| 真实 ETF 日行情 | 13 | `rqdatac.get_price()` → `data/external/etf_prices.parquet` |
| MTM 重建策略 | 4 | `rebuild_daily_returns()` 从原始交易 CSV |

## 管线命令

```bash
# 1. 拉取 ETF 数据 (首次, 后续从 parquet 缓存加载)
uv run python src/data/fetcher.py

# 2. 构建策略 NAV
uv run python src/features/build_etf_strategies.py

# 3. 无泄露快照库
uv run python src/features/build_snapshot.py

# 4. KNN 相似度匹配
uv run python src/models/similarity_engine.py

# 5. 动态回测
uv run python src/backtest/dynamic_backtest.py
```

## 输出文件

| 路径 | 内容 |
|------|------|
| `data/external/etf_prices.parquet` | 13 个 ETF 日频价 (rqdata 缓存) |
| `data/processed/snapshot/features.parquet` | 2670×16 特征矩阵 |
| `data/processed/snapshot/labels.parquet` | 2670×17 策略超额标签 |
| `output/tables/similarity_decisions.csv` | 2630 日决策 |
| `output/tables/backtest_knn_qf_off.csv` | 日频回测净值 |

## 受保护文件

`data/raw/*`, `pyproject.toml`, `uv.lock`, `docs/project-def.md`, `docs/todo.md`
