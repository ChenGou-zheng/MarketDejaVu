# MSA — 相似行情匹配与动态策略选择系统

KNN 相似度 + 统计检验 + Top-K 分散的动态策略选择系统。
16 维市场特征（动量/波动/回撤/资金面/宏观）找出最相似的 30 个历史交易日，
单边 t 检验 + BH-FDR 筛选显著策略，Top-3 等权持有 20 个交易日。

---

## 问题定义

自 1992 年至今，中国股市经历了多种行情。对历史行情进行学习与分类，
当当前行情特征与历史某段行情高度相似时，采用该段行情中表现良好的策略，
以期获取超额收益。

**核心任务**：
1. 用市场特征（K线、波动率、最大回撤、收益率、宏观政策、资金面等）描述行情
2. 判断当前行情与历史行情的相似度
3. 识别历史中哪些策略表现良好
4. 根据当前行情动态选择策略

> 详见 `docs/project-def.md`

---

## 原理

```
数据层                           特征与标签                          相似度引擎
┌──────────┐    ┌────────────────────────┐    ┌───────────────────────────────┐
│ rqdata    │    │ F(t): 16维市场特征      │    │ 加权欧氏 KNN (K=30)           │
│ ETF/指数  │───→│ (动量/波动/回撤/        │───→│ τ+N<t 无未来约束               │
│ 个股收盘价 │    │  宏观/资金面)           │    │ Z-score 标准化                │
│ 策略定义  │    │                        │    │ 时间衰减 (半衰期3年)           │
│ CSV流水   │    │ Labels: N日超额收益     │    │ 单边t检验 + BH-FDR (q=0.1)    │
└──────────┘    └────────────────────────┘    └───────────────┬───────────────┘
                                                              ↓
                                                    回测执行: Top-K 等权持有
                                                    每20日再平衡, DSR 验证
```

详细方法（含公式）见 `report.md`。

---

## 项目结构

```
docs/
├── project-def.md            # 项目定义文档
├── strategy_definitions.csv  # 策略池唯一定义 (59个策略)
├── ricequant-doc-index.md    # rqdata 接口文档索引
└── todo.md                   # 开发日志

src/
├── data/
│   ├── fetcher.py            # 数据拉取 (rqdata ETF/指数/个股/宏观)
│   └── preprocess.py         # 日频数据对齐整合
├── features/
│   ├── build_all_strategies.py   # 构建 ETF/映射策略 NAV
│   ├── build_snapshot.py         # 无泄露特征+标签库 + 质量过滤
│   └── rebuild_mtm_nav.py        # MTM 净值重建 (盯市)
├── models/
│   └── similarity_engine.py      # KNN + t检验 + BH-FDR
├── backtest/
│   └── dynamic_backtest.py       # N=20块回测 + Top-K分散
└── visualize/
    ├── run_all.py                # 一键生成全部图表
    ├── fig01_cumulative_excess.py
    ├── fig02_drawdown_comparison.py
    └── ... (共7个图表脚本)

output/
├── tables/                       # 回测结果 CSV
└── figures/                      # 可视化 PNG
```

---

## 快速复现

```bash
pip install uv && uv sync

# 1-4. 数据管线
uv run python src/data/fetcher.py              # 拉取 ETF/指数/宏观
uv run python src/data/preprocess.py           # 日频对齐
uv run python src/features/rebuild_mtm_nav.py  # MTM 净值重建
uv run python src/features/build_all_strategies.py  # ETF/映射策略

# 5-7. 特征+匹配+回测
uv run python src/features/build_snapshot.py      # 特征+标签
uv run python src/models/similarity_engine.py      # KNN + FDR
uv run python src/backtest/dynamic_backtest.py --top-k 3  # 回测
```

---

## 最终结果精选

| 版本 | 策略数 | Top-K | 年化超额 | IR | 块胜率 | 最大回撤 |
|------|:-----:|:-----:|:-------:|:-:|:-----:|:-------:|
| v2.4 基线 | 34 | 1 | +6.67% | 0.272 | 53.4% | 48.21% |
| v3.1 Top-K | 58 | **3** | +2.41% | 0.143 | 52.6% | 38.85% |
| **v4.3 去重全池** | **37** | **3** | **+27.80%** | **0.364** | 49.6% | 32.69% |

详细实验对比和基线分析见 `report.md`。

---

## 如何自定义与扩展

### 1. 添加新策略

编辑 `docs/strategy_definitions.csv`，按 `source_type` 分三类：

| source_type | 说明 | rqdata_code 要求 |
|-------------|------|-----------------|
| `etf` | 宽基 ETF 长持 | 留空，由代码名匹配 rqdata |
| `mapped` | ETF 映射策略 | 填写 rqdata 代码 (如 `510050.XSHG`) |
| `mtm` | 交易流水盯市重建 | 留空，从 CSV 重建 |

**ETF/映射策略**：添加一行，`build_all_strategies.py` 自动从 rqdata 拉取价格构建 NAV。

**MTM 策略**：将原始交易 CSV 放入 `data/processed/`，命名 `交易记录{名称}.csv`，
需包含 `trade_time, cash_balance, posi_balance, symbol, btype` 列。
`rebuild_mtm_nav.py` 使用 rqdata 个股收盘价进行盯市重建。

### 2. 添加新特征

两步操作：

**Step 1: 数据源** — 在 `src/data/fetcher.py` 中添加拉取函数，
输出 parquet 到 `data/external/`。

**Step 2: 对齐** — 在 `src/data/preprocess.py` 的 `main()` 中
`load_source()` 并 join 到 `aligned` DataFrame。

**Step 3: 特征计算** — 在 `src/features/build_snapshot.py` 的 `compute_features()` 中
添加新列，保证仅使用 t 时刻前可获取的数据。

**Step 4: 权重配置** — 在 `src/models/similarity_engine.py` 的
`FEATURE_WEIGHTS` 字典中添加权重（宏观/资金面建议 1.5x，技术面 0.8~1.0x）。

### 3. 替换数据源

默认使用 **rqdata**（付费，稳定）。如需切换：

- **ETF/指数**：修改 `fetch_price()` 的 `rqdatac.get_price()` 调用
- **个股**：修改 `fetch_stock_prices()` 的数据源
- **宏观/资金面**：当前部分数据来自 AKShare（`src/data/fetcher.py` 中带标记），
  可替换为其他 API

### 4. 调整参数

| 参数 | 位置 | 说明 |
|------|------|------|
| N (持有期) | `build_snapshot.py` + `similarity_engine.py` + `dynamic_backtest.py` | 默认 20 交易日 |
| KNN K | `similarity_engine.py` | 默认 30 |
| FDR q | `similarity_engine.py` | 默认 0.1 |
| Top-K | `dynamic_backtest.py --top-k` | 推荐 3 |
| 半衰期 | `similarity_engine.py` | 默认 3 年 |
| 质量过滤 | `build_snapshot.py` 顶部常量 | `MTM_FILTER_ENABLED` 等 |
| 特征权重 | `similarity_engine.py` FEATURE_WEIGHTS | 调整各特征影响 |

### 5. 快速对比实验

```bash
# 质量过滤开关
# 修改 build_snapshot.py → MTM_FILTER_ENABLED = False
uv run python src/features/build_snapshot.py
uv run python src/models/similarity_engine.py
uv run python src/backtest/dynamic_backtest.py --top-k 3

# Top-K 对比
uv run python src/backtest/dynamic_backtest.py --top-k 1
uv run python src/backtest/dynamic_backtest.py --top-k 3   # 推荐
uv run python src/backtest/dynamic_backtest.py --top-k 5

# 距离度量对比
uv run python src/models/similarity_engine.py --dist euclidean      # 默认
uv run python src/models/similarity_engine.py --dist mahalanobis    # 马氏
```

### 6. 可视化

```bash
uv run python src/visualize/run_all.py
```

输出 7 张学术标准图表到 `output/figures/`，含累计超额净值曲线、回撤对比、
块超额分布、KNN 近邻时间分布、PCA 特征空间、FDR 阈值、DSR 统计显著性。

---

## 数据依赖

| 数据 | 来源 | 接口 | 缓存 |
|------|------|------|------|
| 宽基指数 OHLCV | rqdata | `get_price` | `data/external/index_prices.parquet` |
| ETF 日行情 | rqdata | `all_instruments` + `get_price` | `data/external/etf_prices.parquet` |
| A 股个股收盘价 | rqdata | `get_price` | `data/external/stock_prices.parquet` |
| 收益率曲线 | rqdata | `get_yield_curve` | `data/external/yield_curve.parquet` |
| 北向资金 | AKShare | `moneyflow_hsgt` | `data/external/northbound_flow.parquet` |
| 融资融券 | AKShare | `margin_detail` | `data/external/margin.parquet` |
| 宏观数据 | AKShare | `macro_china_*` | `data/external/macro.parquet` |
| 策略定义 | 手工维护 | `docs/strategy_definitions.csv` | — |
| 原始交易流水 | Excel → CSV | 转换脚本 | `data/processed/*.csv` |

时间范围: 2016-01-08 ~ 2026-04-30

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `docs/strategy_definitions.csv` | 策略池唯一定义入口 |
| `src/features/build_snapshot.py` | 无泄露特征+标签+质量过滤 |
| `src/models/similarity_engine.py` | KNN + 统计检验核心 |
| `src/backtest/dynamic_backtest.py` | 回测执行引擎 |
| `report.md` | 完整实验报告（含方法、结果、权衡） |

---

## 版本

| Tag | 说明 |
|-----|------|
| v4.3 | 去重全池 37 策略, +27.80% 年化超额 |
| v4.0 | rqdata MTM 重建 + 全池 |
| v3.1 | Top-K 分散, K=3 |
| v2.4 | KNN + N=20 块持有 (基线) |
| v1.x | HMM 状态推断 (已废弃) |
