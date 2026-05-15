# MSA — 相似行情匹配与动态策略选择系统

基于 KNN 相似度 + 统计检验 + Top-K 分散的动态策略选择系统。

核心逻辑：用市场特征(F(t): 动量/波动/回撤/宏观)找出最相似的 30 个历史交易日，
检验哪些策略在那些历史日之后超额显著为正，等权持有 Top-3 策略 20 个交易日。

---

## 效果对比

| 指标 | K=1 (单策略) | **K=3 (推荐)** | K=5 (过度) |
|------|:-----------:|:-------------:|:---------:|
| 年化超额 | +0.31% | **+2.41%** | +1.92% |
| 信息比率 (IR) | 0.012 | **0.143** | 0.135 |
| 块胜率 (20日) | 54.1% | 52.6% | **56.4%** |
| 最大回撤 | 60.36% | **38.85%** | 40.19% |
| DSR p-value | < 0.001 | **< 0.001** | < 0.001 |
| 样本内年化 | +0.83% | +2.36% | +3.35% |
| **样本外年化** | **-1.40%** | **+2.56%** | -2.76% |
| 样本外块胜率 | 54.8% | **54.8%** | 54.8% |

**K=3 最优**：IR 提升 12x、回撤降低 35%、样本外由负转正。

---

## 策略池

| 类型 | 数量 | 来源 | 定义文件 |
|------|------|------|---------|
| total_summary 映射 | 41 | 策略名 → rqdata ETF 名称匹配 | `docs/strategy_definitions.csv` |
| 宽基 ETF (长持) | 13 | 沪深300/纳指/黄金等 | `docs/strategy_definitions.csv` |
| MTM 重建 | 4 | 原始交易流水盯市重建 | — |

**总计 58 个策略**，单一来源：`docs/strategy_definitions.csv`

---

## 快速复现

```bash
# 1. 环境
pip install uv
uv sync

# 2. 拉取 ETF/指数行情 (缓存)
uv run python src/data/fetcher.py

# 3. 统一日频表
uv run python src/data/preprocess.py

# 4. 构建 58 个策略 NAV
uv run python src/features/build_all_strategies.py

# 5. 无泄露特征+标签
uv run python src/features/build_snapshot.py

# 6. KNN 相似度+FDR 检验
uv run python src/models/similarity_engine.py

# 7. 动态回测 (对比不同分散度)
uv run python src/backtest/dynamic_backtest.py --top-k 1
uv run python src/backtest/dynamic_backtest.py --top-k 3   # 推荐
uv run python src/backtest/dynamic_backtest.py --top-k 5
```

---

## 核心文件

```
docs/strategy_definitions.csv          # 58 个策略定义 (唯一入口)
src/features/build_all_strategies.py   # 读取定义 → rqdata → 构建净值
src/features/build_snapshot.py         # 无泄露特征+标签库
src/models/similarity_engine.py        # KNN + t检验 + BH-FDR
src/backtest/dynamic_backtest.py       # N=20 块回测 + Top-K 分散
```

---

## 数据依赖

| 数据 | 来源 | 接口 | 缓存路径 |
|------|------|------|---------|
| 指数/ETF 日行情 | rqdata | `get_price` | `data/external/*.parquet` |
| 策略定义 | 手动 | `docs/strategy_definitions.csv` | — |
| 原始交易流水 | Excel | CSV | `data/processed/*.csv` |
| 收益率曲线/北向/融资/宏观 | rqdata/AKShare | — | `data/external/*.parquet` |

时间范围：2016-01-08 ~ 2026-04-30

---

## 方法

1. **特征 F(t)**: 16 维: 动量(20/60日)、RSI(14)、MA5>MA20 排列、偏度、涨跌比、回撤恢复天数、波动率、北向资金、融资余额、PMI/CPI/M2/PPI
2. **KNN**: 加权欧氏距离, 30 个最近邻, 时间衰减(半衰期 3 年), 约束 τ+N<t
3. **统计检验**: 单边 t 检验 + BH-FDR(q=0.1)
4. **选择**: Top-3 显著策略等权持有 20 日, 到期再平衡
5. **显著性验证**: Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

---

## 版本

| Tag | 说明 |
|-----|------|
| v3.1 | Top-K 分散, K=3 最优 |
| v3.0 | 统一策略定义 + 删除过时代码 |
| v2.8 | 清理注册表至 58 策略 |
| v2.4 | KNN + N=20 块持有 (基线) |
