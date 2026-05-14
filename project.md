# Project: 中国股市历史行情与证券交易策略匹配

> **版本**: v2.4 | **截止**: 2026-05-14 | **方法**: KNN 相似度 + N=20 日块持有

---

## 核心发现

### 1. 胜率 ~45% 是结构性问题，非算法失败

90% 的交易日策略收益率为 0（策略不交易），导致 **超额收益 = -HS300 当日收益**。只要 HS300 上涨日 >45%，日频胜率就天然被压制在 ~46%。指标本身不合理。

**正确指标**: N=20 日 **块胜率** — 每 20 天为一个持有周期，看策略是否跑赢 HS300。

### 2. KNN 相似度匹配具有真实预测价值

| 指标 | 旧(日频切换) | 新(N=20块持有) |
|------|------------|--------------|
| 年化超额 | +1.21% | **+6.67%** |
| 信息比率 | 0.049 | **0.272** |
| 块胜率 | — | **53.4%** |
| 样本内块胜率 | — | 53.4% |
| 样本外块胜率 | — | 51.6% |
| DSR p | <0.001 | **<0.001** |

### 3. 质量过滤适得其反

对策略实施历史跟踪记录过滤(要求胜率>50%)反而降低了整体表现(+6.67%→+3.55%)。原因是小样本噪声: 好策略可能在3次选择中运气差1次就被永久排除。

### 4. 策略动量无预测力

过去 20 日块的表现与未来 20 日块的表现不相关(均值相关性 +0.08, 中位数 ~0.00)。"挑上一块最牛的策略"只有 43.6% 胜率(比随机还差)。

---

## 最终结果

| 指标 | KNN 基线 | KNN+质量过滤 | 排名选择器 |
|------|----------|-------------|----------|
| 年化超额 | **+6.67%** | +3.55% | -5.82% |
| 信息比率 | **0.272** | 0.149 | -0.226 |
| 日胜率 | 46.6% | 46.0% | 44.7% |
| 块胜率(20日) | **53.4%** | 50.4% | 42.0% |
| 样本外块胜率 | **51.6%** | 48.4% | — |
| DSR | **显著** | 显著 | 不显著 |

## 管线

```bash
python src/features/build_etf_strategies.py          # 30 合成策略
python src/features/build_snapshot.py                  # 无泄露快照
python src/models/similarity_engine.py                 # KNN+统计检验
python src/backtest/dynamic_backtest.py                # ← 基线结果
python src/backtest/dynamic_backtest.py --quality-filter  # 质量过滤
```

## 命令

```bash
# 回测 KNN 基线 (推荐)
uv run python src/backtest/dynamic_backtest.py

# 块胜率计算
uv run python -c "
import pandas as pd
r = pd.read_csv('output/tables/backtest_knn_qf_off.csv', parse_dates=['trade_date'])
r['bid'] = r.index // 20
be = r.groupby('bid')['excess_return'].apply(lambda x: (1+x).prod()-1)
print(f'Block WR: {(be>0).mean():.1%}, Mean: {be.mean():+.2%}')
"
```

## 策略池 (34个)

30 个合成策略(来自 HS300/ZZ500/CYB 指数 + 21 个申万行业):
- 长持 24个, 均线择时 3个, 动量择时 3个
- 4 个 MTM 重建策略(桥水全天候/均衡持仓/军工/动量轮动)

## 受保护文件

`data/raw/*`, `pyproject.toml`, `uv.lock`, `docs/project-def.md`, `docs/todo.md`
