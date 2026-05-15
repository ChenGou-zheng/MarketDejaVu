# MSA — 相似行情匹配与动态策略选择系统

基于 KNN 相似度 + 统计检验 + Top-K 分散的动态策略选择系统。

核心逻辑：用市场特征(F(t): 动量/波动/回撤/宏观)找出最相似的 30 个历史交易日，
检验哪些策略在那些历史日之后超额显著为正，等权持有 Top-3 策略 20 个交易日。

---

## 效果对比 — 改进历程

| 版本 | 策略数 | 策略来源 | Top-K | 年化超额 | IR | 块胜率 | 最大回撤 | OOS 年化 | OOS 块WR |
|------|-------|---------|:----:|:-------:|---|:-----:|:-------:|:--------:|:--------:|
| v2.4 基线 | 34 | 合成数据(指数/行业×机械规则) | 1 | +6.67% | 0.272 | 53.4% | 48.21% | — | 51.6% |
| v2.5 真实 ETF | 17 | 13 个 rqdata ETF + 4 MTM | 1 | -0.72% | -0.030 | 54.1% | 43.11% | -9.60% | 48.4% |
| v2.6 映射扩容 | 58 | 28 映射+26 ETF+4 MTM(重复) | 1 | +11.58% | 0.441 | 54.9% | 48.21% | +24.83% | 61.3% |
| v2.7 名称匹配 | 80 | 41 映射+35 ETF+4 MTM(含垃圾) | 1 | +8.56% | 0.316 | 52.3% | 41.07% | +8.42% | 59.4% |
| v2.8 清理注册表 | 58 | 41 映射+13 ETF+4 MTM | 1 | +0.59% | 0.022 | 54.9% | 52.62% | +13.40% | 58.1% |
| v3.0 统一定义 | 58 | 同上(代码重构) | 1 | +0.31% | 0.012 | 54.1% | 60.36% | -1.40% | 54.8% |
| v3.1 Top-K | 58 | 同上 | **3** | +2.41% | 0.143 | 52.6% | 38.85% | +2.56% | 54.8% |
| 修正映射错误 | 53 | 39 映射+13 ETF+4 MTM(修正后) | 3 | +3.60% | 0.216 | 53.4% | 30.65% | -0.14% | 51.6% |
| 纯 ETF 测试 | 13 | 仅 ETF(无映射/MTM) | 3 | -0.69% | -0.052 | 48.9% | 51.76% | +2.60% | 51.6% |
| 纯 MTM 测试 | 4 | 4 个 rqdata MTM | 1 | +9.50% | 0.327 | 51.9% | 32.98% | +2.06% | 51.6% |
| 纯 MTM 测试 | 10 | 10 个 rqdata MTM(无ETF/映射) | 3 | -11.49% | -0.048 | 53.1% | 96.62% | +9.76% | 61.3% |
| **纯 MTM 全部(无质量控制)** | **37** | **全部 37 个 MTM(不过滤)** | **3** | **≈∞%** | **∞** | **91.4%** | 91.99% | **≈∞%** | **90.3%** |
| **v4.0 全池** | **63** | 39 映射+13 ETF+**11 rqdata MTM** | **3** | **+23.66%** | **0.311** | 50.4% | 31.43% | **+6.77%** | **54.8%** |

> 37 纯 MTM(无质量控制)结果不可用: 年化超额为 ∞(计算机浮点数溢出), 因包含 26 个 MTM 重建失败的策略(日收益波动 >±20% 占比 4%~80%)。这证明了质量过滤的必要性——若不过滤, KNN 会被虚假极端收益支配。

### 基线对比

| 基线 | 年化超额 | 块胜率 | 说明 |
|------|:-------:|:-----:|------|
| 随机选择 (概率期望) | ~0% | ~50% | 每块随机选一个策略 |
| 等权持有全部策略 | **+5.31%** | **56.0%** | 62 策略简单平均 |
| 始终持有动量轮动 | +11.50% | 50.0% | 最佳单一策略 |
| **KNN 系统 (v4.0 K=3)** | **+23.66%** | 50.4% | 全池选择结果 |

等权持有全部策略的块胜率(56.0%)高于 KNN 系统(50.4%), 说明 **KNN 的选择提高了收益(+23.66% vs +5.31%)但牺牲了稳定性**。高收益来自少数大赢交易日, 非持续稳定输出。

---

## 策略池

| 类型 | 数量 | 说明 |
|------|------|------|
| total_summary 映射 → ETF | 39 | 名称模糊匹配到交易所 ETF |
| 宽基 ETF 长持 | 13 | HS300/纳指/黄金等 |
| rqdata MTM 重建 | 10 | 从原始交易 CSV + rqdata 个股收盘价重建 |
| **总计** | **62** | — |

### MTM 净值重建 (rqdata 个股收盘价)

对 37 个原始交易 CSV，使用 rqdata 获取所有持仓个股的日收盘价(3009 只股票, 2020年起)，
以第一笔银证转入金额为初始本金，逐日计算 NAV = 现金 + Σ(股数×收盘价)。

**通过质量过滤 (10 个)**:
交易记录300, 交易记录etf动量轮动, 交易记录军工, 交易记录双创,
交易记录均衡持仓, 交易记录房地产, 交易记录桥水全天候, 交易记录煤炭,
交易记录科创, 交易记录锤子策略

**失败 (26 个, 日收益波动 >±20% 比例超 3%)**:
交易记录1000, 交易记录800, 交易记录etf动量改, 交易记录中证2000,
交易记录全球etf, 交易记录养殖, 交易记录创业板, 交易记录动量趋势,
交易记录化工, 交易记录半导体, 交易记录国企, 交易记录形态, 交易记录旅游,
交易记录机器人, 交易记录杠铃, 交易记录游戏, 交易记录申万动量,
交易记录百亿etf一只, 交易记录百亿etf两只, 交易记录红利,
交易记录综合拆分1, 交易记录综合拆分2, 交易记录行业, 交易记录计算机,
交易记录通信, 交易记录酒

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

# 2. 拉取数据 (ETF/指数/个股)
uv run python src/data/fetcher.py

# 3. 统一日频表
uv run python src/data/preprocess.py

# 4. 提取股票代码清单 (首次)
uv run python -c "
from pathlib import Path; import pandas as pd, json, os, re
codes = set()
for f in Path('data/processed').glob('交易记录*.csv'):
    df = pd.read_csv(f)
    if 'symbol' not in df.columns: continue
    for s in df['symbol'].dropna().unique():
        s=str(s).strip()
        if s and '以上是' not in s: codes.add(s)
rq = [c.replace('SHSE.','')+'.XSHG' if c.startswith('SHSE.') else c.replace('SZSE.','')+'.XSHE' for c in codes]
rq = [c for c in rq if not c.startswith(('900','200'))]
with open('data/external/stock_codes.json','w') as f: json.dump(rq, f)
print(f'{len(rq)} stock codes saved')
"
python -c "from src.data.fetcher import fetch_stock_prices; fetch_stock_prices()"

# 5. 构建 MTM 策略净值
uv run python src/features/rebuild_mtm_nav.py

# 6. 构建全量策略 NAV (ETF + mapped)
uv run python src/features/build_all_strategies.py

# 7. 无泄露特征+标签
uv run python src/features/build_snapshot.py

# 8. KNN 相似度+FDR 检验
uv run python src/models/similarity_engine.py

# 9. 动态回测
uv run python src/backtest/dynamic_backtest.py --top-k 3   # 推荐
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
| v4.0 | rqdata 个股收盘价 MTM 重建 + 62 策略全池 |
| v3.1 | Top-K 分散, K=3 最优 |
| v3.0 | 统一策略定义 + 删除过时代码 |
| v2.8 | 清理注册表至 58 策略 |
| v2.4 | KNN + N=20 块持有 (基线) |
