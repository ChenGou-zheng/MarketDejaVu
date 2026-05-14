# MSA — A股行情聚类与量化策略匹配

MA304 多元统计分析课程项目。使用 HMM 对 2016 年以来的 A 股多维行情特征进行隐状态推断，建立市场状态标签体系。

---

## 快速复现

```bash
# 1. 环境初始化
pip install uv
uv sync

# 2. 策略数据转换（若 raw/ 中已有 .xlsx 文件）
python src/data/convert_xlsx_to_csv.py
python src/data/strategy_nav.py

# 3. 行情+宏观+资金流数据拉取（首次 3-5 分钟，之后有缓存秒过）
python src/data/fetcher.py

# 4. 整合为统一日频表
python src/data/preprocess.py

# 5. HMM 市场状态推断（自动 BIC 选 K，Train/Val/Test 三集切分）
python src/models/regime_hmm.py

# 6. XGBoost 特征重要性 + Zigzag 对照组
python src/models/regime_analysis.py
```

运行完毕后，所有输出在 `data/processed/`、`data/external/`、`output/tables/` 三个目录下。

---

## 项目结构

```
MSA/
├── src/
│   ├── data/
│   │   ├── convert_xlsx_to_csv.py   # 批量 Excel → CSV（60 张工作表）
│   │   ├── strategy_nav.py          # 逐笔交易流水 → 日频净值（39 个策略）
│   │   ├── fetcher.py               # rqdatac + AKShare 全量数据拉取（7 类数据源）
│   │   └── preprocess.py            # 日频对齐 + 前向填充 → aligned_daily.parquet
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   ├── regime_hmm.py            # 特征工程 (B1-B5 26 维) + HMM 三集切分
│   │   └── regime_analysis.py       # XGBoost 特征重要性 + Zigzag 对照
│   ├── backtest/
│   └── visualize/
├── data/
│   ├── raw/                         # 4 个原始 Excel（仅留档）
│   ├── processed/                   # 净值 CSV、策略注册表、aligned_daily.parquet、hmm_states_*.csv
│   └── external/                    # 7 个 .parquet 缓存（行情/宏观/资金流）
├── notebooks/                       # Jupyter 探索
├── output/tables/                   # 转移矩阵、状态特征、特征重要性、状态分布对比
├── docs/                            # 项目描述、计划、rqdatac 文档索引
├── pyproject.toml
└── uv.lock
```

---

## 数据说明

> **注意**：`data/raw/` 中 4 个 Excel 存在跨文件重复策略（如"沪深300增强"同时出现在两个文件中），
> `strategy_nav.py` 已自动选取时间跨度最长的版本。`data/raw/` 仅作留档。

| 类别 | 内容 | 来源 | 频率 | 原始接口 |
|------|------|------|------|----------|
| 宽基指数 | 沪深300/中证500/创业板指 OHLCV | rqdatac | 日 | `get_price` |
| 收益率曲线 | 1Y / 10Y 国债收益率 + 期限利差 | rqdatac | 日 | `get_yield_curve` |
| 北向资金 | 沪股通+深股通日净买入 | AKShare | 日 | `stock_hsgt_hist_em` |
| 融资融券 | 上交所融资余额 | AKShare | 日 | `stock_margin_sse` |
| 宏观 | PMI / CPI 同比 / M2 同比 / PPI 同比 | AKShare | 月→日前向填充 | `macro_china_pmi` 等 |
| 申万行业 | 31 个一级行业指数日收盘价 | AKShare | 日 | `index_hist_sw` |
| 情绪代理 | 已实现波动、换手变化、动量差 | 从价格自算 | 日 | — |
| 策略流水 | `cash_balance + posi_balance` | Excel | 逐笔→日频 | `total_summary.csv` 清洗版 |

时间范围：
- 行情/宏观：2016-01-08 ~ 2026-04-30
- 策略数据：2020-01-01 ~ 2026-03-31（不同策略覆盖期不同，详见 `strategy_registry.csv`）

---

## 训练 / 验证 / 测试切分

| 集合 | 时间 | 天数 | 用途 |
|------|------|------|------|
| Train | 2016-04-01 ~ 2024-12-31 | 2,283 | HMM 拟合 + 状态定义 |
| Val | 2025-01-01 ~ 2025-12-31 | 261 | 状态外推验证 |
| Test | 2026-01-01 ~ 2026-04-30 | 86 | 最终推荐评估 |

HMM 仅拟合训练集。Scaler 和模型参数在训练集上估计后，apply 到验证集和测试集。
这一设计确保验证/测试数据不参与状态定义，模拟真实外推场景。

---

## 特征空间 (B1-B5, 26 维)

### B1 — 价格/波动 (8)
`ret_HS300` `ret_ZZ500` `ret_CYB` — 三指数日收益
`vol_20d` `vol_60d` — 20/60 日年化波动率
`skew_60d` `kurt_60d` — 60 日收益偏度/峰度
`max_dd_60d` — 60 日最大回撤

### B2 — 资金面 (3)
`northbound_5d` — 北向资金 5 日累计净流入
`margin_chg_20d` — 融资余额 20 日变化率
`m2_yoy` — M2 同比增速

### B3 — 情绪面 (3)
`realized_vol_20d` — 20 日已实现波动率
`turnover_chg` — 5 日换手率变化率
`sent_momentum` — 5 日 − 20 日动量差（短期情绪过热/冷却）

### B4 — 结构/风格 (7)
`size_style` — 中证500收益 − 沪深300收益（大小盘）
`growth_style` — 创业板收益 − 沪深300收益（成长/价值）
`momentum_20d` `momentum_60d` — 短期/中期动量
`industry_dispersion` — 31 个申万行业日收益标准差（普涨 vs 二八分化）
`sector_rotation` — 行业排名变化速度
`vol_ratio_500_300` — 中证500 / 沪深300 成交量比

### B5 — 宏观 (5)
`yield_slope` — 10Y − 1Y 国债利差
`yield_10y` — 10 年期国债收益率
`pmi_mfg` — 制造业 PMI
`cpi_yoy` — CPI 当月同比
`ppi_yoy` — PPI 当月同比

---

## HMM 结果（仅训练集拟合，K=6）

| State | 天数 | 占比 | 累计收益 | 年化波动 | Sharpe | 中文标签 |
|-------|------|------|---------|---------|--------|---------|
| 0 | 181 | 7.9% | +14.21% | 32.17% | +0.73 | 高波震荡 |
| 1 | 432 | 18.9% | −37.05% | 12.42% | −2.11 | 持续阴跌 |
| 2 | 534 | 23.4% | −39.57% | 22.19% | −0.96 | 高波下跌 |
| 3 | 411 | 18.0% | +33.89% | 10.51% | +1.76 | 低波上涨 |
| 4 | 20 | 0.9% | −1.59% | 36.66% | −0.37 | 恐慌暴跌 |
| 5 | 705 | 30.9% | +113.61% | 14.17% | +1.99 | 慢牛趋势 |

转移矩阵（训练集估计）：
```
        to_0   to_1   to_2   to_3   to_4   to_5
from_0  .936   .000   .016   .000   .000   .048
from_1  .002   .984   .005   .000   .000   .009
from_2  .000   .000   .985   .000   .000   .015
from_3  .000   .000   .002   .998   .000   .000
from_4  .050   .000   .000   .000   .950   .000
from_5  .015   .010   .003   .001   .001   .969
```

状态自持概率 93.6%~99.8%，与市场状态的持续性一致。State 4 仅出现 20 天（恐慌暴跌），为极端厚尾事件。

### 验证集外推（2025 全年，261 天）

| State | 天数 | 占比 | 说明 |
|-------|------|------|------|
| 0 | 89 | 34.1% | 2025 年 9 月底政策转向后的快速反弹期 |
| 1 | 169 | 64.8% | 2025 年上半年持续阴跌 |
| 5 | 3 | 1.1% | 零星慢牛日 |

- **未出现新状态** — 验证集全部归入训练集已有状态
- State 2/3/4（高波下跌、低波上涨、恐慌暴跌）未在 2025 年触发

### 测试集外推（2026 年 1-4 月，86 天）

| State | 天数 | 占比 | 说明 |
|-------|------|------|------|
| 0 | 17 | 19.8% | 短期修复反弹 |
| 1 | 68 | 79.1% | 2026 年以持续阴跌为主 |
| 5 | 1 | 1.2% | 零星慢牛日 |

- **无新状态**，模型外推稳定
- 与实盘一致：2026 年初市场偏弱

---

## XGBoost 特征重要性

| 排名 | 特征 | 维度 | 重要性 |
|------|------|------|--------|
| 1 | `margin_chg_20d` | 资金面 | 17.67% |
| 2 | `skew_60d` | 价格 | 11.10% |
| 3 | `pmi_mfg` | 宏观 | 9.68% |
| 4 | `max_dd_60d` | 价格 | 8.35% |
| 5 | `momentum_20d` | 结构 | 8.03% |

资金面 + 宏观合计占 ~27%，验证引入外生变量的必要性。
纯价格特征不足以区分市场状态。

### 方法说明

XGBoost 以 26 个特征为输入、HMM 训练集状态标签为目标做多分类。训练完成后
`feature_importances_` 给出各特征对分类决策的贡献度。AUC 不代表预测能力
（因为状态标签来自无监督 HMM），仅用于特征排名参考。

---

## Zigzag 对照组

Zigzag 参数：上涨阈值 15%、下跌阈值 -10%、最小段长 20 日。

**Zigzag vs HMM Adjusted Rand Index = 0.0694**（几乎不一致）。

这证明了：
1. 基于单一价格曲线的硬边界分段与多维数据驱动的 HMM 状态推断结果差异巨大
2. 人为切出的段边界对阈值高度敏感，在不同波动环境下缺乏鲁棒性
3. HMM 的序贯概率切换更适合刻画市场状态的渐进过渡

---

## 策略注册表说明

`strategy_registry.csv` 包含 39 个策略的关键信息：名称、起止日期、交易日数、全周期收益/波动/最大回撤、源文件数。

### 数据质量注意事项

1. **total_summary.csv 中的文本格式数字**：16 行"累计收益率"列的值包含日期标注
   （如 `141.60%（21年8月至今）`），已清洗为 `total_summary_clean.csv`。
   提取规则：取 `%` 前数值除以 100 → 小数收益率。

2. **短覆盖策略**：10 个策略不足 200 个交易日。其中"交易记录300"（沪深300增强）
   仅 16 个交易日，在行情回溯中不可用。

3. **综合全 vs 综合拆分**：综合全仅覆盖 2024-01 起，应使用综合拆分1/2
   （覆盖 2022-01 起，各 275/268 天）。

4. **去重已正确**：同策略在多个 Excel 出现时，`strategy_nav.py` 自动选取
   时间跨度最长的版本（`source_files=1`）。

---

## 困难与解决方案

### 1. Windows PowerShell 编码问题
- **现象**: emoji 字符在 GBK 编码下报 `UnicodeEncodeError`
- **解决**: 移除 emoji，改用 ASCII 标记；必要时 `$env:PYTHONIOENCODING="utf-8"`

### 2. PowerShell 目录操作兼容性
- **现象**: `mkdir -p` 不存在；`Move-Item` 对中文文件名编码错误
- **解决**: `New-Item -ItemType Directory -Force` 逐个创建；`Get-ChildItem -Filter | Move-Item` 管线移动

### 3. hatchling 构建错误
- **现象**: `uv sync` 报 hatchling 找不到名称匹配的包目录
- **解决**: `[tool.hatch.build.targets.wheel] packages = ["src/data", ...]`

### 4. rqdatac 试用版权限严重受限
- **现象**: 因子模型/一致预期/北向/宏观等 10+ API 全部 `PermissionDenied`
- **可用**: 仅 `get_price`、`get_yield_curve`、`index_components`
- **解决**: 用 AKShare 替代宏观/资金流/行业数据；结构/情绪特征用价格代理计算

### 5. AKShare API 接口名探测
- **现象**: 文档入口多，API 名频繁变更，直接猜测全失败
- **解决**: `[a for a in dir(ak) if 'keyword' in a.lower()]` 搜索 + 逐个测试

### 6. AKShare DataFrame 列名不匹配
- **现象**: 中文列名与预期不符（'制造业-指数' vs '制造业-指标' 等）
- **解决**: 逐 API `repr(list(df.columns))` 打印后修正映射；M2 列名含中文用 `if "M2" in c and "同比增长" in c` 动态匹配

### 7. pandas CSV 序列化索引丢失
- **现象**: `to_csv` → `read_csv` 往返后 join 报 `no overlapping index names`
- **解决**: 所有缓存改用 Parquet 格式

### 8. rqdatac MultiIndex 处理
- **现象**: `get_price` 返回 MultiIndex 或 DatetimeIndex 不统一
- **解决**: `isinstance(idx, MultiIndex)` 检测 + `get_level_values(-1)` 提取 + `index.name = None`

### 9. pyarrow 依赖缺失
- **解决**: `uv add pyarrow`

### 10. HMM 协方差矩阵收敛问题
- **现象**: `covariance_type="full"` 时 26×27/2=351 参数/状态，严重过参数化
- **解决**: 改用 `covariance_type="diag"`（每状态 26 参数），收敛稳定

### 11. SW 行业轮动速度计算超时
- **解决**: 每 5 日采样计算排名变化，前向填充到日频

### 12. 数据清洗：文本格式数字
- **现象**: `total_summary.csv` 中 16 行数值被拼接日期标注
- **解决**: `re.match(r'([\d.]+)\s*%', s)` 提取百分数数值部分

### 13. 初始 HMM 未做 Train/Test 切分
- **现象**: 第 1 版在全部 2016-2026 数据上拟合 HMM，测试集数据参与了状态定义
- **解决**: 改为 Train(2016-2024) / Val(2025) / Test(2026) 三集切分，HMM 仅拟合训练集，验证集和测试集只做 predict

---

## 工程估算

| 项目 | 数值 |
|------|------|
| 训练集天数 | 2,283 (2016-04 ~ 2024-12) |
| 验证集天数 | 261 (2025 全年) |
| 测试集天数 | 86 (2026 年 1-4 月) |
| 特征维度 | 26 (B1:8 + B2:3 + B3:3 + B4:7 + B5:5) |
| HMM 最优 K | 6 (BIC 在训练集上选择) |
| HMM BIC | 129,409 (diag 协方差) |
| 协方差参数/状态 | 26 (diag) vs 351 (full) |
| 策略数量 | 39 (4 个 Excel → 60 张工作表 → 去重) |
| 代码文件 | 6 个核心 .py |
| 数据源 | 7 类 (rqdatac 2 + AKShare 4 + 自算 1) |
| rqdatac 可用率 | 2/10+ API |
| Zigzag vs HMM | ARI = 0.0694 |

---

## 风险提示

- 历史行情相似性不代表未来必然重复，HMM 状态推断仅供研究参考
- rqdatac 试用版 API 受限，风格因子/一致预期/北向行业分布用代理变量替代
- 宏观数据为月度频率前向填充到日频，存在滞后
- 策略净值从交易流水的 `cash_balance + posi_balance` 重建，非交易日持股市值变动无法观测
- 测试集仅 4 个月 (86 天)，统计显著性有限
- 训练集仅 ~2300 天，HMM 6 状态划分存在小样本局限，State 4 仅 20 天可能不稳定
