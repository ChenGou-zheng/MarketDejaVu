# Ricequant Docs Index

## 文档目录

- # [参数配置](https://www.ricequant.com/doc/sources/rqalpha-plus/api/config.md) 各种类型的详尽的参数配置，用于传入 RQalphaPlus 的入口函数，或赋值给策略的 `__config__` 全局变量。
- # [入口函数](https://www.ricequant.com/doc/sources/rqalpha-plus/api/entrypoint.md) 用于运行回测的函数。
- # [约定函数](https://www.ricequant.com/doc/sources/rqalpha-plus/api/callback.md) 策略中可选实现的函数，这些函数会在特定的时间点被调用。
- # [交易接口](https://www.ricequant.com/doc/sources/rqalpha-plus/api/order-api.md) 策略中用于创建订单的函数。
- # [转账融资接口](https://www.ricequant.com/doc/sources/rqalpha-plus/api/transfer-financing-api.md) 策略中与转账融资相关的函数
- # [仓位查询接口](https://www.ricequant.com/doc/sources/rqalpha-plus/api/position-api.md) 策略中用于查询当前持仓的函数。
- # [数据查询接口](https://www.ricequant.com/doc/sources/rqalpha-plus/api/data-api.md) 策略中用于查询行情数据、财务数据等的函数。注：与 rqdata 有区别。
- # [其他接口](https://www.ricequant.com/doc/sources/rqalpha-plus/api/other-api.md) 合约订阅，指标计算，股票组合优化，以及画图等函数。
- # [类](https://www.ricequant.com/doc/sources/rqalpha-plus/api/types.md) 策略中会用到的类。
- # [枚举常量](https://www.ricequant.com/doc/sources/rqalpha-plus/api/enums.md) 策略中会用到的枚举常量。

## RQAlphaPlus 简介

- # [快速上手](https://www.ricequant.com/doc/sources/rqalpha-plus/doc/quick-start.md) 帮助您快速了解和使用回测
- # [进阶教程](https://www.ricequant.com/doc/sources/rqalpha-plus/doc/advance-tutorial.md) 账户和持仓配置、回测频率、支持的标的品种、事前风控、模拟撮合、自定义基准、出入金、管理费用、增量回测、定时器等功能
- # [常见问题](https://www.ricequant.com/doc/sources/rqalpha-plus/doc/question.md)
- # [示例策略](https://www.ricequant.com/doc/sources/rqalpha-plus/doc/example.md)
- # [更新履历](https://www.ricequant.com/doc/sources/rqalpha-plus/doc/changelogs.md)

## RQData HTTP API 手册

- # [数据获取](https://www.ricequant.com/doc/sources/rqdata/http/data-process.md) 通过 HTTP 接口的数据获取的流程和使用方法。
- # [请求示例](https://www.ricequant.com/doc/sources/rqdata/http/examples.md) Python、R、Matlab、Java 等多种编程语言的请求示例。
- # [接口方法](https://www.ricequant.com/doc/sources/rqdata/http/interface-method.md) 举例几种 Python API 不同返回类型的接口获取方式

## RQData Python API 手册

- # [RQData 使用说明](https://www.ricequant.com/doc/sources/rqdata/python/manual.md)
- [基础知识](https://www.ricequant.com/doc/sources/rqdata/python/manual.md#rqdata-basics) 关于 RQData 的架构、版本号、流量配额等信息将在这里阐述。
- [上手指引](https://www.ricequant.com/doc/sources/rqdata/python/manual.md#rqdata-get-started) 如果您是初次使用 RQData，这里提供了一些信息使您能快速熟悉 RQData 的整体使用方式。
- # API 参考
- [跨品种通用 API](https://www.ricequant.com/doc/sources/rqdata/python/generic-api.md) 介绍了 RQData 中对于所有金融标的都通用的 API，是整个 RQData 的基础 API 集合，也是使用最频繁的几个 API。主要包括了查询合约信息、查询历史行情、查询实时行情、检索交易日历等功能。
- [A 股](https://www.ricequant.com/doc/sources/rqdata/python/stock-mod.md) 查询财务数据、分红派息、拆股并股、流通股、板块及行业分类、融资融券、南北向资金、公告相关等及其他股票市场特有信息的 API。
- [港股](https://www.ricequant.com/doc/sources/rqdata/python/stock-hk.md) 查询行情、复权因子、财务数据、行业分类等
- [金融、商品期货](https://www.ricequant.com/doc/sources/rqdata/python/futures-mod.md) 主力合约、仓单数据、升贴水、交易参数等期货市场特有信息的 API。
- [金融、商品期权](https://www.ricequant.com/doc/sources/rqdata/python/options-mod.md) 期权合约、希腊字母、主力月份、PCR/skew 衍生指标等。
- [指数、场内基金](https://www.ricequant.com/doc/sources/rqdata/python/indices-mod.md) 获取指数值、成分及权重。
- [基金](https://www.ricequant.com/doc/sources/rqdata/python/fund-mod.md) 公募基金、ETF 的信息及估值、成分、持仓变动、份额变动、基金经理、分红等信息。
- [可转债](https://www.ricequant.com/doc/sources/rqdata/python/convertible-mod.md) 可转债信息、转债所对应的股票标的信息、强赎、回售、现金流、衍生指标、评级数据等信息。
- [风险因子](https://www.ricequant.com/doc/sources/rqdata/python/risk-factors-mod.md) 获得米筐自研 A 股风险因子模型、包含因子协方差、特异收益率及风险、个股暴露度等数据。
- [现货](https://www.ricequant.com/doc/sources/rqdata/python/spot-goods.md) 获取上海黄金现货交易所上市的现货
- [货币市场](https://www.ricequant.com/doc/sources/rqdata/python/repo.md) 获取国债回购行情、上海银行间同业拆放利率
- [宏观经济数据](https://www.ricequant.com/doc/sources/rqdata/python/macro-economy.md) 获取存款准备金率、货币供应量、宏观因子
- [另类数据](https://www.ricequant.com/doc/sources/rqdata/python/alternative-data.md) 获取一致预期、新闻舆情、ESG 评价数据
- [米筐特色指数](https://www.ricequant.com/doc/sources/rqdata/python/ricequant-index.md) 米筐特色指数行情、成分及指数编制规则等信息
- # [更新履历](https://www.ricequant.com/doc/sources/rqdata/python/changelogs.md)

## RQFactor API 手册

- # [内置因子](https://www.ricequant.com/doc/sources/rqfactor/api/biult-in-factor.md) 解释 RQFactor 内置的行情、财务、和技术类因子的名称、数据来源及调用方式。
- # [内置算子](https://www.ricequant.com/doc/sources/rqfactor/api/built-in-operators.md) 详解数学运算、时间序列、横截面等预设算子的参数含义、格式要求与配置逻辑。
- # [自定义算子](https://www.ricequant.com/doc/sources/rqfactor/api/custom-operators.md) 说明基于`CombinedFactor`/`RollingWindowFactor`等抽象类开发算子时，运算函数参数、输入因子参数、返回格式的配置规范。
- # [因子计算](https://www.ricequant.com/doc/sources/rqfactor/api/factor-calculation.md) 解释`execute_factor`函数中参数的配置要求，及数据预处理（复权、停牌填充）的参数规则。
- # [因子检验](https://www.ricequant.com/doc/sources/rqfactor/api/factor-test.md) 详解因子检验中“预处理”、“因子分析器”、“管道构建”和“执行计算”不同步骤的的配置方式，及结果输出参数的设置逻辑。

## RQFactor用户指南

- # [快速上手](https://www.ricequant.com/doc/sources/rqfactor/manual/quick-start.md) 帮助您快速了解因子开发流程
- # [进阶理解](https://www.ricequant.com/doc/sources/rqfactor/manual/advance-tutorial.md) 帮助您快速掌握自定义算子与因子
- # [使用示例](https://www.ricequant.com/doc/sources/rqfactor/manual/example.md)

## RQOptimizer API 手册

- # [选股API](https://www.ricequant.com/doc/sources/rqoptimize/api/select-stock.md) 介绍选股API的使用及参数详细说明
- # [优化器API](https://www.ricequant.com/doc/sources/rqoptimize/api/optimizer.md) 介绍优化器API的使用及参数详细说明

## RQOptimizer 用户指南

- # [快速上手](https://www.ricequant.com/doc/sources/rqoptimize/doc/quick-start.md)
- 帮助您快速了解优化器使用流程
- # [代码示例](https://www.ricequant.com/doc/sources/rqoptimize/doc/example.md)

## RQPAttr API文档目录

- # [归因API](https://www.ricequant.com/doc/sources/rqpattr/api/pattr-api.md) 介绍RQPAttr API

## 简介

- # [归因模型详解](https://www.ricequant.com/doc/sources/rqpattr/doc/model-introduction.md)
- 介绍权益类 Brinson 行业归因和因子归因的原理
- # [代码示例](https://www.ricequant.com/doc/sources/rqpattr/doc/example.md)

## Ricequant SDK-米筐本地量化开发工具套件文档

- # [操作手册](https://www.ricequant.com/doc/sources/rqsdk/manual-rqsdk.md)
- ## [快速上手](https://www.ricequant.com/doc/sources/rqsdk/manual-rqsdk.md#rqsdk-get-started) RQSDK 的快速上手指南。
- ## [RQSDK组件文档路径](https://www.ricequant.com/doc/sources/rqsdk/manual-rqsdk.md#rqsdk-doc-index) RQData、RQAlpha-Plus、RQFactor、RQOptimizer 四个组件的文档路径。
- ## [Anaconda安装](https://www.ricequant.com/doc/sources/rqsdk/manual-rqsdk.md#rqsdk-conda-isntall) Anaconda 的安装说明和环境管理（推荐使用）
- ## [VS Code 和PyCharm配置](https://www.ricequant.com/doc/sources/rqsdk/manual-rqsdk.md#rqsdk-doc-index-config) VS Code 和 PyCharm 的配置说明。
- ## [AI 编程工具配置指南](https://www.ricequant.com/doc/sources/rqsdk/manual-rqsdk.md#rqsdk-ai-tools) Claude Code、Cursor 和 VS Code Copilot 的配置说明。
- # [常见问题](https://www.ricequant.com/doc/sources/rqsdk/rqsdk-faq.md) RQSDK 常见问题解答。例如：如何解决安装过程中遇到的问题、如何解决使用过程中遇到的问题等。
- # [更新履历](https://www.ricequant.com/doc/sources/rqsdk/changelogs.md)
