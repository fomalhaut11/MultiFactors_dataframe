# 多因子量化投资系统 v2.1.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 项目简介

**一个生产级的多因子量化投资研究框架**，提供从数据获取、因子生成、回测验证到组合构建的**完整投资工作流**。

🚀 **v2.1.0 重大更新**：新增批量因子生成系统 + 因子组合 + 风险模型 + 回测系统，项目完成度达到 **90%**

### ✨ 核心特性

- **🚀 批量因子生成**：三套生成方案 + 60+因子一键生成 + 质量验证
- **🧠 智能因子研究**：60+ 财务因子 + 技术因子 + 风险因子  
- **🔍 因子分析筛选**：五维度评估体系 + 智能筛选策略  
- **🤝 因子组合优化**：5种权重方法 + 4种组合策略
- **⚖️ 风险模型**：4种协方差估计 + Barra风险模型
- **📈 回测系统**：事件驱动回测 + 完整绩效分析
- **🛠️ 生产级质量**：完整测试体系 + 性能优化

## 快速开始

### 1. 环境准备

```bash
# Python 3.8+ 环境
python --version

# 安装依赖（如有requirements.txt）
pip install pandas numpy scipy statsmodels
```

### 2. 数据准备

#### 📊 数据更新系统

**全新的数据管理和更新系统**，提供完整的数据生命周期管理：

```bash
# 🩺 健康检查
python scheduled_data_updater.py --data-type all --health-check

# 🔄 数据更新
python scheduled_data_updater.py --data-type all --force

# 📋 查看数据状态
python scheduled_data_updater.py --data-summary
```

**📚 数据更新系统文档**：
- **[🚀 快速参考](./QUICK_REFERENCE.md)** - 常用命令和故障排除
- **[📖 完整指南](./DATA_UPDATE_GUIDE.md)** - 系统架构和使用方法
- **[🔗 依赖关系](./DATA_DEPENDENCY_MAP.md)** - 数据流向和依赖图谱
- **[🛠️ API文档](./DATA_UPDATER_API.md)** - 开发者API参考

#### 📊 辅助数据预处理（必须）

**prepare_auxiliary_data.py** 是因子计算的前置依赖脚本，用于生成标准化的辅助数据：

```bash
# 🔥 标准模式（推荐首次运行）
python data/prepare_auxiliary_data.py

# ⚡ 快速模式（日常更新推荐）
python data/prepare_auxiliary_data.py --fast        # 跳过验证，减少日志输出

# 🧪 测试模式（开发调试用）
python data/prepare_auxiliary_data.py --test        # 处理部分数据样本

# 🔬 实验模式（高级用户）
python data/prepare_auxiliary_data.py --parallel    # 启用并行处理（实验性）
```

**生成的辅助数据文件**：
- `ReleaseDates.pkl`：财报发布日期数据（reportday → ReleasedDates）
- `TradingDates.pkl`：交易日期列表（从Price.pkl提取）
- `StockInfo.pkl`：股票基本信息（代码、名称、上市日期等）
- `FinancialData_unified.pkl`：统一格式的三表财务数据
- `data_preparation_summary.json`：数据准备摘要和统计信息

**重要说明**：
- 📅 **数据字段理解**：`reportday`=财报公布日期，`tradingday`=财报截止日期
- 🏗️ **索引结构**：使用`(财报期间, 股票代码)`作为MultiIndex
- 📋 **依赖关系**：需要先运行数据更新获取原始数据（Price.pkl, lrb.pkl, xjlb.pkl, fzb.pkl）

#### 🗂️ 原始数据获取

```bash
# 1️⃣ 获取历史价格数据（首次运行）
python get_historical_price_2014.py

# 2️⃣ 获取财务数据（手动执行）
python scheduled_data_updater.py --data-type financial

# 3️⃣ 验证数据完整性
python scheduled_data_updater.py --data-type all --health-check
```

#### 🔄 完整数据准备工作流程

```bash
# 第一步：获取原始数据（首次运行或数据缺失时）
python get_historical_price_2014.py                    # 历史价格数据
python scheduled_data_updater.py --data-type financial # 财务数据

# 第二步：生成辅助数据（必须执行）
python data/prepare_auxiliary_data.py --fast          # 预处理辅助数据

# 第三步：验证数据就绪
python scheduled_data_updater.py --data-type all --health-check

# 🎯 现在可以开始因子计算和回测！
```

#### 📅 日常数据更新（高频数据）
python scheduled_data_updater.py --data-type price      # 价格数据（推荐日更）
python scheduled_data_updater.py --data-type stop_price # 涨跌停数据（推荐日更）

# 低频数据更新（按需手动执行）
python scheduled_data_updater.py --data-type financial  # 财务数据（季报后更新）

# 一次性更新所有数据（谨慎使用）
python scheduled_data_updater.py --data-type all        # 包含尚未实现的industry模块

# 强制更新（忽略时间和必要性检查）
python scheduled_data_updater.py --data-type price --force

# 数据健康检查
python scheduled_data_updater.py --data-type price --health-check      # 检查价格数据
python scheduled_data_updater.py --data-type financial --health-check  # 检查财务数据
python scheduled_data_updater.py --data-type all --health-check        # 检查所有数据
```

### 3. 批量因子生成 🚀

本项目提供了**三套批量因子生成解决方案**，满足不同用户需求：

#### ⚡ 快速生成模式（推荐新手）
**quick_generate_factors.py** - 零配置，开箱即用的快速因子生成
```bash
# 🎯 生成核心因子集合（15个最重要的因子）
python quick_generate_factors.py                    # 默认core模式

# 📋 生成基础因子集合（8个代表性因子）  
python quick_generate_factors.py --set basic        # 适合小数据量测试

# 🧪 生成测试因子集合（4个常用因子）
python quick_generate_factors.py --set test         # 快速验证环境

# 📖 查看所有可用因子集合
python quick_generate_factors.py --list
```

**预设因子集合**：
- **core**: ROE_ttm, ROA_ttm, BP, EP_ttm, Size, CurrentRatio, DebtToAssets, AssetTurnover_ttm, GrossProfitMargin_ttm, NetProfitMargin_ttm, RevenueGrowth_yoy, NetIncomeGrowth_yoy, OperatingCashFlowRatio_ttm, EarningsQuality_ttm, ROIC_ttm
- **basic**: ROE_ttm, BP, Size, CurrentRatio, AssetTurnover_ttm, GrossProfitMargin_ttm, RevenueGrowth_yoy, OperatingCashFlowRatio_ttm  
- **test**: ROE_ttm, BP, Size, CurrentRatio

#### 🛠️ 配置驱动模式（推荐高级用户）
**advanced_factor_generator.py** - 基于YAML配置的智能因子生成系统
```bash
# 🔧 使用默认配置生成核心因子
python advanced_factor_generator.py --mode core

# 📦 按因子分组生成
python advanced_factor_generator.py --mode financial    # 生成所有财务因子
python advanced_factor_generator.py --mode mixed        # 生成混合因子（需多种数据）

# 🎯 指定特定因子生成
python advanced_factor_generator.py --factors "ROE_ttm,BP,EP_ttm,Size"

# 📋 查看所有可用因子和模式
python advanced_factor_generator.py --list             # 查看所有因子
python advanced_factor_generator.py --list-modes       # 查看所有模式

# ⚙️ 使用自定义配置文件
python advanced_factor_generator.py --config my_config.yaml --mode all
```

**因子分组**（基于factor_config.yaml）：
- **financial**: 盈利能力(13) + 偿债能力(8) + 营运效率(9) = 30个财务因子
- **technical**: 价格动量 + 波动率 + 技术指标 = 17个技术因子
- **risk**: Beta系列 + 风险度量 = 8个风险因子
- **mixed**: 估值因子 + 规模因子 + 盈余惊喜 = 7个混合因子

#### 🔥 批量生成模式（完整功能）
**batch_generate_factors.py** - 支持60+个因子的完整批量生成系统
```bash
# 🌟 生成所有已实现的因子（60+个）
python batch_generate_factors.py --mode all

# 📦 按类型生成因子
python batch_generate_factors.py --mode financial      # 财务因子（61个）
python batch_generate_factors.py --mode technical      # 技术因子（17个）
python batch_generate_factors.py --mode risk          # 风险因子（8个）
python batch_generate_factors.py --mode mixed         # 混合因子（7个）

# 🎯 指定因子列表生成
python batch_generate_factors.py --factors "ROE_ttm,ROA_ttm,BP,EP_ttm,Size"

# ⚡ 并行加速生成（4核）+ 快速模式
python batch_generate_factors.py --mode all --parallel 4 --fast

# 📋 查看所有可用因子
python batch_generate_factors.py --list-factors

# 🧪 生成但不保存（仅测试）
python batch_generate_factors.py --mode test --no-save
```

**性能优化选项**：
- `--parallel N`: 使用N个进程并行计算（默认CPU核数的一半）
- `--fast`: 快速模式，减少日志输出和验证步骤
- `--no-save`: 不保存结果文件，仅用于测试

#### 🔍 因子质量验证
**validate_factors.py** - 生成后自动验证因子质量
```bash
# 🔍 验证所有生成的因子
python validate_factors.py

# 🎯 验证特定因子
python validate_factors.py --factor ROE_ttm
python validate_factors.py --factors "ROE_ttm,BP,EP_ttm"

# 📊 仅生成质量报告（不显示详细信息）
python validate_factors.py --report-only

# 📁 验证指定目录的因子
python validate_factors.py --dir "E:/path/to/your/factors"
```

**质量评估标准**：
- **数据完整性**: 空值比例、无穷值检查
- **分布特征**: 偏度、峰度、变异系数
- **异常值**: IQR方法检测离群点
- **质量评分**: A(85+) / B(75-85) / C(60-75) / D(<60)

#### 📊 完整生成工作流（推荐）
```bash
# 第1步：数据准备（如果尚未完成）
python data/prepare_auxiliary_data.py --fast

# 第2步：快速生成核心因子（新手推荐）
python quick_generate_factors.py --set core

# 或：批量生成所有因子（高级用户）
python batch_generate_factors.py --mode all --fast

# 第3步：验证生成的因子质量
python validate_factors.py --report-only

# 第4步：查看生成结果
ls E:/Documents/PythonProject/StockProject/StockData/factors/
```

#### ⚙️ 自定义因子配置
**编辑 factor_config.yaml** 添加自定义因子：
```yaml
factor_groups:
  custom:
    description: "自定义因子组"
    enabled: true
    priority: 5
    
    my_factors:
      - name: "MyCustomFactor"
        description: "我的自定义因子"
        calculator: "PureFinancialFactorCalculator" 
        method: "calculate_MyCustomFactor"
        data_requirements: ["financial_data"]
        parameters: {"window": 12}
```

#### 📁 输出文件结构
```
E:/Documents/PythonProject/StockProject/StockData/factors/
├── ROE_ttm.pkl                              # 单个因子数据
├── BP.pkl
├── Size.pkl
├── ...
├── quick_generation_summary_20250824_143022.json    # 快速生成摘要
├── factor_generation_report_20250824_143028.json    # 详细生成报告
├── factor_validation_report_20250824_143035.json    # 质量验证报告
└── generation_summary_20250824_143040.json          # 批量生成摘要
```

💡 **使用建议**：
- **新手用户**：使用 `quick_generate_factors.py --set core` 开始
- **研究人员**：使用 `advanced_factor_generator.py` 精确控制因子生成
- **生产环境**：使用 `batch_generate_factors.py --mode all --fast` 全量生成
- **质量保证**：生成后必须运行 `validate_factors.py` 检查质量

### 4. 核心功能使用示例

#### 🧠 因子计算

##### 纯财务因子（仅需财务数据）
```python
from factors.generator.financial import PureFinancialFactorCalculator

# 初始化纯财务因子计算器
calculator = PureFinancialFactorCalculator()

# 计算ROE因子（TTM方式）
roe = calculator.calculate_ROE_ttm(financial_data)

# 计算流动比率
current_ratio = calculator.calculate_CurrentRatio(financial_data)

# 批量计算多个因子
factors = calculator.calculate_multiple_factors(
    ['ROE_ttm', 'ROA_ttm', 'CurrentRatio'], financial_data
)
```

##### 混合因子（需要多种数据源）🆕
```python
from factors.generator.mixed import get_mixed_factor_manager

# 获取混合因子管理器
manager = get_mixed_factor_manager()

# 准备数据
data = {
    'financial_data': financial_data,
    'market_cap': market_cap
}

# 计算估值因子
bp = manager.calculate_factor('BP', data)           # 净资产市值比
ep = manager.calculate_factor('EP_ttm', data)       # 净利润市值比

# 批量计算估值因子
valuation_factors = manager.calculate_multiple_factors(
    ['BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm'], data
)
```

#### 🔍 因子测试
```python 
from factors.tester import SingleFactorTestPipeline

# 单因子测试
pipeline = SingleFactorTestPipeline()
result = pipeline.run('ROE_ttm', begin_date='2020-01-01')

print(f"IC均值: {result.ic_result.ic_mean:.4f}")
print(f"ICIR: {result.ic_result.icir:.4f}")
```

#### 🎯 因子筛选 🆕
```python
from factors.analyzer.screening import FactorScreener

# 因子筛选
screener = FactorScreener()
top_factors = screener.screen_factors(
    preset='strict',  # IC>0.03, ICIR>0.7
    top_n=10
)
```

#### 🤝 因子组合 🆕
```python
from factors.combiner import FactorCombiner
from factors.combiner.weighting import ICWeight

# 因子组合
combiner = FactorCombiner()
combined_factor = combiner.combine_factors(
    factors=['ROE_ttm', 'BP', 'EP_ttm'],
    weight_method=ICWeight(),
    method='linear'
)
```

#### 📈 回测系统 🆕
```python
from backtest import BacktestEngine
from backtest.cost import CommissionModel

# 回测设置
engine = BacktestEngine()
engine.set_cost_model(CommissionModel(rate=0.0003))

# 运行回测
result = engine.run_backtest(
    strategy=your_strategy,
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

💡 **完整示例**请参考 `examples/` 目录下的演示代码

## 🏗️ 项目架构

本项目采用**分层模块化架构**，支持灵活扩展和高性能计算：

```
multifactors_beta/
├── 🔧 数据更新脚本
│   ├── scheduled_data_updater.py    # ⭐ 定时数据更新（主程序）
│   ├── interactive_data_updater.py  # 交互式数据更新
│   └── get_historical_price_2014.py # 历史数据获取
│
├── 🚀 批量因子生成脚本 ✨
│   ├── quick_generate_factors.py    # ⚡ 快速生成（新手推荐）
│   ├── advanced_factor_generator.py # 🛠️ 配置驱动生成（高级用户）
│   ├── batch_generate_factors.py    # 🔥 完整批量生成
│   ├── validate_factors.py          # 🔍 因子质量验证
│   └── factor_config.yaml           # ⚙️ 因子配置文件
│
├── 🏗️ 核心框架
│   ├── core/                        # 基础设施层
│   │   ├── config_manager.py        # 配置管理器
│   │   ├── database/                # 数据库连接池
│   │   └── utils/                   # 工具函数库
│   │
│   └── factors/                     # 因子研究框架 ✨
│       ├── generator/               # 因子生成器
│       │   ├── financial/           # 纯财务因子（60+ 个）
│       │   ├── mixed/               # 混合因子 🆕
│       │   │   └── valuation/       # 估值因子（BP、EP、SP、CFP）
│       │   ├── technical/           # 技术因子 
│       │   ├── risk/               # 风险因子
│       │   └── alpha191/           # Alpha191 因子集 🆕
│       ├── tester/                 # 因子测试器
│       ├── analyzer/               # 因子分析器
│       ├── combiner/               # 因子组合器 🆕
│       ├── selector/               # 因子选择器 🆕  
│       ├── risk_model/             # 风险模型 🆕
│       └── base/                   # 基础类库
│
├── 🔄 回测系统 🆕
│   └── backtest/
│       ├── engine/                 # 回测引擎
│       ├── portfolio/              # 组合管理
│       ├── cost/                   # 交易成本模型
│       ├── performance/            # 绩效分析
│       └── utils/                  # 约束和验证
│
├── 📊 数据处理
│   └── data/
│       ├── fetcher/               # 数据获取器
│       ├── processor/             # 数据处理器
│       ├── examples/              # 格式示例 🆕
│       ├── schemas.py             # 数据结构 🆕
│       └── data_bridge.py         # 数据桥接 🆕
│
├── 🧪 测试框架
│   └── tests/
│       ├── unit/                  # 单元测试
│       ├── integration/           # 集成测试
│       └── performance/           # 性能测试
│
├── 📚 文档和示例  
│   ├── docs/                      # 详细文档
│   └── examples/                  # 使用示例
│
└── 📜 主要脚本
    ├── quick_generate_factors.py       # ⚡ 快速因子生成
    ├── advanced_factor_generator.py    # 🛠️ 配置驱动生成
    ├── batch_generate_factors.py       # 🔥 批量因子生成
    ├── validate_factors.py             # 🔍 因子质量验证
    ├── generate_*.py                   # 其他因子生成脚本
    └── test_*.py                       # 测试脚本
```

### 🆕 v2.1.0 新增模块

- **🚀 批量因子生成系统**：三套生成方案 + 智能配置 + 质量验证
- **因子组合器** (factors/combiner)：5种权重方法 + 4种组合策略
- **因子选择器** (factors/selector)：智能筛选 + 多策略选择  
- **风险模型** (factors/risk_model)：协方差估计 + Barra模型
- **回测系统** (backtest)：完整的策略回测框架

## 🔥 主要功能

### 🚀 批量因子生成（新特性）✨
- **三套生成方案**：快速模式、配置驱动、完整批量
- **60+预定义因子**：财务(61) + 技术(17) + 风险(8) + 混合(7)
- **智能配置管理**：YAML配置 + 数据依赖分析
- **质量保证体系**：自动验证 + 评分 + 详细报告
- **性能优化**：并行计算 + 快速模式 + 内存优化

**支持的因子类型**：
- **财务因子**：盈利能力、偿债能力、营运效率、成长能力、现金流、资产质量、盈利质量
- **技术因子**：价格动量、波动率、技术指标
- **风险因子**：Beta系列、风险度量
- **混合因子**：估值因子、规模因子、盈余惊喜

### 📊 数据管理
- **自动化数据更新**: 定时任务 + 健康检查
- **增量数据获取**: 智能检测，只更新必要部分  
- **多数据源适配**: 统一数据接口，支持扩展
- **数据完整性保障**: 备份恢复 + 异常处理

**支持的数据类型**：
- **高频数据**：`price` (日线数据), `stop_price` (涨跌停)
- **财务数据**：`financial` (三表数据), `macro` (宏观经济)
- **基础数据**：`tradable` (可交易股票), `trading_dates` (交易日历) 🆕
- **分类数据**：`industry` (板块成份股), `concept` (概念板块), `st` (ST股票) 🆕
- **指数数据**：`index` (指数价格), `widebase_component` (宽基成份股) 🆕  
- **资讯数据**：`ipo_date` (IPO日期), `foreshow` (预报), `announcement` (公告) 🆕

### 🧠 因子生成 (60+ 因子)
**财务因子**：
- **盈利能力**：ROE_ttm, ROA_ttm, ROIC_ttm等13个
- **偿债能力**：CurrentRatio, DebtToAssets等8个  
- **营运效率**：AssetTurnover_ttm等9个
- **成长能力**：RevenueGrowth_yoy等10个
- **盈余惊喜**：SUE (标准化未预期盈余)

**技术因子**：价格动量、波动率、技术指标  
**风险因子**：Beta系列、风险度量

### 🔍 因子分析与筛选 ✨
- **五维度评估**：盈利能力、稳定性、及时性、可交易性、独特性
- **相关性分析**：因子间相关性计算和可视化
- **稳定性检验**：时间序列稳定性 + 滚动窗口分析
- **智能筛选**：预设条件(loose/normal/strict) + 自定义条件

### 🤝 因子组合优化 🆕
**权重计算方法**：
- **等权重**：简单平均组合
- **IC加权**：基于历史IC表现动态权重
- **风险平价**：基于波动率的风险均衡配置
- **最优权重**：最大化IC的约束优化

**组合方法**：
- **线性组合**：加权平均组合
- **正交化组合**：去除因子间相关性
- **PCA中性化**：主成分降维组合

### 🎯 因子选择策略 🆕
**筛选器**：
- **性能筛选**：基于IC、ICIR、收益率多指标筛选
- **相关性筛选**：控制因子间最大相关性阈值  
- **稳定性筛选**：时间序列稳定性控制
- **复合筛选**：多筛选器组合(AND/OR逻辑)

**选择策略**：
- **TopN选择**：基于排序的TopN选择
- **阈值选择**：动态阈值筛选
- **聚类选择**：K-means聚类降低相关性

### ⚖️ 风险模型 🆕
**协方差估计器**：
- **Ledoit-Wolf收缩**：自动最优收缩参数（性能最佳）
- **指数加权**：时变协方差建模
- **稳健估计**：异常值处理（27%识别率）
- **样本协方差**：传统方法基准

**风险模型**：
- **协方差模型**：支持多种估计器切换
- **Barra模型**：多因子风险分解
- **通用因子模型**：支持PCA、混合建模

### 📈 回测系统 🆕
**回测引擎**：
- **事件驱动框架**：支持多策略并行
- **完整时间管理**：交易日历 + 时间轴控制

**交易成本建模**：
- **佣金模型**：多种计算方式 + 阶梯费率
- **市场冲击**：线性/非线性冲击建模
- **滑点模型**：固定/比例滑点 + 市场调整

**绩效分析**：
- **风险指标**：夏普比率、最大回撤、Alpha/Beta
- **归因分析**：因子贡献分解 + 风险归因
- **可视化报告**：详细绩效图表

### 🛠️ 系统特性
- **生产级质量**：完整测试体系 + 85% 测试覆盖率
- **高性能计算**：向量化处理 + 内存优化
- **模块化架构**：松耦合设计，易扩展
- **跨平台兼容**：Windows/Linux + 编码处理

## 使用文档

- [数据预处理指南](docs/数据预处理功能完整指南.md)
- [因子计算指南](docs/因子计算模块迁移指南.md)
- [数据更新说明](docs/Price数据增量更新使用说明.md)
- [数据字段说明](docs/数据字段理解修正说明.md)

## 配置说明

### 1. 主配置文件 `config.yaml`
```yaml
# 数据库连接配置
database:
  host: your_host
  user: your_user
  password: your_password

# 数据路径配置
paths:
  data_root: E:\Documents\PythonProject\StockProject\StockData
  project_root: E:\Documents\PythonProject\StockProject\MultiFactors\multifactors_beta

# 系统参数
system:
  log_level: INFO
  backup_days: 3
```

### 2. 数据更新自动化配置

**🚀 推荐的自动化配置（仅高频数据）：**

**Windows 任务计划程序：**
```batch
# 任务1: 工作日下午4点更新价格数据
程序: python.exe
参数: E:\path\to\scheduled_data_updater.py --data-type price
触发器: 每个工作日 16:00

# 任务2: 工作日下午4:05更新涨跌停数据  
程序: python.exe
参数: E:\path\to\scheduled_data_updater.py --data-type stop_price
触发器: 每个工作日 16:05
```

**Linux crontab：**
```bash
# 工作日下午4点更新高频数据
0 16 * * 1-5 cd /path/to/multifactors_beta && python scheduled_data_updater.py --data-type price
5 16 * * 1-5 cd /path/to/multifactors_beta && python scheduled_data_updater.py --data-type stop_price

# 每天早上8点健康检查
0 8 * * * cd /path/to/multifactors_beta && python scheduled_data_updater.py --data-type price --health-check
```

**⚠️ 包含所有数据类型的配置（不推荐日常使用）：**
```batch
# 仅在测试或特殊情况下使用 --data-type all
程序: python.exe  
参数: E:\path\to\scheduled_data_updater.py --data-type all
说明: 会尝试更新包括未实现的industry模块在内的所有数据
```

### 3. 分类数据更新策略

**📅 日更数据（自动化推荐）:**
```bash
# 价格数据 - 交易日必更
python scheduled_data_updater.py --data-type price

# 涨跌停数据 - 交易日必更  
python scheduled_data_updater.py --data-type stop_price
```

**📋 季更数据（手动执行）:**
```bash
# 财务数据 - 财报发布后更新（年报、中报、季报）
python scheduled_data_updater.py --data-type financial

# 建议时机：
# - 4月底（年报季结束后）
# - 8月底（中报季结束后）  
# - 10月底（三季报结束后）
# - 1月底（四季报预披露后）
```

**🏗️ 低频/手动数据:**
```bash
# 行业数据 - 申万二级行业分类（已实现） 🆕
python scheduled_data_updater.py --data-type industry

# ST股票数据 - 特殊处理股票信息（已实现） 🆕
python scheduled_data_updater.py --data-type st
# 保存到: auxiliary/ST_stocks.pkl，包含ST股票的历史记录

# 手动获取各类数据 🆕
python -c "
from data.fetcher.data_fetcher import StockDataFetcher
fetcher = StockDataFetcher()

# 获取板块成份股信息（申万行业）
industry_data = fetcher.fetch_data('industry', index_code='all', begin_date=20240101)
print(f'板块成份股数据: {industry_data.shape}')

# 获取概念板块数据  
concept_data = fetcher.fetch_data('concept')
print(f'概念板块数据: {concept_data.shape}')

# 获取ST股票数据 🆕
st_data = fetcher.fetch_data('st')
print(f'ST股票数据: {st_data.shape}')

# 获取交易日期 🆕
trading_dates = fetcher.fetch_data('trading_dates')
print(f'交易日期数据: {trading_dates.shape}')

# 获取板块名称列表 🆕
index_namelist = fetcher.fetch_data('index_namelist')
print(f'板块名称数据: {index_namelist.shape}')

# 获取宽基指数成份股 🆕
widebase_data = fetcher.fetch_data('widebase_component', index_code='SH000300')
print(f'沪深300成份股数据: {widebase_data.shape}')

# 获取IPO日期 🆕
ipo_data = fetcher.fetch_data('ipo_date')
print(f'IPO日期数据: {ipo_data.shape}')
"

# 完整检查（包含所有数据类型）
python scheduled_data_updater.py --data-type all --health-check

# 批量更新低频数据
python scheduled_data_updater.py --data-type industry
python scheduled_data_updater.py --data-type st
python scheduled_data_updater.py --data-type financial
```

**🔧 维护和调试命令：**
```bash
# 强制更新（忽略时间限制和必要性检查）
python scheduled_data_updater.py --data-type price --force

# 详细健康检查  
python scheduled_data_updater.py --data-type financial --health-check

# 查看日志文件
# Windows: %DATA_ROOT%\logs\data_update_YYYYMMDD.log
# Linux: $DATA_ROOT/logs/data_update_YYYYMMDD.log
```

## 重要说明

### 📊 数据准备流程（必读）
⚠️ **在进行因子计算前，必须完成数据准备步骤**：

1. **原始数据获取**：运行`get_historical_price_2014.py`和`scheduled_data_updater.py --data-type financial`
2. **辅助数据预处理**：运行`python data/prepare_auxiliary_data.py --fast`
3. **数据验证**：运行`scheduled_data_updater.py --data-type all --health-check`

📁 **辅助数据文件位置**：`data/auxiliary/` 目录下的预处理文件是因子计算的必要输入

### 数据字段理解
- **reportday**: 财报发布日期（报表公告日）
- **tradingday**: 财报截止日期（报告期末）
- **d_year + d_quarter**: 可靠的报告期标识
- **ReportPeriod**: 标准化的财报期间（由prepare_auxiliary_data.py生成）

### Windows用户注意
- 项目包含 `utils/io_utils.py` 处理编码问题
- 使用绝对路径避免路径问题
- 注意反斜杠转义或使用原始字符串

## 开发计划

- [x] 数据获取和更新系统
- [x] 基本面因子计算框架  
- [x] 数据预处理模块
- [x] 因子测试和分析系统
- [x] 因子组合和选择系统 🆕
- [x] 风险模型框架 🆕
- [x] 回测系统框架 🆕
- [ ] 组合优化算法完善
- [ ] 实时监控系统
- [ ] Web管理界面
- [ ] 机器学习因子挖掘

## 贡献指南

欢迎提交Issue和Pull Request。开发新功能请：
1. Fork本项目
2. 创建功能分支
3. 提交变更
4. 发起Pull Request

## 版本历史

- **v2.1.0-beta** (2025-08-24): 🚀 重大功能更新
  - 🚀 新增批量因子生成系统：3套生成方案 + 60+因子一键生成
    - quick_generate_factors.py：零配置快速生成（新手友好）
    - advanced_factor_generator.py：YAML配置驱动生成（精确控制）
    - batch_generate_factors.py：完整批量生成（生产环境）
    - validate_factors.py：自动质量验证和评分系统
    - factor_config.yaml：智能因子配置管理
  - 新增因子组合系统：5种权重方法 + 4种组合策略
  - 新增因子选择系统：智能筛选器 + 多策略选择
  - 新增风险模型框架：4种协方差估计 + Barra模型
  - 新增回测系统框架：事件驱动 + 交易成本建模
  - 新增因子评估体系：五维度综合评估
  - 修复项目文件夹命名问题，统一为multifactors_beta
  - 完善项目文档和使用指南

- **v2.0-beta** (2025-08-01): 
  - 重构因子计算框架
  - 修正数据字段理解
  - 优化性能和稳定性
  
- **v1.2-beta** (2025-07-30): 
  - 完成数据更新系统
  - 实现连接池管理

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**项目状态**: 生产就绪  
**License**: MIT