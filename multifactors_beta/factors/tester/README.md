# 单因子测试模块 (Single Factor Test Module)

## 模块概述

单因子测试模块提供了完整的因子测试框架，用于评估单个因子的预测能力和投资价值。该模块支持回归分析、分组测试、IC分析等多种测试方法，并能生成详细的测试报告。

## 核心功能

- **回归分析**: 截面回归提取因子收益
- **分组测试**: 因子分组后计算各组收益表现
- **IC分析**: 计算IC、Rank IC、ICIR等指标
- **性能评估**: 夏普比率、最大回撤、胜率等
- **结果管理**: 自动保存和加载测试结果

## 目录结构

```
factors/tester/
├── __init__.py           # 模块导出接口
├── README.md             # 本文档
├── core/                 # 核心功能模块
│   ├── pipeline.py       # 测试流水线
│   ├── data_manager.py   # 数据管理
│   ├── factor_tester.py  # 因子测试核心
│   └── result_manager.py # 结果管理
└── base/                 # 基础类定义
    ├── test_result.py    # 测试结果类
    └── test_config.py    # 测试配置类
```

## 快速开始

### 基础用法

```python
from factors.tester import SingleFactorTestPipeline

# 创建测试流水线
pipeline = SingleFactorTestPipeline()

# 运行因子测试
result = pipeline.run(
    factor_name='BP',                  # 因子名称
    save_result=True,                  # 是否保存结果
    begin_date='2024-01-01',          # 开始日期
    end_date='2024-12-31',            # 结束日期
    group_nums=5,                      # 分组数量
    netral_base=False,                 # 是否中性化
    use_industry=False                 # 是否使用行业
)

# 查看测试结果
if result.ic_result:
    print(f"IC均值: {result.ic_result.ic_mean:.4f}")
    print(f"ICIR: {result.ic_result.icir:.4f}")
    print(f"Rank IC: {result.ic_result.rank_ic_mean:.4f}")

if result.group_result:
    print(f"单调性: {result.group_result.monotonicity_score:.4f}")
    print(f"多空夏普: {result.performance_metrics.get('long_short_sharpe', 0):.4f}")
```

### 批量测试多个因子

```python
from factors.tester import SingleFactorTestPipeline

pipeline = SingleFactorTestPipeline()
factor_list = ['BP', 'EP_ttm', 'ROE_ttm', 'SP_ttm']

results = {}
for factor_name in factor_list:
    print(f"测试因子: {factor_name}")
    result = pipeline.run(
        factor_name,
        save_result=True,
        begin_date='2024-01-01',
        end_date='2024-12-31'
    )
    results[factor_name] = result
    
# 汇总结果
for factor_name, result in results.items():
    if result and result.ic_result:
        print(f"{factor_name}: IC={result.ic_result.ic_mean:.4f}, ICIR={result.ic_result.icir:.4f}")
```

### 自定义测试配置

```python
from factors.tester import SingleFactorTestPipeline

# 自定义配置
custom_config = {
    'group_nums': 10,                    # 使用10分组
    'netral_base': True,                 # 启用市值中性化
    'use_industry': True,                 # 使用行业信息
    'back_test_trading_price': 'vwap',   # 使用VWAP价格
    'return_type': 'daily',              # 日频收益
    'price_type': 'o2o'                  # Open to Open
}

pipeline = SingleFactorTestPipeline()
result = pipeline.run(
    'BP',
    save_result=True,
    begin_date='2024-01-01',
    end_date='2024-12-31',
    **custom_config
)
```

### 加载历史测试结果

```python
from factors.tester.core.result_manager import ResultManager

# 创建结果管理器
manager = ResultManager()

# 加载特定因子的最新测试结果
result = manager.load_latest('BP')
if result:
    print(f"测试时间: {result.test_time}")
    print(f"IC均值: {result.ic_result.ic_mean:.4f}")

# 加载特定日期的所有测试结果
results = manager.load_by_date('2025-08-11')
for result in results:
    print(f"因子: {result.factor_name}, IC: {result.ic_result.ic_mean:.4f}")
```

## 测试结果说明

### IC分析结果
- **ic_mean**: IC均值，衡量因子预测能力
- **ic_std**: IC标准差，衡量预测稳定性
- **icir**: IC信息比率 (IC_mean/IC_std)
- **rank_ic_mean**: Rank IC均值，更稳健的相关性指标
- **ic_decay**: IC衰减序列，观察预测能力持续性

### 分组测试结果
- **group_returns**: 各组日收益率序列
- **group_cumulative_returns**: 各组累计收益曲线
- **long_short_return**: 多空组合收益序列
- **monotonicity_score**: 单调性得分，衡量分组效果

### 回归分析结果
- **factor_return**: 因子日收益序列
- **cumulative_return**: 因子累计收益曲线
- **tvalues**: 回归系数t值
- **rsquared_adj**: 调整后R方

### 性能指标
- **annual_return**: 年化收益率
- **annual_volatility**: 年化波动率
- **long_short_sharpe**: 多空组合夏普比率
- **max_drawdown**: 最大回撤
- **win_rate**: 胜率

## 数据要求

### 必需数据
1. **因子数据**: 存放在 `RawFactors/` 目录，pickle格式
2. **收益率数据**: `LogReturn_daily_o2o.pkl` 等
3. **交易日期**: `TradingDates.pkl`

### 可选数据
1. **市值数据**: `LogMarketCap.pkl` (用于中性化)
2. **行业数据**: `IndustryClassification.pkl`
3. **财务数据**: `FinancialData_unified.pkl`

## 配置说明

### 默认配置路径
- 全局配置: `config.yaml`
- 测试结果: `SingleFactorTestData/`
- 因子数据: `RawFactors/`

### 主要配置参数
```yaml
paths:
  data_root: 'E:/Documents/PythonProject/StockProject/StockData'
  raw_factors: '.../RawFactors'
  single_factor_test: '.../SingleFactorTestData'

factor_test:
  default_group_nums: 5
  default_return_type: 'daily'
  default_price_type: 'o2o'
```

## 高级功能

### 自定义因子测试器

```python
from factors.tester.core.factor_tester import FactorTester

class CustomFactorTester(FactorTester):
    def custom_analysis(self, factor_data, return_data):
        # 实现自定义分析逻辑
        pass

# 使用自定义测试器
tester = CustomFactorTester()
result = tester.test(factor_data, return_data)
```

### 导出测试报告

```python
# 导出分组收益曲线
group_returns = result.group_result.group_cumulative_returns
group_returns.to_csv('group_returns.csv')

# 导出IC序列
ic_series = result.ic_result.ic_series
ic_series.to_csv('ic_series.csv')

# 生成测试摘要
summary = result.to_summary_dict()
import json
with open('test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
```

## 常见问题

### Q1: 如何处理大规模因子测试？
```python
# 使用批处理和并行计算
import multiprocessing as mp

def test_factor(factor_name):
    pipeline = SingleFactorTestPipeline()
    return pipeline.run(factor_name, save_result=True)

# 并行测试多个因子
with mp.Pool(processes=4) as pool:
    factor_list = ['BP', 'EP', 'ROE', 'SP']
    results = pool.map(test_factor, factor_list)
```

### Q2: 如何自定义收益率计算？
```python
# 在运行测试时指定收益率类型
result = pipeline.run(
    'BP',
    return_type='weekly',  # 周频收益
    price_type='vwap'      # 使用VWAP价格
)
```

### Q3: 测试结果保存在哪里？
- 默认保存路径: `SingleFactorTestData/YYYYMMDD/`
- 文件命名: `{factor_name}_{datetime}_{test_id}.pkl`

## 性能优化建议

1. **缓存数据**: DataManager会自动缓存常用数据
2. **批量处理**: 一次性测试多个因子，复用基础数据
3. **时间范围**: 合理设置测试期间，避免过长时间跨度
4. **分组数量**: 通常5-10组即可，过多分组可能导致噪音

## 版本更新

### v1.0.0 (2025-08)
- 初始版本发布
- 实现核心测试功能
- 支持回归、分组、IC分析
- 自动结果保存和管理

## 相关模块

- [因子构建模块](../builder/README.md)
- [因子筛选模块](../analyzer/README.md)
- [回测模块](../../backtest/README.md)

## 联系与支持

如有问题或建议，请联系开发团队或提交Issue。

---
*最后更新: 2025-08-11*