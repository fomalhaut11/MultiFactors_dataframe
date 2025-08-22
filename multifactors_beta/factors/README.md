# Factors模块 - 因子研究框架

## 概述

Factors模块提供了完整的因子研究框架，包括因子生成、测试和分析三个核心功能。本模块经过重构，遵循模块接口设计规范，提供了清晰的API和灵活的扩展性。

## 模块结构

```
factors/
├── generator/          # 因子生成模块
│   ├── financial/      # 财务因子
│   ├── technical/      # 技术因子
│   └── risk/          # 风险因子
├── tester/            # 因子测试模块
├── analyzer/          # 因子分析模块
├── base/              # 基础类和工具
└── utils/             # 通用工具函数
```

## 快速开始

### 1. 生成因子

```python
from factors import generate

# 生成财务因子
roe = generate('ROE_ttm', financial_data)

# 生成技术因子
momentum = generate('Momentum', price_data)

# 生成风险因子
beta = generate('Beta', return_data)
```

### 2. 测试因子

```python
from factors import test

# 测试单个因子
result = test('ROE_ttm')
print(f"IC: {result.ic_result.ic_mean:.4f}")
print(f"ICIR: {result.ic_result.icir:.4f}")

# 批量测试
from factors.tester import batch_test
results = batch_test(['ROE_ttm', 'BP', 'SUE'])
```

### 3. 分析因子

```python
from factors import analyze

# 筛选优质因子
top_factors = analyze(preset='strict')

# 分析特定因子
analysis = analyze(['ROE_ttm', 'BP'])
```

### 4. 完整流水线

```python
from factors import pipeline

# 一键完成生成、测试、分析
results = pipeline('ROE_ttm', financial_data, test=True, analyze=True)

# 获取结果
factor_data = results['factor']
test_result = results['test']
analysis = results['analysis']
```

## 高级用法

### 使用生成器类

```python
from factors.generator import FinancialFactorGenerator

# 创建生成器
generator = FinancialFactorGenerator()

# 查看可用因子
available = generator.get_available_factors()

# 批量生成因子
factors = generator.batch_generate(
    ['ROE_ttm', 'ROA_ttm', 'CurrentRatio'],
    financial_data
)

# 保存因子
generator.save_factor('ROE_ttm')
```

### 自定义因子筛选

```python
from factors.analyzer import FactorScreener

# 创建筛选器
screener = FactorScreener()

# 自定义筛选条件
criteria = {
    'ic_mean_min': 0.03,
    'icir_min': 0.5,
    'sharpe_min': 1.0
}

# 筛选因子
selected = screener.screen_factors(criteria=criteria)
```

### 因子测试配置

```python
from factors.tester import SingleFactorTestPipeline

# 创建测试流水线
pipeline = SingleFactorTestPipeline()

# 自定义测试参数
result = pipeline.run(
    'ROE_ttm',
    begin_date='2020-01-01',
    end_date='2024-12-31',
    return_type='daily',
    price_type='o2o',
    save_result=True
)
```

## 因子类别

### 财务因子 (Financial)
- **盈利能力**: ROE_ttm, ROA_ttm, ROIC_ttm, 利润率等
- **偿债能力**: 流动比率、速动比率、负债率等
- **营运效率**: 周转率、周转天数等
- **成长能力**: 增长率、复合增长率等
- **现金流**: 现金流比率、自由现金流等
- **盈余惊喜**: SUE、盈余修正、盈余动量等

### 技术因子 (Technical)
- **价格因子**: 动量、反转、趋势等
- **波动率因子**: 历史波动率、GARCH等
- **技术指标**: MA、RSI、MACD等
- **成交量因子**: 换手率、量价关系等

### 风险因子 (Risk)
- **市场风险**: Beta、相关性等
- **波动率风险**: 残差波动率、特质波动率等
- **尾部风险**: VaR、CVaR、偏度、峰度等
- **系统性风险**: 行业暴露、因子暴露等

## API参考

### 主模块 (factors)

```python
# 便捷函数
generate(factor_name, data, **kwargs)  # 生成因子
test(factor_name, **kwargs)            # 测试因子
analyze(factor_names, **kwargs)        # 分析因子
pipeline(factor_name, data, **kwargs)  # 完整流水线
```

### 生成器模块 (factors.generator)

```python
# 类
FactorGenerator          # 生成器基类
FinancialFactorGenerator # 财务因子生成器
TechnicalFactorGenerator # 技术因子生成器
RiskFactorGenerator      # 风险因子生成器

# 函数
create_generator(factor_type)        # 创建生成器
generate_factor(factor_name, data)   # 生成单个因子
batch_generate_factors(names, data)  # 批量生成
list_available_factors(factor_type)  # 列出可用因子
```

### 测试模块 (factors.tester)

```python
# 类
SingleFactorTestPipeline  # 测试流水线
TestResult               # 测试结果
DataManager             # 数据管理器
FactorTester            # 因子测试器
ResultManager           # 结果管理器

# 函数
test_factor(factor_name)         # 测试单个因子
batch_test(factor_list)          # 批量测试
load_test_result(factor_name)    # 加载测试结果
```

### 分析模块 (factors.analyzer)

```python
# 类
FactorScreener    # 因子筛选器
AnalyzerConfig    # 分析配置

# 函数
screen_factors(criteria, preset)  # 筛选因子
get_analyzer_config()             # 获取配置
```

## 配置

### 因子路径配置

在`config.yaml`中配置：

```yaml
paths:
  factors: 'path/to/factors'
  single_factor_test: 'path/to/test_results'
  factor_analysis: 'path/to/analysis_results'
```

### 测试参数配置

```yaml
test:
  default_begin_date: '2020-01-01'
  default_end_date: '2024-12-31'
  return_type: 'daily'
  price_type: 'o2o'
```

### 筛选条件配置

```yaml
screening:
  presets:
    loose:
      ic_mean_min: 0.01
      icir_min: 0.3
    normal:
      ic_mean_min: 0.02
      icir_min: 0.5
    strict:
      ic_mean_min: 0.03
      icir_min: 0.7
```

## 扩展开发

### 添加新的因子类型

1. 在`generator`下创建新的子模块
2. 实现因子生成器类，继承`FactorGenerator`
3. 实现`generate()`和`get_available_factors()`方法
4. 在模块`__init__.py`中导出接口

### 添加新的测试指标

1. 在`tester/base/test_result.py`中添加新的指标类
2. 在`tester/core/factor_tester.py`中实现计算逻辑
3. 更新`TestResult`类以包含新指标

### 添加新的分析方法

1. 在`analyzer`下创建新的分析模块
2. 实现分析逻辑
3. 在`FactorScreener`中集成新方法

## 版本历史

- **v2.0.0** (2025-08-12)
  - 重构模块结构，分离生成器和测试器
  - 统一接口设计，遵循模块规范
  - 添加便捷函数和完整流水线
  - 改进文档和示例

- **v1.0.0** (初始版本)
  - 基础功能实现

## 注意事项

1. **数据格式**: 所有因子数据应为DataFrame格式，index为日期，columns为股票代码
2. **因子命名**: 使用下划线分隔，如`ROE_ttm`、`Momentum_20d`
3. **缺失值处理**: 生成器会自动处理缺失值，但建议在输入数据中预处理
4. **性能优化**: 批量操作比单个操作更高效

## 联系和支持

如有问题或建议，请联系开发团队或提交Issue。