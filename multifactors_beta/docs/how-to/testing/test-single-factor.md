# 单因子测试模块使用指南

## 概述

新的单因子测试模块提供了完整、高效、可扩展的因子测试框架，支持回归分析、分组测试、IC分析等多种测试方法。**v2.1 新增了中性化和归一化因子的独立保存和管理功能。**

## 架构设计

```
factors/tester/
├── base/                      # 基础数据结构
│   └── test_result.py        # 测试结果类
├── core/                      # 核心功能
│   ├── data_manager.py       # 数据管理
│   ├── factor_tester.py      # 测试执行
│   ├── result_manager.py     # 结果管理（新增因子保存/加载）
│   └── pipeline.py           # 测试流水线（新增导出/导入接口）
├── engines/                   # 测试引擎（待扩展）
├── analyzers/                 # 分析工具（待扩展）
└── utils/                     # 工具函数
```

## 主要特性

### 1. 完整的测试功能
- **回归分析**: 计算因子收益、t值、p值
- **分组测试**: 分组收益、单调性检验
- **IC分析**: IC、Rank IC、ICIR计算
- **性能指标**: 夏普比率、年化收益等

### 2. 灵活的配置管理
- 支持全局配置和运行时覆盖
- 每个测试结果保存完整配置快照
- 支持多配置并行测试

### 3. 高效的执行
- 数据缓存机制
- 并行批量测试
- 向量化计算优化

### 4. 完善的结果管理
- 结构化结果存储
- 多格式导出（pickle、yaml、json、excel）
- 结果比较和分析工具

### 5. **中性化因子管理（v2.1 新增）**
- **自动保存**: 测试时自动保存处理后的因子到专用目录
- **独立管理**: 处理后的因子与测试结果分离存储
- **版本组织**: 按配置参数自动组织因子存储结构
- **元数据记录**: 保存完整的处理配置和性能指标
- **便捷导入**: 提供简单的加载接口重用处理后的因子

## 快速开始

### 基础使用

```python
from factors.tester import SingleFactorTestPipeline

# 创建测试流水线
pipeline = SingleFactorTestPipeline()

# 测试单个因子（自动保存测试结果和处理后的因子）
result = pipeline.run('ROE_ttm')

# 查看结果
print(f"IC均值: {result.ic_result.ic_mean:.4f}")
print(f"ICIR: {result.ic_result.icir:.4f}")

# 处理后的因子已自动保存到：
# StockData/OrthogonalizationFactors/neutral_industry_outlier3_zscore/ROE_ttm.pkl
```

### 控制因子保存行为

```python
# 只保存测试结果，不保存处理后的因子
result = pipeline.run('ROE_ttm', save_processed_factor=False)

# 只生成因子，不保存测试结果
result = pipeline.run('ROE_ttm', save_result=False, save_processed_factor=True)

# 都不保存（仅用于临时测试）
result = pipeline.run('ROE_ttm', save_result=False, save_processed_factor=False)
```

### 自定义配置

```python
# 覆盖默认配置
result = pipeline.run(
    'EP_ttm',
    begin_date='2020-01-01',
    end_date='2024-12-31',
    group_nums=20,
    netral_base=False
)
```

### 批量测试

```python
# 测试多个因子
factors = ['ROE_ttm', 'ROA_ttm', 'EP_ttm', 'BP']
batch_result = pipeline.batch_run(
    factors,
    parallel=True,
    max_workers=4
)

# 查看汇总
print(batch_result.summary_df)
```

## 高级功能

### 1. 参数敏感性测试

```python
# 定义参数网格
param_grid = {
    'group_nums': [5, 10, 20],
    'outlier_param': [3, 5, 7]
}

# 执行测试
results_df = pipeline.parameter_sensitivity_test(
    'ROE_ttm',
    param_grid
)
```

### 2. 多配置测试

```python
# 使用不同配置测试同一因子
profiles = ['quick_test', 'full_test', 'weekly_test']
results = pipeline.run_with_profiles('ROE_ttm', profiles)
```

### 3. 结果分析

```python
from factors.tester import ResultManager

# 创建结果管理器
manager = ResultManager()

# 加载历史结果
results = manager.load_batch()

# 生成比较表
comparison = manager.compare_results(results)

# 导出到Excel
manager.export_to_excel(results, 'results.xlsx')
```

## 配置说明

### 全局配置（config.yaml）

```yaml
factor_test:
  # 测试时间范围
  begin_date: '2018-01-01'
  end_date: '2025-12-31'
  
  # 回测设置
  backtest_type: 'daily'
  group_nums: 10
  netral_base: true
  back_test_trading_price: 'o2o'
  
  # 基准因子
  base_factors:
    - 'LogMarketCap'
    - 'BP'
    - 'EP_ttm'
  
  # 行业分类
  classification_name: 'classification_one_hot'
  use_industry: true
```

### 运行时配置覆盖

所有配置参数都可以在运行时覆盖：

```python
result = pipeline.run(
    factor_name='ROE_ttm',
    # 覆盖配置
    begin_date='2020-01-01',
    group_nums=20,
    netral_base=False,
    custom_base_factors=['LogMarketCap']
)
```

## 测试结果结构

### TestResult 对象

```python
@dataclass
class TestResult:
    # 元数据
    test_id: str              # 唯一标识
    test_time: datetime       # 测试时间
    factor_name: str          # 因子名称
    
    # 配置快照
    config_snapshot: Dict     # 完整配置
    
    # 测试结果
    regression_result: RegressionResult
    group_result: GroupResult
    ic_result: ICResult
    
    # 性能指标
    performance_metrics: Dict
```

### 访问结果数据

```python
# 回归结果
factor_return = result.regression_result.factor_return
t_values = result.regression_result.tvalues

# 分组结果
group_returns = result.group_result.group_returns
long_short = result.group_result.long_short_return

# IC结果
ic_series = result.ic_result.ic_series
icir = result.ic_result.icir
```

## 性能优化建议

1. **使用数据缓存**: 重复测试时数据自动缓存
2. **并行批量测试**: 设置合适的 `max_workers`
3. **选择性保存**: 参数测试时可设置 `save_results=False`
4. **定期清理**: 使用 `manager.clean_old_results(days=30)`

## 与旧版本的区别

### 主要改进

1. **模块化设计**: 职责分离，易于扩展
2. **配置管理**: 统一配置，支持覆盖
3. **结果保存**: 包含配置快照，可追溯
4. **性能提升**: 缓存机制，并行执行
5. **错误处理**: 完善的异常捕获和日志

### 迁移指南

旧代码：
```python
# 旧版本
testengine = SingleFactorTestandSave(ini_file)
testengine.loading_factor('ROE_ttm')
testengine.backtest()
```

新代码：
```python
# 新版本
pipeline = SingleFactorTestPipeline()
result = pipeline.run('ROE_ttm')
```

## 常见问题

### Q: 如何添加新的测试方法？
A: 在 `engines/` 目录下创建新的测试引擎，继承基类并实现接口。

### Q: 如何自定义数据源？
A: 修改 `DataManager` 的数据加载方法，或通过配置指定数据路径。

### Q: 测试结果保存在哪里？
A: 默认保存在 `StockData/SingleFactorTestData/` 下，按日期分文件夹。

### Q: 如何处理大批量测试？
A: 使用 `batch_run` 方法，设置 `parallel=True` 和合适的 `max_workers`。

## 扩展开发

### 添加新的性能指标

```python
# 在 TestResult.calculate_performance_metrics 中添加
metrics['new_metric'] = calculate_new_metric(self)
```

### 自定义测试流程

```python
class CustomTester(FactorTester):
    def test(self, test_data):
        # 自定义测试逻辑
        pass
```

## 最佳实践

1. **定期清理缓存**: `pipeline.clear_cache()`
2. **批量优于单个**: 尽量使用批量测试
3. **保存配置模板**: 常用配置保存为profile
4. **监控测试质量**: 检查错误和警告信息
5. **版本控制结果**: 重要结果纳入版本管理

## 更新日志

### v2.0.0 (2024-12)
- 完全重构单因子测试模块
- 新增配置管理系统
- 支持并行批量测试
- 完善结果保存机制

## 联系支持

如有问题或建议，请：
1. 查看 `examples/single_factor_test_example.py` 示例
2. 阅读源代码注释
3. 提交Issue或联系开发团队