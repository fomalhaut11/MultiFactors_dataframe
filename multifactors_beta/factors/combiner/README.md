# 因子组合模块 (Factor Combiner)

## 概述

因子组合模块提供了多种因子组合方法，支持线性组合、正交化处理和动态权重优化，是构建多因子策略的核心组件。

## 主要功能

- **多种权重方法**: 等权、IC加权、IR加权、自定义权重
- **线性组合**: 支持因子的加权线性组合
- **正交化处理**: Gram-Schmidt正交化、残差正交化
- **滚动窗口组合**: 动态调整权重的滚动组合
- **灵活配置**: 支持多种标准化方法和缺失值处理

## 快速开始

### 基础使用

```python
from factors.combiner import FactorCombiner
import pandas as pd

# 准备因子数据（MultiIndex Series格式）
factors = {
    'momentum': momentum_factor,  # pd.Series with MultiIndex (date, stock)
    'value': value_factor,
    'quality': quality_factor
}

# 1. 等权重组合
combiner = FactorCombiner(method='equal_weight')
composite = combiner.combine(factors)

# 2. IC加权组合（需要评估结果）
combiner = FactorCombiner(method='ic_weight')
composite = combiner.combine(factors, evaluation_results=eval_results)

# 3. 自定义权重
custom_weights = {'momentum': 0.5, 'value': 0.3, 'quality': 0.2}
composite = combiner.combine(factors, custom_weights=custom_weights)
```

### 正交化处理

```python
# Gram-Schmidt正交化
orthogonal_factors = combiner.orthogonalize(
    factors,
    method='gram_schmidt',
    base_factor='value'  # value因子保持不变
)

# 残差正交化
orthogonal_factors = combiner.orthogonalize(
    factors,
    method='residual',
    base_factor='momentum'
)
```

### 滚动窗口组合

```python
# 使用60天窗口动态调整权重
rolling_composite = combiner.rolling_combine(
    factors,
    window=60,
    min_periods=30
)
```

## 配置选项

### 全局配置

```python
config = {
    # 权重约束
    'min_weight': 0.0,      # 最小权重
    'max_weight': 1.0,      # 最大权重
    'normalize': True,      # 是否归一化权重
    
    # 数据处理
    'handle_missing': 'forward_fill',  # 缺失值处理: 'forward_fill', 'drop', 'zero'
    'normalize_before': False,         # 组合前标准化
    'normalize_after': True,           # 组合后标准化
    'normalize_method': 'zscore',      # 标准化方法: 'zscore', 'minmax', 'rank', 'robust'
}

combiner = FactorCombiner(method='ic_weight', config=config)
```

### IC加权配置

```python
ic_config = {
    'use_abs_ic': True,     # 使用IC绝对值
    'ic_lookback': 12,      # IC回看期数
    'decay_factor': 0.9,    # 衰减因子
    'min_ic': 0.02         # 最小IC阈值
}
```

### IR加权配置

```python
ir_config = {
    'use_abs_ir': True,     # 使用IR绝对值
    'min_ir': 0.3,          # 最小IR阈值
    'ir_lookback': 12       # IR回看期数
}
```

## 权重计算方法

### 1. 等权重 (Equal Weight)
所有因子分配相同权重。

```python
combiner = FactorCombiner(method='equal_weight')
```

### 2. IC加权 (IC Weight)
根据因子的信息系数（IC）分配权重，IC越高权重越大。

```python
combiner = FactorCombiner(method='ic_weight')
# 需要提供evaluation_results或returns数据
composite = combiner.combine(factors, evaluation_results=eval_results)
```

### 3. IR加权 (IR Weight)
根据因子的信息比率（IR = IC/IC_std）分配权重，考虑稳定性。

```python
combiner = FactorCombiner(method='ir_weight')
composite = combiner.combine(factors, evaluation_results=eval_results)
```

## 与评估模块集成

```python
from factors.analyzer.evaluation import FactorEvaluator
from factors.tester import FactorTester

# 1. 测试因子
tester = FactorTester()
test_results = {}
for name, factor in factors.items():
    test_results[name] = tester.test(factor, price_data)

# 2. 评估因子
evaluator = FactorEvaluator()
eval_results = {}
for name, test_result in test_results.items():
    eval_results[name] = evaluator.evaluate(test_result)

# 3. 基于评估结果组合
combiner = FactorCombiner(method='ir_weight')
composite = combiner.combine(factors, evaluation_results=eval_results)
```

## 高级用法

### 分步处理

```python
# 1. 先正交化
orthogonal_factors = combiner.orthogonalize(factors, method='gram_schmidt')

# 2. 再组合
composite = combiner.combine(orthogonal_factors, evaluation_results=eval_results)
```

### 自定义评估函数

```python
def custom_evaluation(window_factors, **kwargs):
    """自定义权重计算函数"""
    # 计算自定义指标
    weights = {}
    for name, factor in window_factors.items():
        # 计算因子得分
        score = calculate_factor_score(factor)
        weights[name] = score
    
    # 归一化
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

# 使用自定义函数进行滚动组合
rolling_composite = combiner.rolling_combine(
    factors,
    window=60,
    evaluation_func=custom_evaluation
)
```

## API参考

### FactorCombiner

主要的因子组合器类。

```python
class FactorCombiner(CombinerBase):
    def __init__(self, method='equal_weight', config=None):
        """
        Parameters:
            method: 权重方法 ('equal_weight', 'ic_weight', 'ir_weight', 'custom')
            config: 配置字典
        """
    
    def combine(self, factors, evaluation_results=None, custom_weights=None, **kwargs):
        """组合因子"""
    
    def orthogonalize(self, factors, method='gram_schmidt', base_factor=None):
        """正交化因子"""
    
    def rolling_combine(self, factors, window=60, min_periods=30, evaluation_func=None):
        """滚动组合"""
```

## 性能优化

- 使用向量化操作提高计算效率
- 支持并行处理大规模因子
- 缓存中间结果减少重复计算

## 注意事项

1. **数据格式**: 所有因子必须是MultiIndex Series格式（日期、股票代码）
2. **对齐处理**: 自动对齐不同因子的索引
3. **缺失值**: 根据配置处理缺失值
4. **权重归一化**: 默认将权重归一化到和为1

## 示例代码

完整的示例代码请参见 `examples/combiner_example.py`

## 更新日志

### v1.0.0 (2024-01)
- 实现基础框架
- 支持三种权重方法
- 实现线性组合和正交化
- 添加滚动窗口功能