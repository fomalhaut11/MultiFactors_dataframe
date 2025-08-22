# 因子选择模块 (Selector)

因子选择模块提供了完整的因子筛选和选择功能，包括多种筛选器和选择策略。

## 模块结构

```
factors/selector/
├── __init__.py                    # 模块导出
├── README.md                      # 本文档
├── DEVELOPMENT_PLAN.md            # 开发计划
│
├── base/                          # 基础类
│   ├── __init__.py
│   └── selector_base.py          # 选择器基类
│
├── filters/                       # 筛选器
│   ├── __init__.py
│   ├── base_filter.py            # 筛选器基类
│   ├── performance_filter.py     # 性能筛选
│   ├── correlation_filter.py     # 相关性筛选
│   ├── stability_filter.py       # 稳定性筛选
│   └── composite_filter.py       # 复合筛选
│
├── strategies/                    # 选择策略
│   ├── __init__.py
│   ├── top_n.py                  # TopN选择
│   ├── threshold.py              # 阈值选择
│   └── clustering.py             # 聚类选择
│
├── factor_selector.py             # 主选择器
└── factor_pool.py                 # 因子池管理
```

## 主要功能

### 1. 因子池管理 (FactorPool)

- 因子数据存储和管理
- 元数据管理和查询
- 因子分类和搜索
- 覆盖率统计

```python
from factors.selector import FactorPool

# 创建因子池
pool = FactorPool()

# 添加因子
pool.add_factor(
    name='momentum_factor',
    factor_data=factor_series,
    category='momentum',
    description='价格动量因子'
)

# 搜索因子
factors = pool.search_factors(
    keyword='momentum',
    min_coverage=0.8
)
```

### 2. 筛选器 (Filters)

#### 性能筛选器 (PerformanceFilter)
基于因子性能指标筛选：
- 总分阈值
- IC均值阈值
- IR阈值
- 夏普比率阈值

```python
from factors.selector import PerformanceFilter

perf_filter = PerformanceFilter(
    min_score=70.0,
    min_ic=0.02,
    min_ir=0.3
)

filtered_factors = perf_filter.filter(factors, evaluation_results)
```

#### 相关性筛选器 (CorrelationFilter)
去除高相关因子：
- 层次化筛选
- 贪心筛选
- 聚类筛选

```python
from factors.selector import CorrelationFilter

corr_filter = CorrelationFilter(
    max_correlation=0.7,
    method='hierarchical'
)

filtered_factors = corr_filter.filter(
    factors, 
    evaluation_results,
    correlation_matrix=corr_matrix
)
```

#### 稳定性筛选器 (StabilityFilter)
基于稳定性指标筛选：
- 稳定性得分
- IC波动率
- 正向期数占比

```python
from factors.selector import StabilityFilter

stab_filter = StabilityFilter(
    min_stability_score=50.0,
    max_ic_volatility=0.5
)

filtered_factors = stab_filter.filter(factors, evaluation_results)
```

#### 复合筛选器 (CompositeFilter)
组合多个筛选器：
- AND逻辑：所有筛选器都通过
- OR逻辑：任一筛选器通过

```python
from factors.selector import CompositeFilter

composite = CompositeFilter(
    filters=[perf_filter, stab_filter],
    logic='AND'
)

filtered_factors = composite.filter(factors, evaluation_results)
```

### 3. 选择策略 (Strategies)

#### TopN选择器 (TopNSelector)
选择评分最高的N个因子：

```python
from factors.selector import TopNSelector

selector = TopNSelector(
    n_factors=10,
    score_metric='total_score',
    tie_breaker='ic_mean'
)

result = selector.select(factors, evaluation_results)
```

#### 阈值选择器 (ThresholdSelector)
基于多个阈值条件选择：

```python
from factors.selector import ThresholdSelector

selector = ThresholdSelector(
    thresholds={
        'total_score': 70.0,
        'ic_mean': 0.02,
        'icir': 0.3
    },
    logic='AND'  # 或 'OR'
)

result = selector.select(factors, evaluation_results)
```

#### 聚类选择器 (ClusteringSelector)
基于聚类方法选择代表性因子：

```python
from factors.selector import ClusteringSelector

selector = ClusteringSelector(
    n_clusters=5,
    clustering_method='kmeans',  # 或 'hierarchical'
    feature_selection='performance',  # 或 'correlation', 'mixed'
    factors_per_cluster=2
)

result = selector.select(factors, evaluation_results)
```

### 4. 主选择器 (FactorSelector)

整合筛选和选择功能的主类：

```python
from factors.selector import FactorSelector

selector = FactorSelector(
    method='top_n',
    config={
        'n_factors': 10,
        'score_metric': 'total_score',
        'filters': {
            'use_performance_filter': True,
            'performance': {
                'min_score': 60.0,
                'min_ic': 0.01
            },
            'use_stability_filter': True,
            'use_correlation_filter': True,
            'correlation': {
                'max_correlation': 0.7
            }
        }
    }
)

# 执行选择
result = selector.select(
    factors_pool=factors,
    evaluation_results=evaluation_results,
    correlation_matrix=correlation_matrix
)

# 获取结果
selected_factors = result['selected_factors']
selection_scores = result['selection_scores']
filtering_summary = result['filtering_summary']
performance_stats = result['performance_stats']
```

## 选择结果格式

所有选择方法返回统一的结果格式：

```python
{
    'selected_factors': ['factor1', 'factor2', ...],  # 选中的因子名称
    'factors_data': {'factor1': Series, ...},         # 因子数据
    'selection_scores': {'factor1': 0.85, ...},       # 选择得分
    'selection_reasons': {'factor1': 'reason', ...},   # 选择原因
    'rejected_factors': {'factor3': 'reason', ...},    # 未选中原因
    'selection_method': 'top_n',                       # 选择方法
    'selection_params': {...},                         # 选择参数
    'summary': {                                        # 摘要信息
        'total_candidates': 20,
        'selected_count': 10,
        'avg_score': 0.75
    },
    'filtering_summary': {...},                        # 筛选摘要
    'performance_stats': {...},                        # 性能统计
    'execution_info': {...}                            # 执行信息
}
```

## 使用示例

### 完整的因子选择流程

```python
from factors.selector import FactorSelector, FactorPool
import pandas as pd

# 1. 创建因子池
pool = FactorPool()

# 添加因子到池中
for name, factor_data in your_factors.items():
    pool.add_factor(
        name=name,
        factor_data=factor_data,
        category=determine_category(name),
        description=f'因子{name}的描述'
    )

# 2. 配置选择器
selector = FactorSelector(
    method='top_n',
    config={
        'n_factors': 15,
        'filters': {
            'use_performance_filter': True,
            'performance': {'min_score': 65},
            'use_correlation_filter': True,
            'correlation': {'max_correlation': 0.6}
        }
    }
)

# 3. 执行选择
factors_pool = pool.get_factors()
result = selector.select(
    factors_pool=factors_pool,
    evaluation_results=your_evaluation_results
)

# 4. 获取选中的因子
selected_factors = result['factors_data']
print(f"选中了 {len(selected_factors)} 个因子")

# 5. 查看选择详情
for name in result['selected_factors']:
    print(f"{name}: {result['selection_reasons'][name]}")
```

### 自定义筛选流程

```python
from factors.selector import (
    PerformanceFilter, 
    CorrelationFilter, 
    TopNSelector
)

# 1. 创建自定义筛选器
filters = [
    PerformanceFilter(min_score=70, min_ic=0.02),
    CorrelationFilter(max_correlation=0.6, method='greedy')
]

# 2. 逐步筛选
current_factors = your_factors.copy()
for filter_obj in filters:
    current_factors = filter_obj.filter(
        current_factors, 
        your_evaluation_results
    )
    print(f"筛选后剩余 {len(current_factors)} 个因子")

# 3. 最终选择
selector = TopNSelector(n_factors=10)
result = selector.select(current_factors, your_evaluation_results)
```

## 性能特点

- **高效筛选**：向量化操作，支持大规模因子池
- **灵活配置**：支持多种组合的筛选和选择策略
- **详细日志**：完整的执行日志和性能统计
- **错误处理**：robust的异常处理和边界情况处理
- **内存友好**：支持增量处理和内存优化

## 扩展性

模块设计遵循开闭原则，易于扩展：

1. **新增筛选器**：继承`BaseFilter`类
2. **新增选择策略**：继承`SelectorBase`类  
3. **自定义评分函数**：实现`score_factors`方法
4. **新增约束条件**：扩展`constraints`参数

## 版本信息

- **版本**: 1.0.0
- **状态**: 已完成基础框架和核心功能
- **测试覆盖**: 包含单元测试和集成测试