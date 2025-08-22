# 因子选择模块（Selector）开发计划

## 模块概述

因子选择模块（selector）负责从因子池中智能选择最优的因子子集，基于评估结果、相关性分析和性能指标进行筛选，是构建稳健多因子策略的关键环节。

## 模块定位

### 在整体架构中的位置
```
factors/
├── generator/  → 生成单因子
├── tester/     → 测试单因子
├── analyzer/   → 分析评估因子
├── combiner/   → 组合多因子
├── selector/   → 选择最优因子 ← 本模块
└── utils/      → 工具支持
```

### 与其他模块的关系
- **依赖**: analyzer（评估结果）、tester（测试结果）
- **被依赖**: combiner（提供筛选后的因子）
- **协作**: 与combiner配合构建最终的多因子组合

## 核心接口设计

### 输入数据格式

#### 1. 因子池
```python
factors_pool = {
    'factor_name': pd.Series(  # MultiIndex Series
        index=pd.MultiIndex.from_product([dates, stocks]),
        data=values
    ),
    ...
}
```

#### 2. 评估结果
```python
evaluation_results = {
    'factor_name': EvaluationResult(
        total_score=85.2,
        dimension_scores={...},
        metrics={...},
        correlation_result={...}
    ),
    ...
}
```

#### 3. 选择配置
```python
selection_config = {
    'method': 'top_n',              # 选择方法
    'n_factors': 10,                # 选择数量
    'min_score': 60,                # 最低分数
    'max_correlation': 0.7,         # 最大相关性
    'diversity_weight': 0.2,        # 多样性权重
    'reselect_freq': 'quarterly'    # 重选频率
}
```

### 输出数据格式

```python
# 选择结果
selection_result = {
    'selected_factors': ['factor1', 'factor2', ...],  # 选中的因子名称
    'factors_data': Dict[str, pd.Series],            # 因子数据
    'selection_scores': Dict[str, float],            # 选择得分
    'selection_reason': Dict[str, str],              # 选择原因
    'rejected_factors': Dict[str, str]               # 未选中因子及原因
}
```

## 模块结构

```
factors/selector/
├── __init__.py                    # 模块导出
├── DEVELOPMENT_PLAN.md            # 本文档
├── README.md                      # 使用说明
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
│   ├── liquidity_filter.py       # 流动性筛选
│   └── composite_filter.py       # 复合筛选
│
├── strategies/                    # 选择策略
│   ├── __init__.py
│   ├── top_n.py                  # TopN选择
│   ├── threshold.py              # 阈值选择
│   ├── clustering.py             # 聚类选择
│   ├── sequential.py             # 序贯选择
│   └── adaptive.py               # 自适应选择
│
├── optimizer/                     # 优化选择
│   ├── __init__.py
│   ├── max_diversity.py          # 最大多样性
│   ├── min_correlation.py        # 最小相关性
│   └── efficient_frontier.py     # 有效前沿
│
├── factor_selector.py             # 主选择器
├── factor_pool.py                 # 因子池管理
└── utils.py                       # 工具函数
```

## 核心类设计

### 1. SelectorBase（基类）

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

class SelectorBase(ABC):
    """因子选择器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.selection_history = []
        
    @abstractmethod
    def select(self,
              factors_pool: Dict[str, pd.Series],
              evaluation_results: Optional[Dict] = None,
              constraints: Optional[Dict] = None,
              **kwargs) -> Dict[str, Any]:
        """选择因子"""
        pass
    
    @abstractmethod
    def score_factors(self,
                     factors_pool: Dict[str, pd.Series],
                     evaluation_results: Optional[Dict] = None,
                     **kwargs) -> Dict[str, float]:
        """为因子打分"""
        pass
    
    def apply_filters(self,
                     factors_pool: Dict[str, pd.Series],
                     filters: List[Any]) -> Dict[str, pd.Series]:
        """应用筛选器"""
        filtered = factors_pool.copy()
        for filter_obj in filters:
            filtered = filter_obj.filter(filtered)
        return filtered
    
    def check_diversity(self,
                       selected_factors: List[str],
                       correlation_matrix: pd.DataFrame) -> float:
        """检查因子多样性"""
        if len(selected_factors) < 2:
            return 1.0
        
        # 计算选中因子间的平均相关性
        correlations = []
        for i, f1 in enumerate(selected_factors):
            for f2 in selected_factors[i+1:]:
                if f1 in correlation_matrix.index and f2 in correlation_matrix.index:
                    correlations.append(abs(correlation_matrix.loc[f1, f2]))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        diversity = 1 - avg_correlation
        return diversity
```

### 2. FactorSelector（主选择器）

```python
class FactorSelector:
    """因子选择器主类"""
    
    def __init__(self, 
                method: str = 'top_n',
                config: Optional[Dict] = None):
        self.method = method
        self.config = config or {}
        self.filters = []
        self.selection_history = []
        self._init_components()
    
    def select(self,
              factors_pool: Dict[str, pd.Series],
              evaluation_results: Optional[Dict] = None,
              test_results: Optional[Dict] = None,
              correlation_matrix: Optional[pd.DataFrame] = None,
              **kwargs) -> Dict[str, Any]:
        """
        选择最优因子组合
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            候选因子池
        evaluation_results : Optional[Dict]
            因子评估结果
        test_results : Optional[Dict]
            因子测试结果
        correlation_matrix : Optional[pd.DataFrame]
            因子相关性矩阵
            
        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        # 第一步：基础筛选
        filtered_factors = self._apply_basic_filters(
            factors_pool,
            evaluation_results
        )
        
        # 第二步：相关性筛选
        if correlation_matrix is not None:
            filtered_factors = self._apply_correlation_filter(
                filtered_factors,
                correlation_matrix
            )
        
        # 第三步：性能筛选
        if evaluation_results:
            filtered_factors = self._apply_performance_filter(
                filtered_factors,
                evaluation_results
            )
        
        # 第四步：最终选择
        selected = self._final_selection(
            filtered_factors,
            evaluation_results,
            correlation_matrix
        )
        
        # 第五步：生成结果
        result = self._generate_result(
            selected,
            factors_pool,
            evaluation_results
        )
        
        # 记录历史
        self.selection_history.append({
            'timestamp': pd.Timestamp.now(),
            'selected': result['selected_factors'],
            'method': self.method
        })
        
        return result
    
    def select_complementary(self,
                           base_factors: List[str],
                           candidates: Dict[str, pd.Series],
                           correlation_matrix: pd.DataFrame,
                           n_additional: int = 5) -> List[str]:
        """
        选择互补因子
        
        Parameters
        ----------
        base_factors : List[str]
            基础因子
        candidates : Dict[str, pd.Series]
            候选因子
        correlation_matrix : pd.DataFrame
            相关性矩阵
        n_additional : int
            额外选择数量
            
        Returns
        -------
        List[str]
            互补因子列表
        """
        complementary = []
        remaining = list(candidates.keys())
        
        for _ in range(n_additional):
            best_factor = None
            best_score = float('-inf')
            
            for factor in remaining:
                if factor in base_factors + complementary:
                    continue
                
                # 计算与已选因子的平均相关性
                correlations = []
                for selected in base_factors + complementary:
                    if factor in correlation_matrix.index and \
                       selected in correlation_matrix.index:
                        correlations.append(
                            abs(correlation_matrix.loc[factor, selected])
                        )
                
                avg_corr = np.mean(correlations) if correlations else 0
                
                # 互补性得分（相关性越低越好）
                score = 1 - avg_corr
                
                if score > best_score:
                    best_score = score
                    best_factor = factor
            
            if best_factor:
                complementary.append(best_factor)
                remaining.remove(best_factor)
        
        return complementary
```

### 3. 筛选器设计

```python
class PerformanceFilter:
    """性能筛选器"""
    
    def __init__(self, 
                min_score: float = 60,
                min_ic: float = 0.02,
                min_ir: float = 0.3):
        self.min_score = min_score
        self.min_ic = min_ic
        self.min_ir = min_ir
    
    def filter(self,
              factors: Dict[str, pd.Series],
              evaluation_results: Dict) -> Dict[str, pd.Series]:
        """筛选高性能因子"""
        filtered = {}
        
        for name, factor in factors.items():
            if name not in evaluation_results:
                continue
            
            result = evaluation_results[name]
            
            # 检查总分
            if result.total_score < self.min_score:
                continue
            
            # 检查IC
            if result.metrics.get('ic_mean', 0) < self.min_ic:
                continue
            
            # 检查IR
            if result.metrics.get('icir', 0) < self.min_ir:
                continue
            
            filtered[name] = factor
        
        return filtered

class CorrelationFilter:
    """相关性筛选器"""
    
    def __init__(self, max_correlation: float = 0.7):
        self.max_correlation = max_correlation
    
    def filter(self,
              factors: Dict[str, pd.Series],
              correlation_matrix: pd.DataFrame,
              keep_best: bool = True) -> Dict[str, pd.Series]:
        """去除高相关因子"""
        if correlation_matrix is None or correlation_matrix.empty:
            return factors
        
        # 构建相关性图
        factor_names = list(factors.keys())
        to_remove = set()
        
        for i, f1 in enumerate(factor_names):
            if f1 in to_remove:
                continue
            
            for f2 in factor_names[i+1:]:
                if f2 in to_remove:
                    continue
                
                if f1 in correlation_matrix.index and \
                   f2 in correlation_matrix.index:
                    corr = abs(correlation_matrix.loc[f1, f2])
                    
                    if corr > self.max_correlation:
                        # 保留评分更高的因子
                        if keep_best:
                            # 这里需要评估结果来决定保留哪个
                            to_remove.add(f2)
                        else:
                            to_remove.add(f2)
        
        # 过滤
        filtered = {
            name: factor 
            for name, factor in factors.items() 
            if name not in to_remove
        }
        
        return filtered
```

## 开发计划

### Phase 1: 基础框架（Week 1）

#### Day 1-2: 搭建模块结构
- [ ] 创建目录结构
- [ ] 实现SelectorBase基类
- [ ] 设计选择结果数据结构
- [ ] 创建单元测试框架

#### Day 3-4: 基础筛选器
- [ ] PerformanceFilter - 性能筛选
- [ ] CorrelationFilter - 相关性筛选
- [ ] StabilityFilter - 稳定性筛选

#### Day 5: TopN选择策略
- [ ] TopNSelector - 选择评分最高的N个因子
- [ ] 支持多种排序标准
- [ ] 处理并列情况

### Phase 2: 高级策略（Week 2）

#### Day 6-7: 阈值选择
- [ ] ThresholdSelector - 基于阈值的选择
- [ ] 多维度阈值设置
- [ ] 灵活的逻辑组合（AND/OR）

#### Day 8-9: 聚类选择
- [ ] ClusteringSelector - 基于聚类的选择
- [ ] 从每个类别选择代表因子
- [ ] 保证多样性

#### Day 10: 序贯选择
- [ ] SequentialSelector - 前向/后向选择
- [ ] 基于边际贡献
- [ ] 早停机制

### Phase 3: 优化方法（Week 3）

#### Day 11-12: 多样性优化
- [ ] MaxDiversityOptimizer
- [ ] 多样性度量方法
- [ ] 优化算法实现

#### Day 13-14: 相关性优化
- [ ] MinCorrelationOptimizer
- [ ] 构建优化问题
- [ ] 求解器集成

#### Day 15: 自适应选择
- [ ] AdaptiveSelector
- [ ] 根据市场状态调整
- [ ] 历史表现学习

### Phase 4: 集成和测试（Week 4）

#### Day 16-17: 与其他模块集成
- [ ] 集成analyzer评估结果
- [ ] 与combiner模块协作
- [ ] 端到端工作流测试

#### Day 18-19: 测试和优化
- [ ] 完整的单元测试
- [ ] 集成测试
- [ ] 性能优化
- [ ] 边界条件处理

#### Day 20: 文档和发布
- [ ] API文档
- [ ] 使用示例
- [ ] 最佳实践
- [ ] 发布准备

## 选择策略详解

### 1. TopN策略
```python
# 选择评分最高的10个因子
selector = FactorSelector(method='top_n')
result = selector.select(
    factors_pool,
    evaluation_results,
    n_factors=10
)
```

### 2. 阈值策略
```python
# 选择满足条件的因子
selector = FactorSelector(method='threshold')
result = selector.select(
    factors_pool,
    evaluation_results,
    min_score=70,
    min_ic=0.03,
    max_correlation=0.6
)
```

### 3. 聚类策略
```python
# 从每个类别选择最佳因子
selector = FactorSelector(method='clustering')
result = selector.select(
    factors_pool,
    evaluation_results,
    n_clusters=5,
    factors_per_cluster=2
)
```

### 4. 互补选择
```python
# 选择互补性强的因子
selector = FactorSelector(method='complementary')
result = selector.select_complementary(
    base_factors=['momentum', 'value'],
    candidates=factors_pool,
    correlation_matrix=corr_matrix,
    n_additional=5
)
```

## 性能指标

### 目标性能
- 1000个因子筛选时间 < 2秒
- 内存占用 < 500MB
- 支持增量选择

### 优化策略
- 使用numpy向量化操作
- 缓存相关性计算结果
- 并行处理大规模因子池

## 配置管理

### 默认配置
```python
DEFAULT_CONFIG = {
    'selection': {
        'method': 'top_n',
        'n_factors': 10,
        'min_score': 60,
        'max_correlation': 0.7
    },
    'filters': {
        'performance': {
            'min_ic': 0.02,
            'min_ir': 0.3,
            'min_sharpe': 0.5
        },
        'stability': {
            'min_stability_score': 50,
            'max_volatility': 0.3
        },
        'liquidity': {
            'min_coverage': 0.5,
            'max_turnover': 10
        }
    },
    'optimization': {
        'diversity_weight': 0.3,
        'performance_weight': 0.7
    }
}
```

### 场景配置
```python
SCENARIO_CONFIGS = {
    'conservative': {
        'n_factors': 5,
        'min_score': 75,
        'max_correlation': 0.5
    },
    'aggressive': {
        'n_factors': 20,
        'min_score': 55,
        'max_correlation': 0.8
    },
    'balanced': {
        'n_factors': 10,
        'min_score': 65,
        'max_correlation': 0.7
    }
}
```

## 测试策略

### 单元测试
- 每个筛选器的功能测试
- 选择策略的正确性验证
- 边界条件处理

### 集成测试
- 完整选择流程
- 与evaluation结果集成
- 大规模因子池处理

### 性能测试
- 不同规模的处理时间
- 内存使用监控
- 并发处理能力

### 对比测试
- 不同选择策略的效果对比
- 与基准方法对比

## 风险和缓解

### 技术风险
1. **过拟合风险**
   - 风险：选择的因子在样本内表现好但样本外差
   - 缓解：使用滚动窗口验证，考虑稳定性指标

2. **计算复杂度**
   - 风险：大规模因子池选择耗时过长
   - 缓解：实现高效算法，支持并行计算

### 业务风险
1. **因子失效**
   - 风险：选中的因子快速失效
   - 缓解：定期重新评估，动态调整

2. **过度集中**
   - 风险：选中的因子过于相似
   - 缓解：强制多样性约束

## 成功标准

1. **功能完整性**
   - 实现所有计划的选择策略
   - 支持灵活的筛选条件
   - 与其他模块无缝集成

2. **性能要求**
   - 满足目标性能指标
   - 支持大规模因子池
   - 实时选择能力

3. **易用性**
   - 清晰的API设计
   - 丰富的示例代码
   - 完整的文档

4. **稳健性**
   - 处理异常情况
   - 支持增量更新
   - 结果可重现

## 后续计划

### 近期（1-2个月）
- 机器学习选择方法
- 因子重要性评估
- 交互式选择工具

### 中期（3-6个月）
- 深度学习选择模型
- 强化学习优化
- 自动化选择流程

### 长期（6-12个月）
- 云端选择服务
- 实时因子筛选
- 智能推荐系统

---

**文档版本**: 1.0.0  
**创建日期**: 2024-01-15  
**负责人**: MultiFactors Team  
**状态**: Planning