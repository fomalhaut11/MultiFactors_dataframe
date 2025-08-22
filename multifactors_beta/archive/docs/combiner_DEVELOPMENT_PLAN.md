# 因子组合模块（Combiner）开发计划

## 模块概述

因子组合模块（combiner）负责将多个单因子通过各种方法组合成综合因子，是多因子策略的核心组件。该模块与analyzer模块紧密配合，利用评估结果优化组合权重。

## 模块定位

### 在整体架构中的位置
```
factors/
├── generator/  → 生成单因子
├── tester/     → 测试单因子
├── analyzer/   → 分析评估因子
├── combiner/   → 组合多因子 ← 本模块
├── selector/   → 选择最优因子
└── utils/      → 工具支持
```

### 数据流
```
单因子(generator) → 测试(tester) → 评估(analyzer) → 组合(combiner) → 选择(selector)
```

## 核心接口设计

### 输入数据格式

#### 1. 因子数据
```python
# 标准MultiIndex Series格式
factors_dict = {
    'factor_name': pd.Series(
        index=pd.MultiIndex.from_product([trading_dates, stock_codes]),
        data=factor_values
    )
}
```

#### 2. 评估结果（可选）
```python
evaluation_results = {
    'factor_name': EvaluationResult(
        total_score=85.2,
        dimension_scores={'profitability': 90, 'stability': 80, ...},
        metrics={'ic_mean': 0.05, 'icir': 0.8, ...}
    )
}
```

#### 3. 配置参数
```python
config = {
    'method': 'ic_weight',           # 组合方法
    'rebalance_freq': 'monthly',     # 调仓频率
    'min_weight': 0.0,               # 最小权重
    'max_weight': 1.0,               # 最大权重
    'normalize': True,               # 是否归一化权重
    'handle_missing': 'forward_fill' # 缺失值处理
}
```

### 输出数据格式

```python
# 组合后的因子，保持MultiIndex Series格式
composite_factor = pd.Series(
    index=pd.MultiIndex.from_product([trading_dates, stock_codes]),
    data=combined_values,
    name='composite_factor'
)
```

## 模块结构

```
factors/combiner/
├── __init__.py                    # 模块导出
├── DEVELOPMENT_PLAN.md            # 本文档
├── README.md                      # 使用说明
│
├── base/                          # 基础类
│   ├── __init__.py
│   └── combiner_base.py          # 组合器基类
│
├── weighting/                     # 权重计算方法
│   ├── __init__.py
│   ├── base_weight.py            # 权重计算基类
│   ├── equal_weight.py           # 等权重
│   ├── ic_weight.py              # IC加权
│   ├── ir_weight.py              # IR加权
│   ├── risk_parity.py            # 风险平价
│   └── optimal_weight.py         # 优化权重
│
├── methods/                       # 组合方法
│   ├── __init__.py
│   ├── linear.py                 # 线性组合
│   ├── orthogonal.py             # 正交化
│   ├── pca.py                    # 主成分分析
│   └── neutralization.py         # 中性化
│
├── optimizer/                     # 权重优化器
│   ├── __init__.py
│   ├── mean_variance.py          # 均值-方差优化
│   ├── max_sharpe.py             # 最大夏普比
│   └── risk_budget.py            # 风险预算
│
├── factor_combiner.py             # 主组合器
└── utils.py                       # 工具函数
```

## 核心类设计

### 1. CombinerBase（基类）

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any

class CombinerBase(ABC):
    """因子组合器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    def combine(self, 
               factors: Dict[str, pd.Series],
               weights: Optional[Dict[str, float]] = None,
               **kwargs) -> pd.Series:
        """组合因子"""
        pass
    
    @abstractmethod
    def calculate_weights(self,
                         factors: Dict[str, pd.Series],
                         evaluation_results: Optional[Dict] = None,
                         **kwargs) -> Dict[str, float]:
        """计算权重"""
        pass
    
    def validate_factors(self, factors: Dict[str, pd.Series]) -> bool:
        """验证因子格式"""
        for name, factor in factors.items():
            if not isinstance(factor.index, pd.MultiIndex):
                raise ValueError(f"Factor {name} must have MultiIndex")
            if factor.index.nlevels != 2:
                raise ValueError(f"Factor {name} must have 2-level MultiIndex")
        return True
    
    def align_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """对齐因子索引"""
        # 找到公共索引
        common_index = None
        for factor in factors.values():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        # 对齐所有因子
        aligned = {}
        for name, factor in factors.items():
            aligned[name] = factor.reindex(common_index)
        
        return aligned
```

### 2. FactorCombiner（主组合器）

```python
class FactorCombiner:
    """因子组合器主类"""
    
    def __init__(self, method: str = 'equal_weight', config: Optional[Dict] = None):
        self.method = method
        self.config = config or {}
        self._init_components()
    
    def combine(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               custom_weights: Optional[Dict[str, float]] = None,
               **kwargs) -> pd.Series:
        """
        组合多个因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典，key为因子名，value为因子值
        evaluation_results : Optional[Dict]
            评估结果，用于计算权重
        custom_weights : Optional[Dict[str, float]]
            自定义权重，优先级最高
        
        Returns
        -------
        pd.Series
            组合后的因子
        """
        # 验证和对齐
        self.validate_factors(factors)
        aligned_factors = self.align_factors(factors)
        
        # 计算权重
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.calculate_weights(
                aligned_factors,
                evaluation_results
            )
        
        # 组合因子
        composite = self._combine_linear(aligned_factors, weights)
        
        # 后处理
        if self.config.get('normalize', True):
            composite = self._normalize(composite)
        
        return composite
    
    def orthogonalize(self,
                     factors: Dict[str, pd.Series],
                     method: str = 'gram_schmidt',
                     base_factor: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        正交化因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子
        method : str
            正交化方法
        base_factor : Optional[str]
            基准因子（保持不变）
        
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        pass
```

## 开发计划

### Phase 1: 基础框架（Week 1）

#### Day 1-2: 搭建模块结构
- [ ] 创建目录结构
- [ ] 实现CombinerBase基类
- [ ] 编写基础验证和对齐功能
- [ ] 创建单元测试框架

#### Day 3-4: 实现基础权重方法
- [ ] EqualWeight - 等权重
- [ ] ICWeight - IC加权
- [ ] IRWeight - IR加权
- [ ] 权重归一化和约束处理

#### Day 5: 实现线性组合
- [ ] LinearCombiner - 线性组合实现
- [ ] 缺失值处理
- [ ] 异常值处理

### Phase 2: 高级功能（Week 2）

#### Day 6-7: 正交化处理
- [ ] Gram-Schmidt正交化
- [ ] 残差正交化
- [ ] 主成分正交化

#### Day 8-9: 风险平价方法
- [ ] RiskParity权重计算
- [ ] 协方差矩阵估计
- [ ] 权重优化算法

#### Day 10: 集成评估结果
- [ ] 从EvaluationResult提取指标
- [ ] 基于评分的权重调整
- [ ] 动态权重更新机制

### Phase 3: 优化器（Week 3）

#### Day 11-12: 均值方差优化
- [ ] MeanVarianceOptimizer
- [ ] 约束条件处理
- [ ] 求解器集成

#### Day 13-14: 其他优化方法
- [ ] MaxSharpeOptimizer
- [ ] RiskBudgetOptimizer
- [ ] BlackLitterman方法

#### Day 15: 性能优化
- [ ] 并行计算
- [ ] 缓存机制
- [ ] 内存优化

### Phase 4: 测试和文档（Week 4）

#### Day 16-17: 完整测试
- [ ] 单元测试完善
- [ ] 集成测试
- [ ] 性能测试
- [ ] 边界条件测试

#### Day 18-19: 文档编写
- [ ] API文档
- [ ] 使用示例
- [ ] 最佳实践指南
- [ ] 性能调优指南

#### Day 20: 发布准备
- [ ] 代码审查
- [ ] 性能基准测试
- [ ] 发布说明
- [ ] 版本标记

## 关键技术决策

### 1. 权重计算频率
- **决策**: 支持多种频率（日度、周度、月度）
- **原因**: 不同策略需要不同的调仓频率
- **实现**: 通过rebalance_freq参数控制

### 2. 缺失值处理
- **决策**: 提供多种处理方式（前向填充、删除、插值）
- **原因**: 不同因子的缺失值含义不同
- **实现**: handle_missing参数

### 3. 权重约束
- **决策**: 支持灵活的权重约束（上下限、和为1）
- **原因**: 满足不同的投资限制
- **实现**: 在优化器中实现约束

### 4. 性能优化
- **决策**: 使用numpy向量化操作
- **原因**: 提高大规模数据处理效率
- **实现**: 尽量避免pandas循环操作

## 与其他模块的集成

### 1. 与analyzer模块集成
```python
# 使用评估结果计算权重
from factors.analyzer.evaluation import FactorEvaluator

evaluator = FactorEvaluator()
eval_results = {}
for name, factor in factors.items():
    test_result = tester.test(factor, price_data)
    eval_results[name] = evaluator.evaluate(test_result)

# 基于评估结果组合
combiner = FactorCombiner(method='score_weight')
composite = combiner.combine(factors, evaluation_results=eval_results)
```

### 2. 与selector模块配合
```python
# 先选择后组合
from factors.selector import FactorSelector

selector = FactorSelector()
selected_factors = selector.select(
    factors,
    evaluation_results,
    n_factors=10,
    min_score=60
)

# 组合选中的因子
combiner = FactorCombiner(method='ic_weight')
final_factor = combiner.combine(selected_factors)
```

## 测试策略

### 1. 单元测试
- 每个权重方法的正确性
- 数据对齐功能
- 缺失值处理
- 权重归一化

### 2. 集成测试
- 完整的组合流程
- 与评估结果的集成
- 大规模数据处理

### 3. 性能测试
- 不同数据规模的处理时间
- 内存使用情况
- 并行加速效果

### 4. 对比测试
- 与现有方法对比
- 不同组合方法的效果对比

## 风险和缓解措施

### 1. 技术风险
- **风险**: 优化算法收敛问题
- **缓解**: 提供多种优化器，设置合理的初始值

### 2. 性能风险
- **风险**: 大规模数据处理缓慢
- **缓解**: 实现并行计算，优化内存使用

### 3. 兼容性风险
- **风险**: 与现有模块接口不匹配
- **缓解**: 严格遵循MultiIndex Series格式

## 成功标准

1. **功能完整性**
   - 实现所有计划的组合方法
   - 支持灵活的权重配置
   - 与评估模块无缝集成

2. **性能指标**
   - 1000个因子组合时间 < 1秒
   - 内存占用 < 1GB
   - 支持100万+ 数据点

3. **代码质量**
   - 测试覆盖率 > 80%
   - 文档完整性 100%
   - 代码规范合规性 100%

4. **用户体验**
   - API简洁直观
   - 错误信息清晰
   - 示例代码丰富

## 里程碑

| 时间节点 | 里程碑 | 交付物 |
|---------|--------|--------|
| Week 1 | 基础框架完成 | CombinerBase、基础权重方法、线性组合 |
| Week 2 | 高级功能完成 | 正交化、风险平价、评估集成 |
| Week 3 | 优化器完成 | 各种优化算法、性能优化 |
| Week 4 | 发布就绪 | 完整测试、文档、示例 |

## 后续扩展

### 近期（1-2个月）
- 机器学习权重预测
- 动态权重调整
- 因子时变组合

### 中期（3-6个月）
- 非线性组合方法
- 深度学习组合
- 自适应组合策略

### 长期（6-12个月）
- 实时组合计算
- 分布式处理
- 云端部署

---

**文档版本**: 1.0.0  
**创建日期**: 2024-01-15  
**负责人**: MultiFactors Team  
**状态**: Planning