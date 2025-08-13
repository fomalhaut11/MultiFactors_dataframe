# FactorEvaluator 综合评估模块设计文档

## 📋 目录
1. [模块概述](#模块概述)
2. [评估体系](#评估体系)
3. [模块结构](#模块结构)
4. [数据流设计](#数据流设计)
5. [接口定义](#接口定义)
6. [实现路径](#实现路径)

---

## 模块概述

### 定位
FactorEvaluator是analyzer模块的核心组件，负责对因子进行全方位的综合评估，为因子选择和组合构建提供决策支持。

### 核心价值
- 提供标准化的因子评估框架
- 多维度综合评分体系
- 场景化的评估策略
- 智能的诊断和建议

---

## 评估体系

### 五维评估模型

```
总分 = Σ(维度得分 × 维度权重)
```

#### 1. 收益能力维度 (Profitability) - 35%
**评估指标**：
- IC均值 (ic_mean)
- IC信息比率 (icir)
- 夏普比率 (sharpe_ratio)
- 多空组合收益 (long_short_return)
- 最大组收益 (top_group_return)

**评分标准**：
```python
IC均值评分 = {
    ">= 0.05": 100,
    "0.04-0.05": 80,
    "0.03-0.04": 60,
    "0.02-0.03": 40,
    "< 0.02": 20
}
```

#### 2. 稳定性维度 (Stability) - 25%
**评估指标**：
- IC稳定性 (ic_stability)
- 滚动窗口稳定性 (rolling_stability)
- 最大回撤 (max_drawdown)
- 结构突变 (structural_breaks)
- 市场适应性 (market_adaptability)

**评分标准**：
- 使用StabilityAnalyzer的稳定性评分
- 考虑时间序列的一致性

#### 3. 可交易性维度 (Tradability) - 20%
**评估指标**：
- 换手率 (turnover_rate)
- 交易成本 (transaction_cost)
- 容量限制 (capacity_limit)
- 流动性 (liquidity)

**评分标准**：
```python
换手率评分 = {
    "< 20%": 100,
    "20%-40%": 80,
    "40%-60%": 60,
    "60%-80%": 40,
    "> 80%": 20
}
```

#### 4. 独特性维度 (Uniqueness) - 10%
**评估指标**：
- 与其他因子相关性 (correlation_with_others)
- 信息贡献度 (information_contribution)
- 冗余程度 (redundancy_level)

**评分标准**：
- 基于CorrelationAnalyzer的分析结果
- 低相关性得高分

#### 5. 时效性维度 (Timeliness) - 10%
**评估指标**：
- IC衰减速度 (ic_decay_rate)
- 有效预测期 (effective_period)
- 信号持续性 (signal_persistence)

**评分标准**：
- 半衰期越长得分越高
- 有效期越长得分越高

### 场景化权重配置

```python
SCENARIO_CONFIGS = {
    "balanced": {  # 均衡型（默认）
        "profitability": 0.35,
        "stability": 0.25,
        "tradability": 0.20,
        "uniqueness": 0.10,
        "timeliness": 0.10
    },
    "high_frequency": {  # 高频交易
        "profitability": 0.25,
        "stability": 0.15,
        "tradability": 0.40,
        "uniqueness": 0.10,
        "timeliness": 0.10
    },
    "value_investing": {  # 价值投资
        "profitability": 0.40,
        "stability": 0.35,
        "tradability": 0.10,
        "uniqueness": 0.10,
        "timeliness": 0.05
    },
    "risk_neutral": {  # 风险中性
        "profitability": 0.25,
        "stability": 0.45,
        "tradability": 0.15,
        "uniqueness": 0.10,
        "timeliness": 0.05
    }
}
```

### 综合评级体系

```python
GRADE_SYSTEM = {
    "AAA": {"score": ">=90", "description": "卓越因子"},
    "AA":  {"score": "80-90", "description": "优秀因子"},
    "A":   {"score": "70-80", "description": "良好因子"},
    "BBB": {"score": "60-70", "description": "合格因子"},
    "BB":  {"score": "50-60", "description": "一般因子"},
    "B":   {"score": "40-50", "description": "较差因子"},
    "C":   {"score": "<40", "description": "不推荐使用"}
}
```

---

## 模块结构

```
evaluation/
├── DESIGN.md                    # 本设计文档
├── __init__.py                  # 模块接口
│
├── factor_evaluator.py          # 主评估器
│   ├── FactorEvaluator         # 评估器主类
│   └── EvaluationResult        # 评估结果类
│
├── dimensions/                  # 评估维度实现
│   ├── __init__.py
│   ├── base_dimension.py       # 维度基类
│   ├── profitability.py        # 收益能力维度
│   ├── stability.py            # 稳定性维度
│   ├── tradability.py          # 可交易性维度
│   ├── uniqueness.py           # 独特性维度
│   └── timeliness.py           # 时效性维度
│
├── scoring/                     # 评分系统
│   ├── __init__.py
│   ├── score_calculator.py     # 分数计算
│   ├── weight_manager.py       # 权重管理
│   └── grade_mapper.py         # 等级映射
│
└── diagnostics/                 # 诊断系统
    ├── __init__.py
    ├── factor_diagnostics.py   # 因子诊断
    └── recommendation.py        # 改进建议
```

---

## 数据流设计

```
输入数据流：
TestResult ──┐
             ├──> FactorEvaluator ──> EvaluationResult
CorrelationResult ──┤                      │
                    │                      ├──> 综合评分
StabilityResult ────┘                      ├──> 评级
                                          ├──> 诊断报告
                                          └──> 改进建议

处理流程：
1. 数据收集 -> 从各分析器收集结果
2. 指标提取 -> 提取各维度所需指标
3. 维度评分 -> 计算五个维度的得分
4. 综合评分 -> 加权计算总分
5. 等级评定 -> 映射到评级体系
6. 诊断分析 -> 识别优劣势
7. 生成建议 -> 提供改进方向
```

---

## 接口定义

### 主要接口

```python
class FactorEvaluator:
    
    def __init__(self, scenario: str = 'balanced', config: Dict = None):
        """初始化评估器"""
        
    # 核心评估
    def evaluate(self, 
                test_result: TestResult,
                correlation_result: Optional[Dict] = None,
                stability_result: Optional[Dict] = None,
                **kwargs) -> EvaluationResult:
        """评估单个因子"""
        
    def batch_evaluate(self,
                      factors: Dict[str, TestResult],
                      **kwargs) -> Dict[str, EvaluationResult]:
        """批量评估因子"""
    
    # 比较和排名
    def compare_factors(self,
                       evaluation_results: Dict[str, EvaluationResult],
                       dimensions: Optional[List[str]] = None) -> pd.DataFrame:
        """因子对比分析"""
        
    def rank_factors(self,
                    evaluation_results: Dict[str, EvaluationResult],
                    by: str = 'total_score') -> pd.DataFrame:
        """因子排名"""
    
    # 推荐系统
    def recommend_top_factors(self,
                             n: int = 10,
                             min_score: float = 60) -> List[str]:
        """推荐顶级因子"""
        
    def suggest_portfolio(self,
                         factors: List[str],
                         target: str = 'balanced') -> Dict:
        """建议因子组合"""
    
    # 诊断功能
    def diagnose_factor(self,
                       evaluation_result: EvaluationResult) -> Dict:
        """因子诊断"""
        
    def get_improvement_suggestions(self,
                                   evaluation_result: EvaluationResult) -> List[str]:
        """获取改进建议"""
    
    # 配置管理
    def set_scenario(self, scenario: str):
        """设置评估场景"""
        
    def set_custom_weights(self, weights: Dict[str, float]):
        """自定义权重"""
```

### 数据结构

```python
@dataclass
class EvaluationResult:
    # 基本信息
    factor_name: str
    evaluation_time: datetime
    scenario: str
    
    # 维度得分 (0-100)
    dimension_scores: Dict[str, float] = {
        'profitability': 0.0,
        'stability': 0.0,
        'tradability': 0.0,
        'uniqueness': 0.0,
        'timeliness': 0.0
    }
    
    # 综合评估
    total_score: float  # 0-100
    grade: str  # AAA-C
    rank: Optional[int] = None
    
    # 详细指标
    metrics: Dict[str, Any]
    
    # 诊断信息
    strengths: List[str]
    weaknesses: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    # 推荐信息
    recommendation: Dict[str, Any]
```

---

## 实现路径

### Phase 1: 基础框架（优先）
1. ✅ 创建基础目录结构
2. ✅ 实现维度基类 (base_dimension.py)
3. ✅ 实现评分计算器 (score_calculator.py)
4. ✅ 实现主评估器框架 (factor_evaluator.py)

### Phase 2: 维度实现
1. ⏳ 实现收益能力维度 (profitability.py)
2. ⏳ 实现稳定性维度 (stability.py)
3. ⏳ 实现可交易性维度 (tradability.py)
4. ⏳ 实现独特性维度 (uniqueness.py)
5. ⏳ 实现时效性维度 (timeliness.py)

### Phase 3: 高级功能
1. ⏳ 实现诊断系统 (factor_diagnostics.py)
2. ⏳ 实现推荐系统 (recommendation.py)
3. ⏳ 实现权重管理 (weight_manager.py)
4. ⏳ 实现等级映射 (grade_mapper.py)

### Phase 4: 集成和优化
1. ⏳ 与其他analyzer模块集成
2. ⏳ 性能优化
3. ⏳ 添加可视化功能
4. ⏳ 完善文档和示例

---

## 使用示例

```python
from factors.analyzer.evaluation import FactorEvaluator

# 创建评估器
evaluator = FactorEvaluator(scenario='balanced')

# 评估单个因子
test_result = ...  # 从tester获取
correlation_result = ...  # 从correlation analyzer获取
stability_result = ...  # 从stability analyzer获取

evaluation = evaluator.evaluate(
    test_result,
    correlation_result=correlation_result,
    stability_result=stability_result
)

# 查看结果
print(f"因子: {evaluation.factor_name}")
print(f"总分: {evaluation.total_score:.1f}")
print(f"评级: {evaluation.grade}")
print(f"优势: {', '.join(evaluation.strengths)}")
print(f"建议: {', '.join(evaluation.suggestions)}")

# 批量评估和排名
results = evaluator.batch_evaluate(factor_dict)
ranking = evaluator.rank_factors(results)
print(ranking[['factor', 'total_score', 'grade']].head(10))
```

---

*文档版本: 1.0.0*  
*创建日期: 2025-08-13*  
*作者: AI Assistant*