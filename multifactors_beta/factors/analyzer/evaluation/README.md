# 因子综合评估模块 (Factor Evaluation Module)

## 概述

因子综合评估模块提供了一个全面的因子质量评估框架，通过五个维度对因子进行多角度分析，帮助量化研究人员识别高质量的alpha因子。

## 核心功能

### 1. 五维度评估体系

| 维度 | 权重(默认) | 评估内容 | 核心指标 |
|------|------------|----------|----------|
| **收益能力** (Profitability) | 35% | 因子的收益预测能力 | IC均值、ICIR、夏普比率、多空收益 |
| **稳定性** (Stability) | 25% | 因子表现的稳定性 | IC波动率、自相关性、结构突变 |
| **可交易性** (Tradability) | 20% | 实际交易的可行性 | 换手率、交易成本、流动性、容量 |
| **独特性** (Uniqueness) | 10% | 因子的信息增量 | 相关性、正交性、独立贡献 |
| **时效性** (Timeliness) | 10% | 预测的时间效力 | IC衰减、最优持仓期、信号稳定性 |

### 2. 场景化评估

系统支持四种预设场景，每种场景有不同的维度权重配置：

- **balanced** (均衡): 默认权重配置，适合大多数策略
- **high_frequency** (高频): 强调可交易性(35%)和时效性(15%)
- **value_investing** (价值投资): 强调收益能力(40%)和稳定性(30%)
- **risk_neutral** (风险中性): 强调稳定性(35%)，降低风险暴露

### 3. 评级体系

采用九级评分制度：

| 等级 | 分数范围 | 含义 |
|------|----------|------|
| AAA | 90-100 | 卓越 |
| AA | 85-90 | 优秀 |
| A | 80-85 | 良好 |
| BBB | 70-80 | 中上 |
| BB | 60-70 | 中等 |
| B | 50-60 | 中下 |
| CCC | 40-50 | 较差 |
| C | 30-40 | 差 |
| D/F | <30 | 不合格 |

## 模块结构

```
evaluation/
├── dimensions/              # 维度评估模块
│   ├── base_dimension.py   # 基础维度类
│   ├── profitability.py    # 收益能力维度
│   ├── stability.py        # 稳定性维度
│   ├── tradability.py      # 可交易性维度
│   ├── uniqueness.py       # 独特性维度
│   └── timeliness.py       # 时效性维度
├── scoring/                 # 评分系统
│   └── score_calculator.py # 分数计算器
├── diagnostics/            # 诊断系统（Phase 3）
├── factor_evaluator.py     # 主评估器
└── DESIGN.md              # 详细设计文档
```

## 快速开始

### 基础使用

```python
from factors.analyzer.evaluation import FactorEvaluator
from factors.tester import FactorTester

# 1. 先进行因子测试
tester = FactorTester()
test_result = tester.test(factor_data, price_data)

# 2. 创建评估器
evaluator = FactorEvaluator(scenario='balanced')

# 3. 执行评估
evaluation_result = evaluator.evaluate(test_result)

# 4. 查看结果
print(f"总分: {evaluation_result.total_score:.1f}")
print(f"等级: {evaluation_result.grade}")
print(f"优势: {evaluation_result.strengths}")
print(f"劣势: {evaluation_result.weaknesses}")
```

### 高级功能

```python
# 1. 结合其他分析结果
from factors.analyzer.correlation import CorrelationAnalyzer
from factors.analyzer.stability import StabilityAnalyzer

# 相关性分析
corr_analyzer = CorrelationAnalyzer()
correlation_result = corr_analyzer.analyze(factor_dict)

# 稳定性分析
stab_analyzer = StabilityAnalyzer()
stability_result = stab_analyzer.analyze(test_result)

# 综合评估
evaluation_result = evaluator.evaluate(
    test_result,
    correlation_result=correlation_result,
    stability_result=stability_result
)

# 2. 批量评估多个因子
factors = {
    'momentum': test_result_1,
    'value': test_result_2,
    'quality': test_result_3
}

batch_results = evaluator.batch_evaluate(factors)

# 3. 因子对比和排名
comparison_df = evaluator.compare_factors(batch_results)
ranking_df = evaluator.rank_factors(batch_results)
top_factors = evaluator.recommend_top_factors(n=5, min_score=60)

# 4. 生成评估报告
report = evaluator.generate_evaluation_report(evaluation_result)
print(report)
```

## 维度详解

### 1. 收益能力维度 (ProfitabilityDimension)

评估因子的收益预测能力，主要指标包括：

- **IC均值** (30%): 因子与未来收益的相关性
- **ICIR** (30%): IC的稳定性，越高越好
- **夏普比率** (20%): 风险调整后收益
- **多空收益** (10%): 多空组合的年化收益
- **最大组收益** (10%): 最优组的表现

### 2. 稳定性维度 (StabilityDimension)

评估因子表现的稳定性和一致性：

- **IC稳定性** (30%): IC序列的波动率
- **收益稳定性** (25%): 收益序列的稳定性
- **因子值稳定性** (20%): 因子值的自相关性
- **市场一致性** (15%): 不同市场状态下的表现
- **结构稳定性** (10%): 是否存在结构性突变

### 3. 可交易性维度 (TradabilityDimension)

评估因子的实际可交易性：

- **换手率** (30%): 调仓频率，影响交易成本
- **交易成本** (25%): 预估成本占收益比例
- **流动性** (20%): 标的股票的流动性
- **覆盖度** (15%): 可投资股票数量
- **策略容量** (10%): 策略的资金容量限制

### 4. 独特性维度 (UniquenessDimension)

评估因子的独特价值和信息增量：

- **最大相关性** (25%): 与其他因子的最大相关性
- **平均相关性** (25%): 与其他因子的平均相关性
- **独立贡献** (20%): R²增量或独立解释力
- **正交性** (15%): 与现有因子的正交程度
- **信息增量** (15%): 对组合的信息贡献

### 5. 时效性维度 (TimelinesDimension)

评估因子预测的时间效力：

- **IC衰减率** (25%): IC随时间的衰减速度
- **预测持续性** (25%): 预测能力的持续时间
- **最优持仓期** (20%): 最佳的调仓周期
- **IC半衰期** (15%): IC下降到一半的时间
- **信号稳定性** (15%): 因子信号的翻转频率

## API 参考

### FactorEvaluator

主评估器类，协调各维度评估并生成综合结果。

```python
class FactorEvaluator(AnalyzerBase):
    def __init__(self, scenario='balanced', config=None):
        """
        Parameters:
            scenario: 评估场景
            config: 配置参数
        """
    
    def evaluate(self, test_result, correlation_result=None, 
                stability_result=None, **kwargs):
        """执行单个因子评估"""
    
    def batch_evaluate(self, factors, **kwargs):
        """批量评估多个因子"""
    
    def compare_factors(self, evaluation_results, dimensions=None):
        """因子对比分析"""
    
    def rank_factors(self, evaluation_results, by='total_score'):
        """因子排名"""
    
    def recommend_top_factors(self, n=10, min_score=60):
        """推荐顶级因子"""
```

### EvaluationResult

评估结果数据类，包含所有评估信息。

```python
@dataclass
class EvaluationResult:
    factor_name: str              # 因子名称
    evaluation_time: datetime     # 评估时间
    scenario: str                 # 评估场景
    dimension_scores: Dict        # 各维度得分
    total_score: float           # 总分
    grade: str                   # 等级
    metrics: Dict                # 详细指标
    dimension_results: Dict      # 维度详细结果
    strengths: List[str]         # 优势
    weaknesses: List[str]        # 劣势
    warnings: List[str]          # 预警
    suggestions: List[str]       # 建议
    recommendation: Dict         # 使用推荐
```

## 配置选项

### 全局配置

```python
config = {
    # 评估标准
    'min_ic': 0.02,           # 最小IC阈值
    'min_icir': 0.3,          # 最小ICIR阈值
    'max_turnover': 0.8,      # 最大换手率
    
    # 交易成本
    'transaction_cost': {
        'commission': 0.0005,   # 佣金率
        'slippage': 0.001,      # 滑点
        'stamp_tax': 0.001      # 印花税
    },
    
    # 输出设置
    'save_results': True,       # 是否保存结果
    'output_dir': './results',  # 输出目录
    'report_format': 'html'     # 报告格式
}

evaluator = FactorEvaluator(config=config)
```

### 维度特定配置

每个维度都可以有自定义的评分阈值：

```python
config = {
    'ic_thresholds': {
        'excellent': 0.05,
        'good': 0.03,
        'fair': 0.02,
        'poor': 0.01
    },
    'turnover_thresholds': {
        'excellent': 2.0,  # 年换手率
        'good': 4.0,
        'fair': 8.0,
        'poor': 12.0
    }
}
```

## 最佳实践

1. **完整的因子评估流程**
   - 先运行FactorTester获取测试结果
   - 运行CorrelationAnalyzer分析因子相关性
   - 运行StabilityAnalyzer分析稳定性
   - 最后使用FactorEvaluator综合评估

2. **选择合适的场景**
   - 日内策略使用high_frequency场景
   - 长期投资使用value_investing场景
   - 市场中性策略使用risk_neutral场景
   - 不确定时使用balanced场景

3. **关注关键指标**
   - IC > 0.03 且 ICIR > 0.5 通常是好因子
   - 年换手率 < 10 有助于控制成本
   - 与现有因子相关性 < 0.5 确保独特性

4. **批量评估和筛选**
   - 使用batch_evaluate同时评估多个因子
   - 设置min_score筛选高质量因子
   - 使用compare_factors进行横向对比

## 注意事项

1. **数据要求**
   - 需要完整的TestResult作为输入
   - 相关性和稳定性分析结果可选但推荐提供
   - 因子数据应该是标准的MultiIndex格式

2. **性能考虑**
   - 批量评估大量因子时考虑使用并行处理
   - 可以缓存中间结果避免重复计算
   - 对于大规模数据考虑采样评估

3. **解释性**
   - 总分和等级仅供参考，需结合具体业务理解
   - 关注diagnostics中的具体建议
   - 不同场景下的评分不能直接比较

## 更新日志

### v2.0.0 (2024-01)
- 实现五维度评估体系
- 支持场景化评估
- 集成相关性和稳定性分析
- 添加批量评估功能

### 后续计划 (Phase 3)
- 实现智能诊断系统
- 添加因子组合优化
- 支持自定义评估维度
- 添加机器学习评分模型