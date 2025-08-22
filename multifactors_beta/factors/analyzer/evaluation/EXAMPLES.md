# 因子评估模块使用示例

本文档提供了因子评估模块的详细使用示例，从基础用法到高级应用。

## 目录

1. [基础示例](#基础示例)
2. [完整评估流程](#完整评估流程)
3. [场景化评估](#场景化评估)
4. [批量评估](#批量评估)
5. [自定义配置](#自定义配置)
6. [结果分析](#结果分析)
7. [实战案例](#实战案例)

## 基础示例

### 示例1：单因子快速评估

```python
import pandas as pd
from factors.analyzer.evaluation import FactorEvaluator
from factors.tester import FactorTester

# 准备数据（假设已有因子数据和价格数据）
factor_data = pd.Series(...)  # MultiIndex: (date, stock_code)
price_data = pd.DataFrame(...)  # 股票价格数据

# Step 1: 测试因子
tester = FactorTester()
test_result = tester.test(
    factor_data, 
    price_data,
    method='rank_ic'
)

# Step 2: 评估因子
evaluator = FactorEvaluator(scenario='balanced')
evaluation_result = evaluator.evaluate(test_result)

# Step 3: 查看结果
print(f"因子名称: {evaluation_result.factor_name}")
print(f"总分: {evaluation_result.total_score:.1f}/100")
print(f"等级: {evaluation_result.grade}")
print(f"\n维度得分:")
for dim, score in evaluation_result.dimension_scores.items():
    print(f"  {dim}: {score:.1f}")
```

### 示例2：获取详细诊断信息

```python
# 继续使用上面的evaluation_result

# 查看优劣势
print("\n优势:")
for strength in evaluation_result.strengths:
    print(f"  • {strength}")

print("\n劣势:")
for weakness in evaluation_result.weaknesses:
    print(f"  • {weakness}")

# 查看建议
print("\n改进建议:")
for suggestion in evaluation_result.suggestions:
    print(f"  → {suggestion}")

# 查看推荐信息
rec = evaluation_result.recommendation
print(f"\n使用推荐: {rec['usage']}")
print(f"建议权重: {rec['weight']:.1%}")
print(f"优先级: {rec['priority']}")
```

## 完整评估流程

### 示例3：包含相关性和稳定性分析的完整评估

```python
from factors.analyzer.evaluation import FactorEvaluator
from factors.analyzer.correlation import CorrelationAnalyzer
from factors.analyzer.stability import StabilityAnalyzer
from factors.tester import FactorTester

# 假设有多个因子
factors_dict = {
    'momentum': momentum_factor,
    'value': value_factor,
    'quality': quality_factor,
    'growth': growth_factor
}

# Step 1: 测试目标因子
tester = FactorTester()
target_factor = factors_dict['momentum']
test_result = tester.test(target_factor, price_data)

# Step 2: 分析因子相关性
corr_analyzer = CorrelationAnalyzer()
correlation_result = corr_analyzer.analyze(
    factors_dict,
    method=['pearson', 'spearman'],
    clustering=True
)

# Step 3: 分析稳定性
stab_analyzer = StabilityAnalyzer()
stability_result = stab_analyzer.analyze(
    test_result,
    window_size=60,
    check_structural_break=True
)

# Step 4: 综合评估
evaluator = FactorEvaluator(scenario='balanced')
evaluation_result = evaluator.evaluate(
    test_result,
    correlation_result=correlation_result,
    stability_result=stability_result
)

# Step 5: 生成报告
report = evaluator.generate_evaluation_report(evaluation_result)
print(report)

# 保存报告到文件
with open('factor_evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
```

### 示例4：获取维度详细信息

```python
# 获取特定维度的详细结果
profitability_detail = evaluator.get_dimension_details(
    'momentum', 
    'profitability'
)

if profitability_detail:
    print(f"收益能力维度详情:")
    print(f"  得分: {profitability_detail.score:.1f}")
    print(f"  等级: {profitability_detail.grade}")
    print(f"  描述: {profitability_detail.description}")
    print(f"\n  详细指标:")
    for metric, value in profitability_detail.metrics.items():
        print(f"    {metric}: {value:.4f}")
```

## 场景化评估

### 示例5：高频交易场景评估

```python
# 高频交易场景：重视可交易性和时效性
evaluator_hf = FactorEvaluator(scenario='high_frequency')

# 自定义高频交易的评估配置
hf_config = {
    'max_turnover': 20.0,  # 高频允许更高换手率
    'transaction_cost': {
        'commission': 0.0003,  # 更低的佣金
        'slippage': 0.0005,    # 更小的滑点
        'stamp_tax': 0.001
    }
}

evaluator_hf = FactorEvaluator(
    scenario='high_frequency',
    config=hf_config
)

# 评估
hf_result = evaluator_hf.evaluate(test_result)

# 高频场景下的权重分配
print("高频场景维度权重:")
for dim_name, dimension in evaluator_hf.dimensions.items():
    print(f"  {dim_name}: {dimension.weight:.0%}")
```

### 示例6：价值投资场景评估

```python
# 价值投资场景：重视收益能力和稳定性
evaluator_vi = FactorEvaluator(scenario='value_investing')

# 价值投资的配置
vi_config = {
    'min_ic': 0.03,      # 要求更高的IC
    'min_icir': 0.5,     # 要求更高的ICIR
    'max_turnover': 4.0   # 限制换手率（年化）
}

evaluator_vi = FactorEvaluator(
    scenario='value_investing',
    config=vi_config
)

vi_result = evaluator_vi.evaluate(test_result)

# 对比不同场景的评估结果
scenarios = ['balanced', 'high_frequency', 'value_investing', 'risk_neutral']
scenario_results = {}

for scenario in scenarios:
    evaluator = FactorEvaluator(scenario=scenario)
    result = evaluator.evaluate(test_result)
    scenario_results[scenario] = {
        'total_score': result.total_score,
        'grade': result.grade,
        'recommendation': result.recommendation['usage']
    }

# 显示对比结果
comparison_df = pd.DataFrame(scenario_results).T
print("\n不同场景下的评估结果:")
print(comparison_df)
```

## 批量评估

### 示例7：批量评估多个因子

```python
# 准备多个因子的测试结果
test_results = {}
for factor_name, factor_data in factors_dict.items():
    test_results[factor_name] = tester.test(factor_data, price_data)

# 批量评估
evaluator = FactorEvaluator(scenario='balanced')
batch_results = evaluator.batch_evaluate(
    test_results,
    correlation_results=correlation_result,  # 可选
    stability_results=stability_results      # 可选
)

# 查看所有因子的评估结果
for factor_name, result in batch_results.items():
    print(f"\n{factor_name}:")
    print(f"  总分: {result.total_score:.1f}")
    print(f"  等级: {result.grade}")
    print(f"  排名: {result.rank}")
    print(f"  百分位: {result.percentile:.1f}%")
```

### 示例8：因子对比和筛选

```python
# 因子对比
comparison_df = evaluator.compare_factors(batch_results)
print("\n因子对比表:")
print(comparison_df)

# 因子排名
ranking_df = evaluator.rank_factors(
    batch_results,
    by='total_score',
    ascending=False
)
print("\n因子排名:")
print(ranking_df)

# 推荐顶级因子
top_factors = evaluator.recommend_top_factors(
    n=5,          # 推荐前5个
    min_score=60  # 最低分数要求
)
print(f"\n推荐使用的因子: {top_factors}")

# 筛选特定维度表现好的因子
high_profit_factors = [
    name for name, result in batch_results.items()
    if result.dimension_scores.get('profitability', 0) > 80
]
print(f"\n高收益能力因子: {high_profit_factors}")
```

## 自定义配置

### 示例9：自定义评分阈值

```python
# 自定义各维度的评分阈值
custom_config = {
    # 收益能力维度阈值
    'ic_thresholds': {
        'excellent': 0.06,  # 更严格的标准
        'good': 0.04,
        'fair': 0.02,
        'poor': 0.01
    },
    
    # 稳定性维度阈值
    'ic_volatility_thresholds': {
        'excellent': 0.08,
        'good': 0.12,
        'fair': 0.18,
        'poor': 0.25
    },
    
    # 可交易性维度阈值
    'turnover_thresholds': {
        'excellent': 1.5,   # 年换手率
        'good': 3.0,
        'fair': 6.0,
        'poor': 10.0
    }
}

evaluator_custom = FactorEvaluator(
    scenario='balanced',
    config=custom_config
)

custom_result = evaluator_custom.evaluate(test_result)
```

### 示例10：自定义维度权重

```python
# 通过继承创建自定义场景
class CustomEvaluator(FactorEvaluator):
    def _get_scenario_weights(self):
        """自定义维度权重"""
        if self.scenario == 'custom_strategy':
            return {
                'profitability': 0.40,  # 更重视收益
                'stability': 0.20,
                'tradability': 0.25,    # 也重视可交易性
                'uniqueness': 0.10,
                'timeliness': 0.05
            }
        else:
            return super()._get_scenario_weights()

# 使用自定义评估器
custom_evaluator = CustomEvaluator(scenario='custom_strategy')
custom_result = custom_evaluator.evaluate(test_result)
```

## 结果分析

### 示例11：深度分析评估结果

```python
def analyze_evaluation_result(result):
    """深度分析评估结果"""
    
    print(f"因子: {result.factor_name}")
    print("=" * 60)
    
    # 1. 总体评价
    print(f"\n【总体评价】")
    print(f"总分: {result.total_score:.1f}/100 (等级: {result.grade})")
    
    if result.rank:
        print(f"排名: 第{result.rank}位 (百分位: {result.percentile:.1f}%)")
    
    # 2. 维度分析
    print(f"\n【维度分析】")
    dim_scores = result.dimension_scores
    
    # 找出最强和最弱的维度
    best_dim = max(dim_scores.items(), key=lambda x: x[1])
    worst_dim = min(dim_scores.items(), key=lambda x: x[1])
    
    print(f"最强维度: {best_dim[0]} ({best_dim[1]:.1f}分)")
    print(f"最弱维度: {worst_dim[0]} ({worst_dim[1]:.1f}分)")
    
    # 3. 关键指标
    print(f"\n【关键指标】")
    if result.metrics:
        key_metrics = ['ic_mean', 'icir', 'sharpe_ratio', 'annual_turnover']
        for metric in key_metrics:
            if metric in result.metrics:
                value = result.metrics[metric]
                print(f"  {metric}: {value:.4f}")
    
    # 4. 风险提示
    if result.warnings:
        print(f"\n【风险提示】")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    # 5. 使用建议
    print(f"\n【使用建议】")
    rec = result.recommendation
    print(f"  推荐度: {rec['usage']}")
    print(f"  建议权重: {rec['weight']:.1%}")
    
    if result.suggestions:
        print(f"\n  改进方向:")
        for suggestion in result.suggestions:
            print(f"    • {suggestion}")
    
    return result

# 使用分析函数
analyzed_result = analyze_evaluation_result(evaluation_result)
```

### 示例12：导出评估结果

```python
import json
from datetime import datetime

def export_evaluation_results(results, filename=None):
    """导出评估结果为JSON格式"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    export_data = {}
    
    for factor_name, result in results.items():
        export_data[factor_name] = {
            'evaluation_time': result.evaluation_time.isoformat(),
            'scenario': result.scenario,
            'total_score': result.total_score,
            'grade': result.grade,
            'rank': result.rank,
            'percentile': result.percentile,
            'dimension_scores': result.dimension_scores,
            'strengths': result.strengths,
            'weaknesses': result.weaknesses,
            'warnings': result.warnings,
            'suggestions': result.suggestions,
            'recommendation': result.recommendation,
            'key_metrics': {
                k: v for k, v in result.metrics.items()
                if k in ['ic_mean', 'icir', 'sharpe_ratio', 'annual_turnover']
            }
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已导出到: {filename}")
    return filename

# 导出批量评估结果
export_file = export_evaluation_results(batch_results)
```

## 实战案例

### 案例1：动量因子评估

```python
import numpy as np

# 创建动量因子
def create_momentum_factor(price_data, lookback=20):
    """创建动量因子"""
    returns = price_data.pct_change(lookback)
    # 转换为MultiIndex格式
    factor_data = returns.stack()
    factor_data.index.names = ['date', 'stock_code']
    return factor_data

# 生成因子
momentum_factor = create_momentum_factor(price_data, lookback=20)

# 完整评估流程
def evaluate_momentum_factor(factor_data, price_data):
    # 1. 测试
    tester = FactorTester()
    test_result = tester.test(
        factor_data,
        price_data,
        method='rank_ic',
        group_num=5
    )
    
    # 2. 评估（动量因子适合高频场景）
    evaluator = FactorEvaluator(scenario='high_frequency')
    result = evaluator.evaluate(test_result)
    
    # 3. 分析结果
    print(f"\n动量因子评估结果:")
    print(f"总分: {result.total_score:.1f}")
    print(f"等级: {result.grade}")
    
    # 4. 检查是否适合高频交易
    if result.dimension_scores['tradability'] >= 70:
        print("✓ 适合高频交易")
    else:
        print("✗ 不适合高频交易，换手率可能过高")
    
    # 5. 检查时效性
    if result.dimension_scores['timeliness'] >= 60:
        print("✓ 时效性良好")
    else:
        print("✗ 信号衰减较快，需要频繁更新")
    
    return result

momentum_result = evaluate_momentum_factor(momentum_factor, price_data)
```

### 案例2：多因子组合评估

```python
def evaluate_factor_combination(factors_dict, price_data):
    """评估因子组合"""
    
    # 1. 分别测试各因子
    tester = FactorTester()
    test_results = {}
    for name, factor in factors_dict.items():
        test_results[name] = tester.test(factor, price_data)
    
    # 2. 分析因子相关性
    corr_analyzer = CorrelationAnalyzer()
    correlation_result = corr_analyzer.analyze(factors_dict)
    
    # 3. 批量评估
    evaluator = FactorEvaluator(scenario='balanced')
    batch_results = evaluator.batch_evaluate(test_results)
    
    # 4. 筛选互补因子
    selected_factors = []
    used_types = set()
    
    # 按分数排序
    sorted_factors = sorted(
        batch_results.items(),
        key=lambda x: x[1].total_score,
        reverse=True
    )
    
    for factor_name, result in sorted_factors:
        # 筛选条件
        if (result.total_score >= 65 and  # 分数足够高
            result.dimension_scores['uniqueness'] >= 60):  # 独特性好
            
            # 检查与已选因子的相关性
            is_unique = True
            if correlation_result and 'correlation_matrix' in correlation_result:
                corr_matrix = correlation_result['correlation_matrix']
                for selected in selected_factors:
                    if factor_name in corr_matrix.columns and selected in corr_matrix.columns:
                        if abs(corr_matrix.loc[factor_name, selected]) > 0.6:
                            is_unique = False
                            break
            
            if is_unique:
                selected_factors.append(factor_name)
                print(f"选中因子: {factor_name} (分数: {result.total_score:.1f})")
    
    # 5. 计算组合权重
    weights = {}
    total_score = sum(batch_results[f].total_score for f in selected_factors)
    
    for factor in selected_factors:
        # 基于分数的权重
        score_weight = batch_results[factor].total_score / total_score
        # 考虑推荐权重
        rec_weight = batch_results[factor].recommendation['weight']
        # 综合权重
        weights[factor] = (score_weight + rec_weight) / 2
    
    # 归一化
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    print(f"\n因子组合权重:")
    for factor, weight in weights.items():
        print(f"  {factor}: {weight:.1%}")
    
    return selected_factors, weights, batch_results

# 使用示例
factors = {
    'momentum': momentum_factor,
    'value': value_factor,
    'quality': quality_factor,
    'growth': growth_factor,
    'volatility': volatility_factor
}

selected, weights, results = evaluate_factor_combination(factors, price_data)
```

### 案例3：因子改进建议实施

```python
def improve_factor_based_on_evaluation(factor_data, evaluation_result):
    """根据评估结果改进因子"""
    
    improved_factor = factor_data.copy()
    improvements_made = []
    
    # 1. 如果换手率过高，考虑平滑处理
    if 'tradability' in evaluation_result.dimension_scores:
        if evaluation_result.dimension_scores['tradability'] < 50:
            # 应用移动平均平滑
            improved_factor = improved_factor.groupby(level=1).rolling(
                window=3, min_periods=1
            ).mean()
            improvements_made.append("应用3期移动平均降低换手率")
    
    # 2. 如果稳定性差，考虑去极值
    if 'stability' in evaluation_result.dimension_scores:
        if evaluation_result.dimension_scores['stability'] < 50:
            # 去极值处理
            mean = improved_factor.mean()
            std = improved_factor.std()
            improved_factor = improved_factor.clip(
                lower=mean - 3*std,
                upper=mean + 3*std
            )
            improvements_made.append("去除3倍标准差外的极值")
    
    # 3. 如果IC太低，考虑因子变换
    if evaluation_result.metrics.get('ic_mean', 0) < 0.02:
        # 尝试排名变换
        improved_factor = improved_factor.groupby(level=0).rank(pct=True)
        improvements_made.append("应用排名变换提升IC")
    
    print("因子改进措施:")
    for improvement in improvements_made:
        print(f"  • {improvement}")
    
    # 重新评估改进后的因子
    tester = FactorTester()
    new_test_result = tester.test(improved_factor, price_data)
    
    evaluator = FactorEvaluator(scenario=evaluation_result.scenario)
    new_evaluation = evaluator.evaluate(new_test_result)
    
    # 对比改进效果
    print(f"\n改进效果:")
    print(f"原始总分: {evaluation_result.total_score:.1f}")
    print(f"改进后总分: {new_evaluation.total_score:.1f}")
    print(f"提升: {new_evaluation.total_score - evaluation_result.total_score:+.1f}")
    
    return improved_factor, new_evaluation

# 应用改进
improved_factor, new_result = improve_factor_based_on_evaluation(
    momentum_factor,
    momentum_result
)
```

## 总结

本文档展示了因子评估模块的各种使用场景，从基础的单因子评估到复杂的多因子组合分析。关键要点：

1. **完整流程**：测试 → 分析 → 评估 → 优化
2. **场景选择**：根据策略类型选择合适的评估场景
3. **批量处理**：利用批量评估功能提高效率
4. **结果解读**：关注维度得分和具体建议，而非仅看总分
5. **持续改进**：根据评估结果优化因子

通过合理使用评估模块，可以系统地识别高质量因子，构建稳健的量化策略。