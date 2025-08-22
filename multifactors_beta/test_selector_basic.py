"""
测试Selector模块基础功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from factors.selector import (
    FactorSelector,
    FactorPool,
    PerformanceFilter,
    CorrelationFilter,
    StabilityFilter,
    TopNSelector
)


def create_test_data():
    """创建测试数据"""
    # 生成测试日期和股票
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    stocks = [f'stock_{i:03d}' for i in range(100)]
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
    
    factors = {}
    evaluation_results = {}
    
    # 生成5个测试因子
    np.random.seed(42)
    
    for i in range(5):
        factor_name = f'test_factor_{i+1}'
        
        # 生成因子数据
        n_obs = len(index)
        factor_values = np.random.randn(n_obs) * 0.5 + i * 0.1
        factor_data = pd.Series(factor_values, index=index, name=factor_name)
        
        # 添加一些缺失值
        mask = np.random.random(n_obs) < 0.05  # 5%缺失率
        factor_data.loc[mask] = np.nan
        
        factors[factor_name] = factor_data
        
        # 模拟评估结果
        class MockEvaluationResult:
            def __init__(self, factor_id):
                self.total_score = 60 + factor_id * 10 + np.random.random() * 20
                self.metrics = {
                    'ic_mean': 0.02 + factor_id * 0.01 + (np.random.random() - 0.5) * 0.02,
                    'icir': 0.3 + factor_id * 0.1 + (np.random.random() - 0.5) * 0.2
                }
                # 模拟稳定性结果
                self.stability_result = type('obj', (object,), {
                    'stability_score': 40 + factor_id * 15 + np.random.random() * 30,
                    'ic_volatility': 0.2 + np.random.random() * 0.3
                })()
        
        evaluation_results[factor_name] = MockEvaluationResult(i)
    
    return factors, evaluation_results


def test_factor_pool():
    """测试FactorPool"""
    print("=== 测试FactorPool ===")
    
    factors, evaluation_results = create_test_data()
    
    # 创建因子池
    pool = FactorPool()
    
    # 添加因子
    for name, factor_data in factors.items():
        success = pool.add_factor(
            name=name,
            factor_data=factor_data,
            category='test',
            description=f'Test factor {name}'
        )
        print(f"添加因子 {name}: {'成功' if success else '失败'}")
    
    # 获取统计信息
    stats = pool.get_statistics()
    print(f"因子池统计: {stats['total_factors']} 个因子")
    print(f"类别: {list(stats['categories'].keys())}")
    
    # 搜索因子
    searched = pool.search_factors(keyword='factor_1')
    print(f"搜索 'factor_1': {searched}")
    
    print("FactorPool测试完成\n")


def test_performance_filter():
    """测试性能筛选器"""
    print("=== 测试PerformanceFilter ===")
    
    factors, evaluation_results = create_test_data()
    
    # 创建性能筛选器
    perf_filter = PerformanceFilter(
        min_score=70.0,
        min_ic=0.02,
        min_ir=0.3
    )
    
    # 应用筛选
    filtered = perf_filter.filter(factors, evaluation_results)
    
    print(f"原始因子数: {len(factors)}")
    print(f"筛选后因子数: {len(filtered)}")
    print(f"筛选后因子: {list(filtered.keys())}")
    
    # 获取筛选摘要
    summary = perf_filter.get_filter_summary()
    print(f"筛选摘要: {summary}")
    
    print("PerformanceFilter测试完成\n")


def test_correlation_filter():
    """测试相关性筛选器"""
    print("=== 测试CorrelationFilter ===")
    
    factors, evaluation_results = create_test_data()
    
    # 创建相关性筛选器
    corr_filter = CorrelationFilter(
        max_correlation=0.7,
        method='hierarchical'
    )
    
    # 应用筛选
    filtered = corr_filter.filter(factors, evaluation_results)
    
    print(f"原始因子数: {len(factors)}")
    print(f"筛选后因子数: {len(filtered)}")
    print(f"筛选后因子: {list(filtered.keys())}")
    
    print("CorrelationFilter测试完成\n")


def test_top_n_selector():
    """测试TopN选择器"""
    print("=== 测试TopNSelector ===")
    
    factors, evaluation_results = create_test_data()
    
    # 创建TopN选择器
    selector = TopNSelector(
        n_factors=3,
        score_metric='total_score',
        tie_breaker='ic_mean'
    )
    
    # 进行选择
    result = selector.select(factors, evaluation_results)
    
    print(f"选中的因子: {result['selected_factors']}")
    print(f"选择得分: {result['selection_scores']}")
    print(f"选择摘要: {result['summary']}")
    
    print("TopNSelector测试完成\n")


def test_factor_selector():
    """测试主要的FactorSelector"""
    print("=== 测试FactorSelector ===")
    
    factors, evaluation_results = create_test_data()
    
    # 创建因子选择器
    selector = FactorSelector(
        method='top_n',
        config={
            'n_factors': 3,
            'score_metric': 'total_score',
            'filters': {
                'use_performance_filter': True,
                'performance': {
                    'min_score': 60.0,
                    'min_ic': 0.01
                }
            }
        }
    )
    
    # 进行选择
    result = selector.select(factors, evaluation_results)
    
    print(f"选中的因子: {result['selected_factors']}")
    print(f"筛选摘要: {result['filtering_summary']}")
    print(f"性能统计: {result['performance_stats']}")
    print(f"执行信息: {result['execution_info']['duration_seconds']:.4f}秒")
    
    print("FactorSelector测试完成\n")


def main():
    """运行所有测试"""
    print("开始测试Selector模块\n")
    
    try:
        test_factor_pool()
        test_performance_filter()
        test_correlation_filter()
        test_top_n_selector()
        test_factor_selector()
        
        print("所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()