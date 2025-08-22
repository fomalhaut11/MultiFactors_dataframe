"""
测试Selector模块的所有选择策略
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
    TopNSelector,
    ThresholdSelector,
    ClusteringSelector,
    PerformanceFilter,
    CompositeFilter
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
    
    # 生成8个测试因子
    np.random.seed(42)
    
    for i in range(8):
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
                self.total_score = 50 + factor_id * 8 + np.random.random() * 25
                self.metrics = {
                    'ic_mean': 0.015 + factor_id * 0.008 + (np.random.random() - 0.5) * 0.02,
                    'icir': 0.25 + factor_id * 0.08 + (np.random.random() - 0.5) * 0.2
                }
                # 模拟稳定性结果
                self.stability_result = type('obj', (object,), {
                    'stability_score': 35 + factor_id * 12 + np.random.random() * 30,
                    'ic_volatility': 0.15 + np.random.random() * 0.25
                })()
        
        evaluation_results[factor_name] = MockEvaluationResult(i)
    
    return factors, evaluation_results


def test_threshold_selector():
    """测试阈值选择器"""
    print("=== 测试ThresholdSelector ===")
    
    factors, evaluation_results = create_test_data()
    
    # 测试AND逻辑
    selector_and = ThresholdSelector(
        thresholds={
            'total_score': 70.0,
            'ic_mean': 0.02,
            'icir': 0.3
        },
        logic='AND'
    )
    
    result_and = selector_and.select(factors, evaluation_results)
    print(f"AND逻辑选择结果: {len(result_and['selected_factors'])} 个因子")
    print(f"选中因子: {result_and['selected_factors']}")
    
    # 测试OR逻辑
    selector_or = ThresholdSelector(
        thresholds={
            'total_score': 85.0,
            'ic_mean': 0.04,
            'icir': 0.5
        },
        logic='OR'
    )
    
    result_or = selector_or.select(factors, evaluation_results)
    print(f"OR逻辑选择结果: {len(result_or['selected_factors'])} 个因子")
    print(f"选中因子: {result_or['selected_factors']}")
    
    print("ThresholdSelector测试完成\n")


def test_clustering_selector():
    """测试聚类选择器"""
    print("=== 测试ClusteringSelector ===")
    
    factors, evaluation_results = create_test_data()
    
    # 测试KMeans聚类
    selector_kmeans = ClusteringSelector(
        n_clusters=3,
        clustering_method='kmeans',
        feature_selection='performance',
        factors_per_cluster=1
    )
    
    result_kmeans = selector_kmeans.select(factors, evaluation_results)
    print(f"KMeans聚类选择结果: {len(result_kmeans['selected_factors'])} 个因子")
    print(f"选中因子: {result_kmeans['selected_factors']}")
    print(f"聚类数量: {result_kmeans['summary']['clusters_formed']}")
    
    # 测试层次聚类
    selector_hier = ClusteringSelector(
        n_clusters=4,
        clustering_method='hierarchical',
        feature_selection='mixed',
        factors_per_cluster=2
    )
    
    result_hier = selector_hier.select(factors, evaluation_results)
    print(f"层次聚类选择结果: {len(result_hier['selected_factors'])} 个因子")
    print(f"选中因子: {result_hier['selected_factors']}")
    print(f"聚类数量: {result_hier['summary']['clusters_formed']}")
    
    print("ClusteringSelector测试完成\n")


def test_composite_filter():
    """测试复合筛选器"""
    print("=== 测试CompositeFilter ===")
    
    factors, evaluation_results = create_test_data()
    
    # 创建多个筛选器
    perf_filter = PerformanceFilter(
        min_score=60.0,
        min_ic=0.01
    )
    
    # 创建复合筛选器（AND逻辑）
    composite_and = CompositeFilter(
        filters=[perf_filter],
        logic='AND'
    )
    
    filtered_and = composite_and.filter(factors, evaluation_results)
    print(f"AND复合筛选结果: {len(filtered_and)} 个因子")
    
    # 获取详细结果
    detailed = composite_and.get_detailed_results()
    print(f"筛选详情: {detailed}")
    
    print("CompositeFilter测试完成\n")


def test_different_methods():
    """测试不同选择方法的FactorSelector"""
    print("=== 测试不同选择方法 ===")
    
    factors, evaluation_results = create_test_data()
    
    # TopN方法
    selector_topn = FactorSelector(
        method='top_n',
        config={
            'n_factors': 3,
            'score_metric': 'total_score'
        }
    )
    
    result_topn = selector_topn.select(factors, evaluation_results)
    print(f"TopN方法: {result_topn['selected_factors']}")
    
    print("不同选择方法测试完成\n")


def test_performance_comparison():
    """测试不同方法的性能对比"""
    print("=== 性能对比测试 ===")
    
    factors, evaluation_results = create_test_data()
    
    methods = [
        ('TopN', TopNSelector(n_factors=3)),
        ('Threshold', ThresholdSelector({'total_score': 70}, 'AND')),
        ('Clustering', ClusteringSelector(n_clusters=3, factors_per_cluster=1))
    ]
    
    results = {}
    
    for method_name, selector in methods:
        start_time = datetime.now()
        result = selector.select(factors, evaluation_results)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        results[method_name] = {
            'selected_count': len(result['selected_factors']),
            'duration': duration,
            'selected_factors': result['selected_factors']
        }
        
        print(f"{method_name}: {results[method_name]['selected_count']} 因子, "
              f"{duration:.4f}秒")
    
    print("\n性能对比完成\n")


def test_edge_cases():
    """测试边界情况"""
    print("=== 边界情况测试 ===")
    
    # 空因子池
    try:
        selector = TopNSelector(n_factors=3)
        result = selector.select({}, {})
        print(f"空因子池测试: {len(result['selected_factors'])} 个因子")
    except Exception as e:
        print(f"空因子池测试失败: {e}")
    
    # 只有一个因子
    factors, evaluation_results = create_test_data()
    single_factor = {'test_factor_1': list(factors.values())[0]}
    single_eval = {'test_factor_1': list(evaluation_results.values())[0]}
    
    try:
        selector = ClusteringSelector(n_clusters=2)
        result = selector.select(single_factor, single_eval)
        print(f"单因子聚类测试: {len(result['selected_factors'])} 个因子")
    except Exception as e:
        print(f"单因子聚类测试失败: {e}")
    
    print("边界情况测试完成\n")


def main():
    """运行所有测试"""
    print("开始测试Selector模块的所有策略\n")
    
    try:
        test_threshold_selector()
        test_clustering_selector()
        test_composite_filter()
        test_different_methods()
        test_performance_comparison()
        test_edge_cases()
        
        print("所有策略测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()