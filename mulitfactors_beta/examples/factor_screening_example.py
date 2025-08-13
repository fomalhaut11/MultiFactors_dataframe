#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子筛选分析示例

演示如何使用FactorScreener进行因子筛选和分析
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from factors.analyzer import FactorScreener
from factors.tester import SingleFactorTestPipeline
import pandas as pd


def example_1_basic_screening():
    """
    示例1：基础因子筛选
    """
    print("\n" + "="*60)
    print("示例1：基础因子筛选")
    print("="*60)
    
    # 创建筛选器
    screener = FactorScreener()
    
    # 加载所有测试结果
    print("\n1. 加载历史测试结果...")
    results = screener.load_all_results()
    print(f"   加载了 {len(results)} 个因子的测试结果")
    
    # 使用默认标准筛选
    print("\n2. 使用默认标准筛选因子...")
    selected_factors = screener.screen_factors()
    print(f"   筛选出 {len(selected_factors)} 个优质因子:")
    for factor in selected_factors[:10]:
        print(f"   - {factor}")
    
    # 获取因子排名
    print("\n3. 获取因子排名（按ICIR）...")
    ranking_df = screener.get_factor_ranking(metric='icir', top_n=10)
    print(ranking_df.to_string())
    
    return selected_factors


def example_2_custom_criteria():
    """
    示例2：自定义筛选标准
    """
    print("\n" + "="*60)
    print("示例2：自定义筛选标准")
    print("="*60)
    
    # 创建筛选器
    screener = FactorScreener()
    
    # 定义严格的筛选标准
    strict_criteria = {
        'ic_mean_min': 0.03,      # IC均值 > 0.03
        'icir_min': 0.7,           # ICIR > 0.7
        'monotonicity_min': 0.7,   # 单调性 > 0.7
        'sharpe_min': 1.5,         # 夏普比率 > 1.5
        't_value_min': 2.5         # t值 > 2.5
    }
    
    print("\n严格筛选标准:")
    for key, value in strict_criteria.items():
        print(f"  {key}: {value}")
    
    # 筛选因子
    selected_factors = screener.screen_factors(criteria=strict_criteria)
    print(f"\n筛选结果: {len(selected_factors)} 个因子通过严格筛选")
    
    # 查看筛选详情
    if hasattr(screener, 'screening_details'):
        print("\n筛选详情:")
        print(screener.screening_details[screener.screening_details['selected']].to_string())
    
    return selected_factors


def example_3_stability_analysis():
    """
    示例3：因子稳定性分析
    """
    print("\n" + "="*60)
    print("示例3：因子稳定性分析")
    print("="*60)
    
    # 创建筛选器
    screener = FactorScreener()
    screener.load_all_results()
    
    # 分析特定因子的稳定性
    factor_name = 'BP'
    print(f"\n分析 {factor_name} 因子的稳定性...")
    
    stability = screener.analyze_factor_stability(factor_name, lookback_days=30)
    
    if 'error' not in stability:
        print(f"\n稳定性分析结果:")
        print(f"  测试次数: {stability.get('test_count', 0)}")
        print(f"  时间范围: {stability.get('date_range', ['N/A', 'N/A'])[0]} 至 {stability.get('date_range', ['N/A', 'N/A'])[1]}")
        print(f"  IC均值: {stability.get('ic_mean', 0):.4f}")
        print(f"  IC标准差: {stability.get('ic_std', 0):.4f}")
        print(f"  IC稳定性: {stability.get('ic_stability', 0):.2f}")
        print(f"  稳定性等级: {stability.get('stability_grade', 'Unknown')}")
        
        # 显示历史IC值
        if stability.get('historical_ic'):
            print(f"\n  历史IC值:")
            for date, ic in zip(stability['historical_dates'][-5:], stability['historical_ic'][-5:]):
                print(f"    {date}: {ic:.4f}")
    else:
        print(f"  错误: {stability['error']}")
    
    return stability


def example_4_factor_comparison():
    """
    示例4：多因子比较
    """
    print("\n" + "="*60)
    print("示例4：多因子比较")
    print("="*60)
    
    # 创建筛选器
    screener = FactorScreener()
    screener.load_all_results()
    
    # 比较多个因子
    factors_to_compare = ['BP', 'EP', 'ROE']
    print(f"\n比较因子: {factors_to_compare}")
    
    comparison_df = screener.compare_factors(factors_to_compare)
    
    if not comparison_df.empty:
        print("\n比较结果:")
        print(comparison_df.to_string())
        
        # 找出最佳因子
        best_factor = comparison_df.iloc[0]
        print(f"\n最佳因子: {best_factor['factor_name']}")
        print(f"  综合评分: {best_factor['score']:.2f}")
        print(f"  类别: {best_factor.get('category', 'Unknown')}")
    else:
        print("无比较结果")
    
    return comparison_df


def example_5_generate_report():
    """
    示例5：生成筛选报告
    """
    print("\n" + "="*60)
    print("示例5：生成筛选报告")
    print("="*60)
    
    # 创建筛选器
    screener = FactorScreener()
    
    # 生成报告
    print("\n生成因子筛选报告...")
    report_df = screener.generate_screening_report(
        output_path='factor_screening_report.csv',
        top_n=20
    )
    
    print(f"\n报告已生成，包含 {len(report_df)} 个因子")
    print("\nTop 5 因子:")
    print(report_df.head().to_string())
    
    return report_df


def example_6_preset_screening():
    """
    示例6：使用预设筛选标准
    """
    print("\n" + "="*60)
    print("示例6：使用预设筛选标准")
    print("="*60)
    
    # 创建筛选器
    screener = FactorScreener()
    screener.load_all_results()
    
    # 测试不同预设
    presets = ['strict', 'normal', 'loose']
    
    for preset in presets:
        print(f"\n使用 '{preset}' 预设标准筛选...")
        selected = screener.screen_factors(preset=preset)
        print(f"  筛选出 {len(selected)} 个因子")
        
        # 显示前5个
        if selected:
            print(f"  前5个因子: {selected[:5]}")


def example_7_batch_test_and_screen():
    """
    示例7：批量测试并筛选（完整流程）
    """
    print("\n" + "="*60)
    print("示例7：批量测试并筛选（完整流程）")
    print("="*60)
    
    # 步骤1：批量测试因子
    print("\n步骤1：批量测试因子...")
    pipeline = SingleFactorTestPipeline()
    
    # 定义要测试的因子列表
    factor_list = ['BP', 'EP', 'ROE']  # 实际应用中可以更多
    
    print(f"测试因子: {factor_list}")
    for factor in factor_list:
        try:
            print(f"  测试 {factor}...")
            result = pipeline.run(
                factor, 
                save_result=True,  # 保存结果到磁盘
                begin_date='2024-01-01',
                end_date='2024-06-30'
            )
            if result.ic_result:
                print(f"    IC: {result.ic_result.ic_mean:.4f}, ICIR: {result.ic_result.icir:.4f}")
        except Exception as e:
            print(f"    测试失败: {e}")
    
    # 步骤2：筛选因子
    print("\n步骤2：筛选高质量因子...")
    screener = FactorScreener()
    screener.load_all_results(force_reload=True)  # 强制重新加载
    
    # 筛选
    selected = screener.screen_factors({
        'ic_mean_min': 0.01,
        'icir_min': 0.3
    })
    
    print(f"\n筛选结果: {len(selected)} 个因子通过筛选")
    for factor in selected:
        score = screener.get_factor_score(factor)
        print(f"  {factor}: 评分 {score:.2f}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("因子筛选分析示例程序")
    print("="*60)
    
    # 运行各个示例
    examples = [
        ("基础筛选", example_1_basic_screening),
        ("自定义标准", example_2_custom_criteria),
        ("稳定性分析", example_3_stability_analysis),
        ("因子比较", example_4_factor_comparison),
        ("生成报告", example_5_generate_report),
        ("预设标准", example_6_preset_screening),
        # ("完整流程", example_7_batch_test_and_screen)  # 这个会实际运行测试，比较耗时
    ]
    
    for name, func in examples:
        try:
            print(f"\n运行: {name}")
            func()
        except Exception as e:
            print(f"示例 {name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()