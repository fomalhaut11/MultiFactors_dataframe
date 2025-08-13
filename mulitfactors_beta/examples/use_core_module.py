#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用core模块统一入口的示例
展示如何通过core模块快速访问单因子测试等核心功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 导入core模块的便捷函数
from core import (
    test_single_factor,
    screen_factors,
    batch_test_factors,
    load_factor_data,
    get_factor_test_result,
    get_config,
    get_path
)


def example_1_quick_test():
    """示例1: 快速测试单个因子"""
    print("\n" + "="*60)
    print("示例1: 快速测试单个因子")
    print("="*60)
    
    # 一行代码完成因子测试
    result = test_single_factor(
        'BP',
        begin_date='2024-01-01',
        end_date='2024-06-30',
        group_nums=5
    )
    
    # 查看结果
    if result and result.ic_result:
        print(f"BP因子测试结果:")
        print(f"  IC均值: {result.ic_result.ic_mean:.4f}")
        print(f"  ICIR: {result.ic_result.icir:.4f}")
        print(f"  夏普比率: {result.performance_metrics.get('long_short_sharpe', 0):.4f}")


def example_2_batch_test():
    """示例2: 批量测试多个因子"""
    print("\n" + "="*60)
    print("示例2: 批量测试多个因子")
    print("="*60)
    
    # 批量测试
    factor_list = ['BP', 'EP_ttm', 'SP_ttm']
    results = batch_test_factors(
        factor_list,
        begin_date='2024-01-01',
        end_date='2024-06-30'
    )
    
    # 汇总结果
    print("\n测试结果汇总:")
    for factor, result in results.items():
        if result and result.ic_result:
            print(f"  {factor:10s}: IC={result.ic_result.ic_mean:7.4f}, ICIR={result.ic_result.icir:7.4f}")
        else:
            print(f"  {factor:10s}: 测试失败")


def example_3_screen_factors():
    """示例3: 筛选高质量因子"""
    print("\n" + "="*60)
    print("示例3: 筛选高质量因子")
    print("="*60)
    
    # 使用预设标准筛选
    print("\n使用正常标准筛选:")
    normal_factors = screen_factors(preset='normal')
    print(f"筛选出 {len(normal_factors)} 个因子: {normal_factors}")
    
    # 使用自定义标准筛选
    print("\n使用自定义标准筛选:")
    custom_factors = screen_factors(criteria={
        'ic_mean_min': 0.01,
        'icir_min': 0.1,
        'monotonicity_min': 0.0
    })
    print(f"筛选出 {len(custom_factors)} 个因子: {custom_factors}")


def example_4_load_factor():
    """示例4: 加载因子数据"""
    print("\n" + "="*60)
    print("示例4: 加载因子数据")
    print("="*60)
    
    try:
        # 加载BP因子
        bp_factor = load_factor_data('BP')
        print(f"BP因子数据:")
        print(f"  形状: {bp_factor.shape}")
        print(f"  均值: {bp_factor.mean():.4f}")
        print(f"  标准差: {bp_factor.std():.4f}")
        print(f"  非空值率: {bp_factor.notna().mean():.2%}")
    except FileNotFoundError as e:
        print(f"错误: {e}")


def example_5_get_test_result():
    """示例5: 获取历史测试结果"""
    print("\n" + "="*60)
    print("示例5: 获取历史测试结果")
    print("="*60)
    
    # 获取BP因子的最新测试结果
    result = get_factor_test_result('BP')
    
    if result:
        print(f"BP因子最新测试结果:")
        print(f"  测试时间: {result.test_time}")
        print(f"  测试期间: {result.config_snapshot.get('begin_date')} 至 {result.config_snapshot.get('end_date')}")
        print(f"  IC均值: {result.ic_result.ic_mean:.4f}")
        print(f"  ICIR: {result.ic_result.icir:.4f}")
    else:
        print("未找到BP因子的测试结果")


def example_6_config_usage():
    """示例6: 使用配置管理"""
    print("\n" + "="*60)
    print("示例6: 使用配置管理")
    print("="*60)
    
    # 获取配置
    db_config = get_config('database')
    if db_config:
        print("数据库配置:")
        print(f"  数据库: {db_config.get('database', 'N/A')}")
    
    # 获取路径
    print("\n数据路径:")
    print(f"  因子路径: {get_path('raw_factors')}")
    print(f"  测试结果路径: {get_path('single_factor_test')}")
    print(f"  数据根目录: {get_path('data_root')}")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("Core模块使用示例")
    print("="*80)
    
    # 运行各个示例
    try:
        example_1_quick_test()
    except Exception as e:
        print(f"示例1执行失败: {e}")
    
    try:
        example_2_batch_test()
    except Exception as e:
        print(f"示例2执行失败: {e}")
    
    try:
        example_3_screen_factors()
    except Exception as e:
        print(f"示例3执行失败: {e}")
    
    try:
        example_4_load_factor()
    except Exception as e:
        print(f"示例4执行失败: {e}")
    
    try:
        example_5_get_test_result()
    except Exception as e:
        print(f"示例5执行失败: {e}")
    
    try:
        example_6_config_usage()
    except Exception as e:
        print(f"示例6执行失败: {e}")
    
    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)


if __name__ == "__main__":
    main()