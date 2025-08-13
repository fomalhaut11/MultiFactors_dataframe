#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core模块API使用示例
展示如何通过core模块的统一入口使用单因子测试功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 导入core模块的功能
from core import (
    test_single_factor,
    screen_factors,
    batch_test_factors,
    load_factor_data,
    get_factor_test_result,
    get_config,
    get_path
)


def example_1_test_single_factor():
    """示例1: 测试单个因子"""
    print("\n" + "="*60)
    print("示例1: 测试单个因子")
    print("="*60)
    
    # 测试BP因子
    result = test_single_factor(
        'BP',
        begin_date='2024-01-01',
        end_date='2024-06-30',
        group_nums=5
    )
    
    if result and result.ic_result:
        print(f"BP因子测试结果:")
        print(f"  IC均值: {result.ic_result.ic_mean:.4f}")
        print(f"  ICIR: {result.ic_result.icir:.4f}")
        print(f"  Rank IC: {result.ic_result.rank_ic_mean:.4f}")
        
    if result and result.group_result:
        print(f"  单调性: {result.group_result.monotonicity_score:.4f}")
        
    return result


def example_2_batch_test():
    """示例2: 批量测试多个因子"""
    print("\n" + "="*60)
    print("示例2: 批量测试多个因子")
    print("="*60)
    
    # 批量测试因子
    factor_list = ['BP', 'EP_ttm', 'SP_ttm']
    results = batch_test_factors(
        factor_list,
        begin_date='2024-01-01',
        end_date='2024-06-30'
    )
    
    # 显示结果汇总
    print("\n测试结果汇总:")
    print("-"*40)
    for factor_name, result in results.items():
        if result and result.ic_result:
            print(f"{factor_name:10s}: IC={result.ic_result.ic_mean:7.4f}, "
                  f"ICIR={result.ic_result.icir:7.4f}")
        else:
            print(f"{factor_name:10s}: 测试失败")
    
    return results


def example_3_screen_factors():
    """示例3: 筛选高质量因子"""
    print("\n" + "="*60)
    print("示例3: 筛选高质量因子")
    print("="*60)
    
    # 使用不同标准筛选
    print("\n1. 宽松标准筛选:")
    loose_factors = screen_factors(preset='loose')
    print(f"   筛选出 {len(loose_factors)} 个因子: {loose_factors}")
    
    print("\n2. 正常标准筛选:")
    normal_factors = screen_factors(preset='normal')
    print(f"   筛选出 {len(normal_factors)} 个因子: {normal_factors}")
    
    print("\n3. 自定义标准筛选:")
    custom_factors = screen_factors({
        'ic_mean_min': 0.01,
        'icir_min': 0.1,
        'monotonicity_min': 0.0
    })
    print(f"   筛选出 {len(custom_factors)} 个因子: {custom_factors}")
    
    return custom_factors


def example_4_load_factor_data():
    """示例4: 加载因子数据"""
    print("\n" + "="*60)
    print("示例4: 加载因子数据")
    print("="*60)
    
    try:
        # 加载BP因子数据
        bp_factor = load_factor_data('BP')
        
        print(f"BP因子数据:")
        print(f"  数据形状: {bp_factor.shape}")
        print(f"  数据点数: {len(bp_factor):,}")
        
        # 获取时间范围
        if hasattr(bp_factor.index, 'levels'):
            dates = bp_factor.index.get_level_values(0)
            stocks = bp_factor.index.get_level_values(1)
            print(f"  时间范围: {dates.min()} 至 {dates.max()}")
            print(f"  股票数量: {len(stocks.unique())}")
        
        # 基本统计
        print(f"\n数值分布:")
        print(f"  均值: {bp_factor.mean():.4f}")
        print(f"  中位数: {bp_factor.median():.4f}")
        print(f"  标准差: {bp_factor.std():.4f}")
        
        return bp_factor
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return None


def example_5_get_test_result():
    """示例5: 获取历史测试结果"""
    print("\n" + "="*60)
    print("示例5: 获取历史测试结果")
    print("="*60)
    
    # 获取BP因子的最新测试结果
    result = get_factor_test_result('BP')
    
    if result:
        print(f"BP因子历史测试结果:")
        print(f"  测试时间: {result.test_time}")
        print(f"  测试ID: {result.test_id}")
        
        if result.config_snapshot:
            config = result.config_snapshot
            print(f"  测试期间: {config.get('begin_date')} 至 {config.get('end_date')}")
        
        if result.ic_result:
            print(f"  IC均值: {result.ic_result.ic_mean:.4f}")
            print(f"  ICIR: {result.ic_result.icir:.4f}")
    else:
        print("未找到BP因子的测试结果")
    
    return result


def example_6_config_usage():
    """示例6: 使用配置管理"""
    print("\n" + "="*60)
    print("示例6: 配置管理")
    print("="*60)
    
    # 获取配置
    print("数据路径配置:")
    print(f"  数据根目录: {get_path('data_root')}")
    print(f"  因子目录: {get_path('raw_factors')}")
    print(f"  测试结果目录: {get_path('single_factor_test')}")
    
    # 获取特定配置
    db_config = get_config('database')
    if db_config:
        print(f"\n数据库配置:")
        print(f"  数据库: {db_config.get('database', 'N/A')}")
    
    return True


def main():
    """主函数：运行所有示例"""
    print("\n" + "="*80)
    print("                 Core模块API使用示例")
    print("="*80)
    
    # 示例1: 测试单个因子
    # result = example_1_test_single_factor()
    
    # 示例2: 批量测试
    # results = example_2_batch_test()
    
    # 示例3: 筛选因子
    # factors = example_3_screen_factors()
    
    # 示例4: 加载因子数据
    bp_data = example_4_load_factor_data()
    
    # 示例5: 获取测试结果
    test_result = example_5_get_test_result()
    
    # 示例6: 配置管理
    example_6_config_usage()
    
    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)
    
    print("\n提示:")
    print("1. 可以通过 'from core import test_single_factor' 直接导入功能")
    print("2. 所有因子测试功能都可以通过core模块访问")
    print("3. 配置管理也集成在core模块中")


if __name__ == "__main__":
    main()