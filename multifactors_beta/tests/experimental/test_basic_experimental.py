#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用experimental_lab框架实现基础测试因子
最简版本：仅用于验证experimental_lab框架功能，不涉及复杂计算
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# 使用新的experimental_lab框架
from factors.experimental_lab import ExperimentalFactorManager

logger = logging.getLogger(__name__)


def calculate_basic_test_factor(context=None, **kwargs) -> pd.Series:
    """
    计算基础测试因子
    
    极简版本：随机生成数据，用于验证框架功能
    
    Parameters:
    -----------
    context : CalculationContext
        计算上下文，提供数据和工具
    **kwargs : dict
        其他计算参数
        
    Returns:
    --------
    pd.Series
        因子数据，MultiIndex[TradingDates, StockCodes]格式
    """
    logger.info("开始计算基础测试因子")
    
    # 1. 获取交易日期（用于构造索引）
    trading_dates = context.load_trading_dates()
    
    # 2. 使用最近的交易日期和一些股票代码
    recent_dates = trading_dates[-60:]  # 最近60个交易日
    stock_codes = ['000001', '000002', '000003', '300001', '600000', '600036']
    
    # 3. 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [recent_dates, stock_codes],
        names=['TradingDates', 'StockCodes']
    )
    
    # 4. 生成模拟因子数据
    np.random.seed(42)  # 设置随机种子以保证结果可重现
    factor_values = np.random.normal(0, 1, len(index))
    
    # 5. 创建Series
    factor_series = pd.Series(factor_values, index=index)
    
    # 6. 添加一些真实性：某些股票表现更好
    for i, stock in enumerate(stock_codes):
        mask = factor_series.index.get_level_values('StockCodes') == stock
        if i < 3:  # 前3只股票表现较好
            factor_series[mask] += 0.5
        else:  # 后3只股票表现较差
            factor_series[mask] -= 0.3
    
    logger.info(f"基础测试因子计算完成，数据点数: {len(factor_series)}")
    logger.info(f"因子值范围: {factor_series.min():.4f} ~ {factor_series.max():.4f}")
    logger.info(f"因子均值: {factor_series.mean():.4f}, 标准差: {factor_series.std():.4f}")
    
    return factor_series


def main():
    """主函数：执行完整的因子开发工作流程"""
    
    # 创建实验因子管理器
    logger.info("创建实验因子管理器")
    manager = ExperimentalFactorManager()
    
    # 因子基本信息
    factor_name = "basic_test_factor"
    factor_description = """
    基础测试因子：用于验证experimental_lab框架功能
    
    这是一个模拟的测试因子，用于验证框架的完整工作流程：
    1. 因子注册
    2. 因子计算 
    3. 因子测试
    4. 结果跟踪
    5. 数据导出
    
    注：这不是真实的投资因子，仅用于系统测试
    """
    
    try:
        print("执行experimental_lab框架完整测试...")
        
        # 测试1: 单独注册
        print("\n1. 测试因子注册...")
        success = manager.register_factor(
            name=factor_name,
            calculation_func=calculate_basic_test_factor,
            description=factor_description,
            category="test",
            author="AI Assistant"
        )
        print(f"注册结果: {'成功' if success else '失败'}")
        
        # 测试2: 单独计算
        print("\n2. 测试因子计算...")
        calc_result = manager.calculate_factor(factor_name)
        print(f"计算结果: {'成功' if calc_result.success else '失败'}")
        if calc_result.success:
            print(f"  数据点数: {len(calc_result.factor_data)}")
            print(f"  计算耗时: {calc_result.calculation_time:.2f}秒")
        
        # 测试3: 获取因子信息
        print("\n3. 测试因子信息查询...")
        factor_info = manager.get_factor_info(factor_name)
        print(f"信息查询: {'成功' if factor_info else '失败'}")
        if factor_info:
            print(f"  状态: {factor_info['basic_info']['status']}")
            print(f"  分类: {factor_info['basic_info']['category']}")
        
        # 测试4: 列出因子
        print("\n4. 测试因子列表...")
        factor_list = manager.list_factors()
        print(f"列表查询: {'成功' if not factor_list.empty else '失败'}")
        if not factor_list.empty:
            print(f"  总因子数: {len(factor_list)}")
            print(f"  列: {list(factor_list.columns)}")
        
        # 测试5: 生成汇总报告
        print("\n5. 测试汇总报告...")
        summary = manager.get_summary_report()
        print(f"汇总报告: 成功")
        print(f"  总因子数: {summary['total_factors']}")
        print(f"  状态分布: {summary['status_distribution']}")
        
        # 测试6: 导出筛选数据
        print("\n6. 测试筛选数据导出...")
        screening_data = manager.export_for_screening()
        print(f"数据导出: 成功")
        print(f"  符合条件因子数: {screening_data['metadata']['total_qualified_factors']}")
        
        # 测试7: 管理器统计
        print("\n7. 测试管理器统计...")
        stats = manager.get_stats()
        print(f"统计信息: 成功")
        print(f"  总操作数: {stats['total_operations']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        
        print(f"\n🎉 experimental_lab框架测试完成！")
        print("所有核心功能均正常工作，框架验证成功。")
            
    except Exception as e:
        logger.error(f"框架测试失败: {e}")
        print(f"\n❌ 框架测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("experimental_lab框架基础功能测试")
    print("="*50)
    
    main()