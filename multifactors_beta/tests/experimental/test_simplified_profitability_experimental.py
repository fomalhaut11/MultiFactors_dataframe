#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用experimental_lab框架实现简化版盈利能力因子
简化版本: TTM净利润 / 总流动负债 / 5日收益率z-score

这个版本简化了原来的复杂公式，专注验证experimental_lab框架的完整工作流程
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# 使用新的experimental_lab框架
from factors.experimental_lab import ExperimentalFactorManager

# 必须使用的工具集
from factors.generators import (
    calculate_ttm,
    expand_to_daily_vectorized,
    FinancialReportProcessor
)

# 数据加载器
from factors.utils.data_loader import FactorDataLoader

logger = logging.getLogger(__name__)


def calculate_simplified_profitability_factor(context=None, **kwargs) -> pd.Series:
    """
    计算简化版盈利能力因子
    
    公式: TTM净利润 / 总流动负债 / 5日收益率截面z-score
    
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
    logger.info("开始计算简化版盈利能力因子")
    
    # 1. 获取必要的数据（使用context提供的标准接口）
    financial_data = context.load_financial_data()
    price_data = context.load_price_data()
    trading_dates = context.load_trading_dates()
    
    # 2. 获取generators工具集（严禁重复实现）
    tools = context.get_generators_tools()
    calculate_ttm_func = tools['calculate_ttm']
    expand_to_daily_func = tools['expand_to_daily_vectorized']
    
    # 3. 计算TTM净利润
    logger.info("计算TTM净利润")
    ttm_data = calculate_ttm_func(financial_data)
    ttm_profit = ttm_data.get('NET_PROFIT_IS_ttm', pd.Series())
    
    if ttm_profit.empty:
        raise ValueError("TTM净利润数据缺失，请检查NET_PROFIT_IS_ttm字段")
    
    # 4. 提取总流动负债
    logger.info("提取总流动负债数据")
    current_liabilities = financial_data.get('TOT_CUR_LIAB', pd.Series())
    
    if current_liabilities.empty:
        raise ValueError("总流动负债数据缺失，请检查TOT_CUR_LIAB字段")
    
    # 使用最新的负债数据
    latest_liabilities = current_liabilities.groupby('StockCodes').last()
    
    # 5. 计算基础因子：TTM净利润 / 总流动负债
    logger.info("计算基础因子值")
    
    # 将流动负债扩展到与TTM利润相同的索引
    expanded_liabilities = pd.Series(index=ttm_profit.index, dtype=float)
    for stock_code in ttm_profit.index.get_level_values('StockCodes').unique():
        if stock_code in latest_liabilities.index:
            mask = ttm_profit.index.get_level_values('StockCodes') == stock_code
            expanded_liabilities.loc[mask] = latest_liabilities.loc[stock_code]
    
    # 计算比率，处理除零情况
    with np.errstate(divide='ignore', invalid='ignore'):
        basic_factor = ttm_profit / expanded_liabilities.fillna(1)
    
    # 处理异常值
    basic_factor = basic_factor.replace([np.inf, -np.inf], np.nan)
    basic_factor = basic_factor.dropna()
    
    if basic_factor.empty:
        raise ValueError("基础因子计算结果为空，请检查数据质量")
    
    # 6. 计算5日收益率
    logger.info("计算5日收益率")
    
    # 确保价格数据格式正确
    if not isinstance(price_data.index, pd.MultiIndex):
        raise ValueError("价格数据必须是MultiIndex[TradingDates, StockCodes]格式")
    
    # 计算5日对数收益率（简化版本：只取一个样本）
    price_sample = price_data.iloc[:100000]  # 限制数据量以提高速度
    price_sample_sorted = price_sample.sort_index()
    returns_5d = price_sample_sorted.groupby(level=1).apply(
        lambda x: np.log(x / x.shift(5))
    ).dropna()
    
    # 7. 计算5日收益率的截面z-score
    logger.info("计算收益率截面z-score")
    
    def calculate_cross_sectional_zscore(group):
        """计算截面z-score"""
        if len(group) < 2:
            return group
        return (group - group.mean()) / (group.std() + 1e-8)
    
    returns_zscore = returns_5d.groupby(level=0).apply(calculate_cross_sectional_zscore)
    returns_zscore = returns_zscore.dropna()
    
    # 8. 使用官方日频扩展工具（简化版本）
    logger.info("扩展财务数据到日频")
    
    # 将基础因子转换为DataFrame格式
    basic_factor_df = basic_factor.to_frame('simplified_profitability_raw')
    
    # 简化的发布日期处理
    base_dates = basic_factor.index.get_level_values('TradingDates')
    release_dates = base_dates + pd.DateOffset(months=1)
    
    # 使用官方扩展工具（仅处理最近一年的数据以提高速度）
    recent_trading_dates = trading_dates[-252:]  # 最近一年交易日
    
    daily_basic_factor = expand_to_daily_func(
        factor_data=basic_factor_df,
        release_dates=release_dates,
        trading_dates=recent_trading_dates
    )
    
    # 提取Series
    daily_basic_factor = daily_basic_factor['simplified_profitability_raw']
    
    # 9. 最终计算：基础因子 / 收益率z-score
    logger.info("计算最终因子值")
    
    # 对齐两个数据（只取共同的索引）
    common_index = daily_basic_factor.index.intersection(returns_zscore.index)
    
    if len(common_index) < 100:  # 确保有足够的数据点
        logger.warning(f"共同数据点较少: {len(common_index)}，使用基础因子作为最终结果")
        final_factor = daily_basic_factor
    else:
        aligned_basic = daily_basic_factor.loc[common_index]
        aligned_returns_zscore = returns_zscore.loc[common_index]
        
        # 最终计算，处理除零
        with np.errstate(divide='ignore', invalid='ignore'):
            final_factor = aligned_basic / (aligned_returns_zscore + 1e-8)
        
        # 清理异常值
        final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
        final_factor = final_factor.dropna()
    
    # 10. 数据质量检查
    if final_factor.empty:
        raise ValueError("最终因子计算结果为空")
    
    if not isinstance(final_factor.index, pd.MultiIndex):
        raise ValueError("返回数据必须是MultiIndex格式")
    
    logger.info(f"简化版盈利能力因子计算完成，数据点数: {len(final_factor)}")
    logger.info(f"因子值范围: {final_factor.min():.4f} ~ {final_factor.max():.4f}")
    logger.info(f"因子均值: {final_factor.mean():.4f}, 标准差: {final_factor.std():.4f}")
    
    return final_factor


def main():
    """主函数：执行完整的因子开发工作流程"""
    
    # 创建实验因子管理器
    logger.info("创建实验因子管理器")
    manager = ExperimentalFactorManager()
    
    # 因子基本信息
    factor_name = "simplified_profitability_factor"
    factor_description = """
    简化版盈利能力因子：TTM净利润 / 总流动负债 / 5日收益率截面z-score
    
    经济含义：
    - 分子衡量企业的盈利能力（TTM净利润）
    - 分母衡量企业短期偿债压力（总流动负债）
    - 除以收益率z-score进行市场情绪调整
    
    预期：该因子应该能够识别具有良好盈利能力且短期偿债压力较小的股票
    
    注：这是复杂版本的简化实现，用于验证experimental_lab框架
    """
    
    try:
        # 执行完整工作流程
        logger.info(f"开始执行因子 {factor_name} 完整工作流程")
        
        workflow_result = manager.full_workflow(
            name=factor_name,
            calculation_func=calculate_simplified_profitability_factor,
            description=factor_description,
            category="profitability",
            calculation_params={},
            test_params={
                'group_nums': 5,  # 减少分组数以提高速度
                'outlier_method': 'IQR',
                'outlier_param': 3,
                'ic_decay_periods': 10  # 减少周期数
            },
            auto_decision=True
        )
        
        # 分析工作流程结果
        print("\n" + "="*60)
        print("简化版盈利能力因子开发工作流程完成")
        print("="*60)
        
        print(f"因子名称: {workflow_result['factor_name']}")
        print(f"工作流程成功: {workflow_result['success']}")
        print(f"最终状态: {workflow_result['final_status']}")
        print(f"总耗时: {workflow_result.get('total_time', 0):.2f}秒")
        
        # 展示各阶段结果
        print("\n阶段执行情况:")
        for stage_name, stage_result in workflow_result['stages'].items():
            status = "✓ 成功" if stage_result['success'] else "✗ 失败"
            print(f"  {stage_name}: {status}")
            if 'time_cost' in stage_result:
                print(f"    耗时: {stage_result['time_cost']:.2f}秒")
            if 'error_msg' in stage_result and stage_result['error_msg']:
                print(f"    错误: {stage_result['error_msg']}")
        
        # 展示性能指标
        if 'performance_metrics' in workflow_result:
            print("\n性能指标:")
            metrics = workflow_result['performance_metrics']
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # 获取因子详细信息
        factor_info = manager.get_factor_info(factor_name)
        if factor_info:
            print(f"\n因子详细信息:")
            print(f"  分类: {factor_info['basic_info']['category']}")
            print(f"  状态: {factor_info['basic_info']['status']}")
            
            if 'lifecycle' in factor_info:
                lifecycle = factor_info['lifecycle']
                print(f"  计算成功: {lifecycle['calculation_success']}")
                print(f"  测试成功: {lifecycle['test_success']}")
                print(f"  提升决策: {lifecycle['promotion_decision'] or '待定'}")
        
        # 生成汇总报告
        summary_report = manager.get_summary_report()
        print(f"\n管理器汇总:")
        print(f"  总因子数: {summary_report['total_factors']}")
        print(f"  各状态分布: {summary_report['status_distribution']}")
        
        # 导出给筛选器使用的数据
        screening_data = manager.export_for_screening(performance_threshold=0)
        print(f"\n筛选器数据导出:")
        print(f"  符合条件的因子数: {screening_data['metadata']['total_qualified_factors']}")
        
        if workflow_result['success']:
            print(f"\n🎉 简化版盈利能力因子开发成功！")
            print(f"experimental_lab框架工作流程验证完成。")
        else:
            print(f"\n❌ 因子开发过程中出现问题，请检查日志获取详细信息。")
            
    except Exception as e:
        logger.error(f"执行工作流程失败: {e}")
        print(f"\n❌ 工作流程执行失败: {e}")
        raise


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("开始使用experimental_lab框架开发简化版盈利能力因子")
    print("="*60)
    
    main()