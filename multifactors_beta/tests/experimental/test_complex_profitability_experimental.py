#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用experimental_lab框架实现复杂盈利能力因子
{(TTM利润-TTM财务费用)-单季度存货}/短期债务 / 5日收益率截面z-score

严格遵循项目约束：
1. 必须使用factors.generators工具集
2. 返回MultiIndex[TradingDates, StockCodes]格式
3. 使用experimental_lab完整工作流程
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
    calculate_single_quarter,
    expand_to_daily_vectorized,
    FinancialReportProcessor
)

# 数据加载器
from factors.utils.data_loader import FactorDataLoader

logger = logging.getLogger(__name__)


def calculate_complex_profitability_factor(context=None, **kwargs) -> pd.Series:
    """
    计算复杂盈利能力因子
    
    公式: {(TTM利润-TTM财务费用)-单季度存货}/短期债务 / 5日收益率截面z-score
    
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
    logger.info("开始计算复杂盈利能力因子")
    
    # 1. 获取必要的数据（使用context提供的标准接口）
    financial_data = context.load_financial_data()
    price_data = context.load_price_data()
    trading_dates = context.load_trading_dates()
    
    # 2. 获取generators工具集（严禁重复实现）
    tools = context.get_generators_tools()
    calculate_ttm_func = tools['calculate_ttm']
    calculate_single_quarter_func = tools['calculate_single_quarter']
    expand_to_daily_func = tools['expand_to_daily_vectorized']
    
    # 3. 计算财务组件
    logger.info("计算TTM财务指标")
    
    # 使用官方TTM计算工具
    ttm_data = calculate_ttm_func(financial_data)
    
    # 提取需要的TTM指标（使用实际的字段名）
    ttm_profit = ttm_data.get('NET_PROFIT_IS_ttm', pd.Series())
    ttm_financial_expense = ttm_data.get('FIN_EXP_IS_ttm', pd.Series())
    
    if ttm_profit.empty:
        raise ValueError("TTM净利润数据缺失，请检查NET_PROFIT_IS_ttm字段")
    
    if ttm_financial_expense.empty:
        logger.warning("TTM财务费用数据缺失，将使用0填充")
        ttm_financial_expense = pd.Series(0, index=ttm_profit.index)
    
    # 计算调整后的TTM利润
    ttm_adjusted_profit = ttm_profit - ttm_financial_expense
    
    # 使用官方单季度计算工具
    logger.info("计算单季度存货")
    single_quarter_data = calculate_single_quarter_func(financial_data)
    single_quarter_inventory = single_quarter_data.get('INVENTORIES', pd.Series())
    
    if single_quarter_inventory.empty:
        logger.warning("单季度存货数据缺失，将使用0填充")
        single_quarter_inventory = pd.Series(0, index=ttm_adjusted_profit.index)
    
    # 4. 计算分子：(TTM调整利润 - 单季度存货)
    logger.info("计算因子分子")
    
    # 确保数据对齐
    common_index = ttm_adjusted_profit.index.intersection(single_quarter_inventory.index)
    if common_index.empty:
        raise ValueError("TTM数据与单季度数据无法对齐，请检查数据一致性")
    
    numerator = (ttm_adjusted_profit.loc[common_index] - 
                single_quarter_inventory.loc[common_index])
    
    # 5. 计算分母：短期债务（使用实际字段名）
    logger.info("提取短期债务数据")
    short_term_debt = financial_data.get('TOT_CUR_LIAB', 
                                        financial_data.get('ST_BORROW', pd.Series()))
    
    if short_term_debt.empty:
        raise ValueError("短期债务数据缺失，请检查财务数据中的TOT_CUR_LIAB或ST_BORROW字段")
    
    # 使用最新的短期债务数据
    latest_debt = short_term_debt.groupby('StockCodes').last()
    
    # 6. 计算基础因子值
    logger.info("计算基础因子值")
    
    # 将短期债务扩展到与分子相同的索引
    expanded_debt = pd.Series(index=numerator.index, dtype=float)
    for stock_code in numerator.index.get_level_values('StockCodes').unique():
        if stock_code in latest_debt.index:
            mask = numerator.index.get_level_values('StockCodes') == stock_code
            expanded_debt.loc[mask] = latest_debt.loc[stock_code]
    
    # 计算比率，处理除零情况
    with np.errstate(divide='ignore', invalid='ignore'):
        basic_factor = numerator / expanded_debt.fillna(1)
    
    # 处理异常值
    basic_factor = basic_factor.replace([np.inf, -np.inf], np.nan)
    basic_factor = basic_factor.dropna()
    
    if basic_factor.empty:
        raise ValueError("基础因子计算结果为空，请检查数据质量")
    
    # 7. 计算5日收益率
    logger.info("计算5日收益率")
    
    # 确保价格数据格式正确
    if not isinstance(price_data.index, pd.MultiIndex):
        raise ValueError("价格数据必须是MultiIndex[TradingDates, StockCodes]格式")
    
    # 计算5日对数收益率
    price_data_sorted = price_data.sort_index()
    returns_5d = price_data_sorted.groupby(level=1).apply(
        lambda x: np.log(x / x.shift(5))
    ).dropna()
    
    # 8. 计算5日收益率的截面z-score
    logger.info("计算收益率截面z-score")
    
    def calculate_cross_sectional_zscore(group):
        """计算截面z-score"""
        return (group - group.mean()) / group.std()
    
    returns_zscore = returns_5d.groupby(level=0).apply(calculate_cross_sectional_zscore)
    returns_zscore = returns_zscore.dropna()
    
    # 9. 使用官方日频扩展工具
    logger.info("扩展财务数据到日频")
    
    # 准备财务数据发布日期（简化处理，实际应该从配置中获取）
    # 这里假设财务数据在每个季度结束后1个月发布
    financial_calendar = basic_factor.index.get_level_values('TradingDates').to_series()
    release_dates = financial_calendar + pd.DateOffset(months=1)
    
    # 将基础因子转换为DataFrame格式（expand_to_daily_vectorized需要）
    basic_factor_df = basic_factor.to_frame('complex_profitability_raw')
    
    # 使用官方扩展工具
    daily_basic_factor = expand_to_daily_func(
        factor_data=basic_factor_df,
        release_dates=release_dates,
        trading_dates=trading_dates
    )
    
    # 提取Series
    daily_basic_factor = daily_basic_factor['complex_profitability_raw']
    
    # 10. 最终计算：基础因子 / 收益率z-score
    logger.info("计算最终因子值")
    
    # 对齐两个数据
    common_index = daily_basic_factor.index.intersection(returns_zscore.index)
    if common_index.empty:
        raise ValueError("财务因子与市场数据无法对齐")
    
    aligned_basic = daily_basic_factor.loc[common_index]
    aligned_returns_zscore = returns_zscore.loc[common_index]
    
    # 最终计算，处理除零
    with np.errstate(divide='ignore', invalid='ignore'):
        final_factor = aligned_basic / (aligned_returns_zscore + 1e-8)  # 加小数避免除零
    
    # 清理异常值
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = final_factor.dropna()
    
    # 11. 数据质量检查
    if final_factor.empty:
        raise ValueError("最终因子计算结果为空")
    
    if not isinstance(final_factor.index, pd.MultiIndex):
        raise ValueError("返回数据必须是MultiIndex格式")
    
    if final_factor.index.names != ['TradingDates', 'StockCodes']:
        logger.warning(f"索引名称不标准: {final_factor.index.names}")
    
    logger.info(f"复杂盈利能力因子计算完成，数据点数: {len(final_factor)}")
    logger.info(f"因子值范围: {final_factor.min():.4f} ~ {final_factor.max():.4f}")
    logger.info(f"因子均值: {final_factor.mean():.4f}, 标准差: {final_factor.std():.4f}")
    
    return final_factor


def main():
    """主函数：执行完整的因子开发工作流程"""
    
    # 创建实验因子管理器
    logger.info("创建实验因子管理器")
    manager = ExperimentalFactorManager()
    
    # 因子基本信息
    factor_name = "complex_profitability_factor"
    factor_description = """
    复杂盈利能力因子：{(TTM利润-TTM财务费用)-单季度存货}/短期债务 / 5日收益率截面z-score
    
    经济含义：
    - 分子衡量企业真实的盈利能力，减去财务成本和存货变动的影响
    - 分母考虑企业短期偿债压力
    - 除以收益率z-score进行市场情绪调整
    
    预期：该因子应该能够识别具有真实盈利能力且不受短期市场波动影响的股票
    """
    
    try:
        # 执行完整工作流程
        logger.info(f"开始执行因子 {factor_name} 完整工作流程")
        
        workflow_result = manager.full_workflow(
            name=factor_name,
            calculation_func=calculate_complex_profitability_factor,
            description=factor_description,
            category="profitability",
            calculation_params={},
            test_params={
                'group_nums': 10,
                'outlier_method': 'IQR',
                'outlier_param': 3,
                'ic_decay_periods': 20
            },
            auto_decision=True
        )
        
        # 分析工作流程结果
        print("\n" + "="*60)
        print("复杂盈利能力因子开发工作流程完成")
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
            print(f"  创建时间: {factor_info['timestamps']['created_time']}")
            
            if 'lifecycle' in factor_info:
                lifecycle = factor_info['lifecycle']
                print(f"  计算成功: {lifecycle['calculation_success']}")
                print(f"  测试成功: {lifecycle['test_success']}")
                print(f"  提升决策: {lifecycle['promotion_decision'] or '待定'}")
        
        # 生成汇总报告
        summary_report = manager.get_summary_report()
        print(f"\n管理器汇总:")
        print(f"  总因子数: {summary_report['total_factors']}")
        print(f"  操作成功率: {summary_report['manager_stats']['successful_operations']}/{summary_report['manager_stats']['total_operations']}")
        
        # 导出给筛选器使用的数据
        screening_data = manager.export_for_screening(performance_threshold=0)
        print(f"\n筛选器数据导出:")
        print(f"  符合条件的因子数: {screening_data['metadata']['total_qualified_factors']}")
        
        if workflow_result['success']:
            print(f"\n🎉 复杂盈利能力因子开发成功！")
            print(f"因子已保存到experimental_lab模块中，可用于后续筛选和分析。")
        else:
            print(f"\n❌ 因子开发过程中出现问题，请检查日志获取详细信息。")
            
    except Exception as e:
        logger.error(f"执行工作流程失败: {e}")
        print(f"\n❌ 工作流程执行失败: {e}")
        
        # 尝试获取管理器统计信息
        try:
            stats = manager.get_stats()
            print(f"管理器统计: {stats}")
        except:
            pass
        
        raise


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("开始使用experimental_lab框架开发复杂盈利能力因子")
    print("="*60)
    
    main()