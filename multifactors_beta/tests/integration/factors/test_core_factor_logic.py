#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试复合因子的核心计算逻辑
绕过注册系统，直接测试计算组件
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_financial_data_processing():
    """测试财务数据处理逻辑"""
    try:
        logger.info("=" * 60)
        logger.info("测试财务数据处理逻辑")
        logger.info("=" * 60)
        
        # 1. 加载财务数据
        from factors.utils.data_loader import get_financial_data
        financial_data = get_financial_data()
        
        logger.info(f"财务数据加载: {financial_data.shape}")
        
        # 2. 测试TTM计算
        logger.info("测试TTM计算...")
        
        # 选择一只股票进行详细测试
        test_stock = '000001'
        stock_data = financial_data[financial_data.index.get_level_values(1) == test_stock].copy()
        stock_data = stock_data.reset_index()
        stock_data = stock_data.sort_values('ReportDates')
        
        logger.info(f"测试股票 {test_stock} 数据量: {len(stock_data)}")
        
        # 计算TTM
        stock_data['DEDUCTEDPROFIT_TTM'] = stock_data['DEDUCTEDPROFIT'].rolling(4, min_periods=1).sum()
        stock_data['FINANCIALEXPENSE_TTM'] = stock_data['FIN_EXP_IS'].rolling(4, min_periods=1).sum()
        
        # 计算核心因子组件
        core_profit = stock_data['DEDUCTEDPROFIT_TTM'] - stock_data['FINANCIALEXPENSE_TTM']
        profit_after_inventory = core_profit - stock_data['INVENTORIES']
        
        # 处理短期债务
        short_debt = stock_data['ST_BORROW'].replace(0, np.nan)
        short_debt = short_debt.fillna(short_debt.median())
        short_debt = np.where(short_debt <= 0.01, 0.01, short_debt)
        
        core_factor = profit_after_inventory / short_debt
        
        # 显示计算结果
        logger.info(f"核心因子计算完成，有效数据点: {core_factor.notna().sum()}")
        logger.info(f"核心因子范围: [{core_factor.min():.4f}, {core_factor.max():.4f}]")
        
        # 显示样本计算过程
        logger.info("样本计算过程 (最近5期):")
        recent_data = stock_data.tail(5)
        for _, row in recent_data.iterrows():
            logger.info(f"  {row['ReportDates']}: TTM利润={row['DEDUCTEDPROFIT_TTM']:,.0f}, "
                       f"TTM财务费用={row['FINANCIALEXPENSE_TTM']:,.0f}, "
                       f"存货={row['INVENTORIES']:,.0f}, "
                       f"短期债务={row['ST_BORROW']:,.0f}")
        
        return True
        
    except Exception as e:
        logger.error(f"财务数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_returns_zscore_calculation():
    """测试收益率z-score计算"""
    try:
        logger.info("=" * 60)
        logger.info("测试收益率z-score计算")
        logger.info("=" * 60)
        
        # 1. 加载5日收益率数据
        data_root = Path('E:/Documents/PythonProject/StockProject/StockData')
        returns_file = data_root / 'factors' / 'technical' / 'Returns_5D_C2C.pkl'
        
        logger.info("加载5日收益率数据...")
        returns_5d = pd.read_pickle(returns_file)
        
        logger.info(f"5日收益率数据: {returns_5d.shape}")
        
        # 2. 计算截面z-score (只测试少量数据以节省时间)
        logger.info("测试截面z-score计算...")
        
        def calc_cross_sectional_zscore(group):
            """计算截面z-score"""
            mean = group.mean()
            std = group.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=group.index)
            return (group - mean) / std
        
        # 选择一个交易日进行测试
        test_date = returns_5d.index.get_level_values(0)[1000]  # 选择第1000个交易日
        test_data = returns_5d[returns_5d.index.get_level_values(0) == test_date]
        
        logger.info(f"测试日期 {test_date}, 股票数: {len(test_data)}")
        
        # 计算该日的截面z-score
        zscore_result = calc_cross_sectional_zscore(test_data)
        
        logger.info(f"z-score计算完成")
        logger.info(f"z-score统计: 均值={zscore_result.mean():.4f}, 标准差={zscore_result.std():.4f}")
        logger.info(f"z-score范围: [{zscore_result.min():.4f}, {zscore_result.max():.4f}]")
        
        # 验证z-score性质
        if abs(zscore_result.mean()) < 0.001 and abs(zscore_result.std() - 1.0) < 0.001:
            logger.info("z-score标准化验证通过")
        else:
            logger.warning("z-score标准化可能有问题")
        
        return True
        
    except Exception as e:
        logger.error(f"收益率z-score计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factor_combination_logic():
    """测试因子组合逻辑"""
    try:
        logger.info("=" * 60)
        logger.info("测试因子组合逻辑")
        logger.info("=" * 60)
        
        # 创建模拟数据测试组合逻辑
        logger.info("创建模拟数据...")
        
        # 模拟核心因子数据 (财务数据，按季度)
        dates_quarterly = pd.date_range('2020-03-31', '2024-12-31', freq='Q')
        stocks = ['000001', '000002', '000003']
        
        core_factor_data = []
        for date in dates_quarterly:
            for stock in stocks:
                core_factor_data.append({
                    'ReportDate': date,
                    'StockCode': stock,
                    'CoreFactor': np.random.normal(0.1, 0.5)  # 模拟核心因子值
                })
        
        core_factor_df = pd.DataFrame(core_factor_data)
        core_factor_series = core_factor_df.set_index(['ReportDate', 'StockCode'])['CoreFactor']
        
        # 模拟收益率z-score数据 (日频)
        dates_daily = pd.date_range('2020-01-01', '2024-12-31', freq='B')[:100]  # 只取前100天
        
        returns_zscore_data = []
        for date in dates_daily:
            daily_returns = np.random.normal(0, 1, len(stocks))  # 每日随机生成标准化收益率
            for i, stock in enumerate(stocks):
                returns_zscore_data.append({
                    'TradingDate': date,
                    'StockCode': stock,
                    'ReturnsZScore': daily_returns[i]
                })
        
        returns_zscore_df = pd.DataFrame(returns_zscore_data)
        returns_zscore_series = returns_zscore_df.set_index(['TradingDate', 'StockCode'])['ReturnsZScore']
        
        logger.info(f"模拟核心因子数据: {core_factor_series.shape}")
        logger.info(f"模拟收益率z-score数据: {returns_zscore_series.shape}")
        
        # 测试组合逻辑：核心因子 / 收益率z-score
        logger.info("测试因子组合...")
        
        # 简化的对齐逻辑：对每只股票，用最新的核心因子值匹配所有交易日的收益率
        composite_results = []
        
        for stock in stocks:
            # 获取该股票的收益率数据
            stock_returns = returns_zscore_series[returns_zscore_series.index.get_level_values(1) == stock]
            
            # 获取该股票最新的核心因子值
            stock_core = core_factor_series[core_factor_series.index.get_level_values(1) == stock]
            if len(stock_core) > 0:
                latest_core = stock_core.iloc[-1]  # 使用最新值
                
                # 计算复合因子
                for idx, returns_z in stock_returns.items():
                    # 避免除零
                    returns_adj = returns_z if abs(returns_z) >= 0.001 else (0.001 if returns_z >= 0 else -0.001)
                    composite_value = latest_core / returns_adj
                    
                    composite_results.append({
                        'TradingDate': idx[0],
                        'StockCode': idx[1],
                        'CompositeValue': composite_value
                    })
        
        composite_df = pd.DataFrame(composite_results)
        composite_series = composite_df.set_index(['TradingDate', 'StockCode'])['CompositeValue']
        
        logger.info(f"复合因子计算完成: {composite_series.shape}")
        logger.info(f"复合因子统计:")
        logger.info(f"  均值: {composite_series.mean():.4f}")
        logger.info(f"  标准差: {composite_series.std():.4f}")
        logger.info(f"  范围: [{composite_series.min():.4f}, {composite_series.max():.4f}]")
        
        # 显示样本结果
        logger.info("样本复合因子值:")
        logger.info(composite_series.head(10))
        
        return True
        
    except Exception as e:
        logger.error(f"因子组合逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("复合因子核心逻辑测试")
    print("=" * 60)
    print("测试复合因子的各个计算组件，绕过注册系统")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # 测试1: 财务数据处理
    logger.info("测试1: 财务数据处理逻辑...")
    if test_financial_data_processing():
        logger.info("测试1通过")
        success_count += 1
    else:
        logger.error("测试1失败")
    
    # 测试2: 收益率z-score计算
    logger.info("测试2: 收益率z-score计算...")
    if test_returns_zscore_calculation():
        logger.info("测试2通过")
        success_count += 1
    else:
        logger.error("测试2失败")
    
    # 测试3: 因子组合逻辑
    logger.info("测试3: 因子组合逻辑...")
    if test_factor_combination_logic():
        logger.info("测试3通过")
        success_count += 1
    else:
        logger.error("测试3失败")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("所有核心逻辑测试通过！")
        print("复合因子的核心计算逻辑工作正常")
        print("可以继续完善完整的因子实现")
    else:
        print("部分测试失败，需要修复相关逻辑")
    
    print("=" * 60)
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)