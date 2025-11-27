#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化因子测试脚本
直接测试主要因子类是否可以正常实例化和使用
"""

import pandas as pd
import numpy as np
import logging
import os
import sys

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建测试财务数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')  # 季度数据
    stocks = [f"{i:06d}.SH" for i in range(600000, 600005)]  # 5只股票
    
    financial_data = []
    for date in dates:
        for stock in stocks:
            financial_data.append({
                'TradingDates': date,
                'StockCodes': stock,
                'DEDUCTEDPROFIT': np.random.normal(1e8, 5e7),
                'TOT_SHRHLDR_EQY_EXCL_MIN_INT': np.random.normal(5e9, 1e9),
                'TOT_ASSETS': np.random.normal(1e10, 2e9),
                'TOT_OPER_REV': np.random.normal(5e9, 1e9),
                'OPER_COST': np.random.normal(4e9, 8e8),
                'TOT_LIAB': np.random.normal(5e9, 1e9),
                'TOT_CUR_ASSETS': np.random.normal(3e9, 6e8),
                'TOT_CUR_LIAB': np.random.normal(2e9, 4e8),
                'NET_CASH_FLOWS_OPER_ACT': np.random.normal(8e7, 4e7),
                'FIN_EXP': np.random.normal(1e7, 5e6),
                'INC_TAX': np.random.normal(2e7, 1e7),
                'OPER_PROFIT': np.random.normal(1.2e8, 6e7),
                'd_quarter': f"Q{((date.month-1)//3)+1}"
            })
    
    financial_df = pd.DataFrame(financial_data).set_index(['TradingDates', 'StockCodes'])
    
    # 创建价格数据
    dates_daily = pd.date_range('2020-01-01', '2023-12-31', freq='D')[-500:]  # 最近500天
    price_data = []
    for stock in stocks:
        price = 10.0
        for date in dates_daily:
            price *= (1 + np.random.normal(0, 0.02))
            price = max(price, 1.0)
            price_data.append({
                'TradingDates': date,
                'StockCodes': stock,
                'close': price,
                'open': price * (1 + np.random.normal(0, 0.01)),
                'high': price * (1 + abs(np.random.normal(0, 0.015))),
                'low': price * (1 - abs(np.random.normal(0, 0.015))),
                'volume': np.random.randint(1000000, 10000000)
            })
    
    price_df = pd.DataFrame(price_data).set_index(['TradingDates', 'StockCodes'])
    
    # 创建市值数据
    market_cap = price_df['close'] * 1e8  # 假设1亿股
    
    print(f"财务数据形状: {financial_df.shape}")
    print(f"价格数据形状: {price_df.shape}")
    print(f"市值数据形状: {market_cap.shape}")
    
    return financial_df, price_df, market_cap

def test_financial_report_processor():
    """测试财务报表处理器"""
    print("\n测试FinancialReportProcessor...")
    
    try:
        # 添加搜索路径
        sys.path.insert(0, os.path.dirname(__file__))
        
        from factors.generator.financial.financial_report_processor import FinancialReportProcessor
        
        # 创建简单测试数据
        test_data = pd.DataFrame({
            'DEDUCTEDPROFIT': [100, 200, 150, 300, 120, 250, 180, 350],
            'd_quarter': ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4']
        })
        
        # 测试TTM计算
        ttm_result = FinancialReportProcessor.calculate_ttm(test_data)
        print(f"✓ TTM计算成功，结果形状: {ttm_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ FinancialReportProcessor测试失败: {e}")
        return False

def test_individual_factors():
    """逐个测试因子类"""
    print("\n测试各个因子类...")
    
    financial_data, price_data, market_cap = create_test_data()
    
    test_results = {}
    
    # 测试ROE因子
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from factors.generator.financial.profitability_factors import ROE_ttm_Factor
        
        roe_factor = ROE_ttm_Factor()
        roe_result = roe_factor.calculate(financial_data)
        test_results['ROE_ttm'] = {
            'success': True, 
            'shape': roe_result.shape,
            'non_null_count': roe_result.count()
        }
        print(f"✓ ROE_ttm因子计算成功: {roe_result.shape}, 非空值: {roe_result.count()}")
        
    except Exception as e:
        test_results['ROE_ttm'] = {'success': False, 'error': str(e)}
        print(f"✗ ROE_ttm因子失败: {e}")
    
    # 测试ROA因子
    try:
        from factors.generator.financial.profitability_factors import ROA_ttm_Factor
        
        roa_factor = ROA_ttm_Factor()
        roa_result = roa_factor.calculate(financial_data)
        test_results['ROA_ttm'] = {
            'success': True, 
            'shape': roa_result.shape,
            'non_null_count': roa_result.count()
        }
        print(f"✓ ROA_ttm因子计算成功: {roa_result.shape}, 非空值: {roa_result.count()}")
        
    except Exception as e:
        test_results['ROA_ttm'] = {'success': False, 'error': str(e)}
        print(f"✗ ROA_ttm因子失败: {e}")
    
    # 测试流动比率因子
    try:
        from factors.generator.financial.solvency_factors import CurrentRatio_Factor
        
        cr_factor = CurrentRatio_Factor()
        cr_result = cr_factor.calculate(financial_data)
        test_results['CurrentRatio'] = {
            'success': True, 
            'shape': cr_result.shape,
            'non_null_count': cr_result.count()
        }
        print(f"✓ CurrentRatio因子计算成功: {cr_result.shape}, 非空值: {cr_result.count()}")
        
    except Exception as e:
        test_results['CurrentRatio'] = {'success': False, 'error': str(e)}
        print(f"✗ CurrentRatio因子失败: {e}")
    
    # 测试EP比率因子 (需要市值数据)
    try:
        from factors.generator.financial.value_factors import EPRatioFactor
        
        ep_factor = EPRatioFactor()
        ep_result = ep_factor.calculate(financial_data, market_cap)
        test_results['EP_Ratio'] = {
            'success': True, 
            'shape': ep_result.shape,
            'non_null_count': ep_result.count()
        }
        print(f"✓ EP_Ratio因子计算成功: {ep_result.shape}, 非空值: {ep_result.count()}")
        
    except Exception as e:
        test_results['EP_Ratio'] = {'success': False, 'error': str(e)}
        print(f"✗ EP_Ratio因子失败: {e}")
    
    # 测试动量因子 (需要价格数据)
    try:
        from factors.generator.technical.momentum_factors import MomentumFactor
        
        momentum_factor = MomentumFactor()
        momentum_result = momentum_factor.calculate(price_data)
        test_results['Momentum'] = {
            'success': True, 
            'shape': momentum_result.shape,
            'non_null_count': momentum_result.count()
        }
        print(f"✓ Momentum因子计算成功: {momentum_result.shape}, 非空值: {momentum_result.count()}")
        
    except Exception as e:
        test_results['Momentum'] = {'success': False, 'error': str(e)}
        print(f"✗ Momentum因子失败: {e}")
    
    return test_results

def main():
    """主函数"""
    print("开始因子系统简化测试...")
    print("="*60)
    
    # 测试1：财务报表处理器
    processor_success = test_financial_report_processor()
    
    # 测试2：各个因子类
    factor_results = test_individual_factors()
    
    # 生成测试报告
    print("\n" + "="*60)
    print("测试报告总结")
    print("="*60)
    
    success_count = sum(1 for r in factor_results.values() if r.get('success', False))
    total_count = len(factor_results)
    
    print(f"FinancialReportProcessor: {'成功' if processor_success else '失败'}")
    print(f"因子测试成功率: {success_count}/{total_count}")
    
    print("\n详细结果:")
    for factor_name, result in factor_results.items():
        if result.get('success', False):
            shape = result.get('shape', 'N/A')
            non_null = result.get('non_null_count', 'N/A')
            print(f"  {factor_name}: 成功 - 形状{shape}, 非空值{non_null}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"  {factor_name}: 失败 - {error}")
    
    overall_success = processor_success and success_count > 0
    print(f"\n整体评估: {'通过' if overall_success else '部分失败'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)