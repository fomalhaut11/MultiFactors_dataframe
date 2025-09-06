#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EP_ttm因子使用示例
展示清晰的引用关系和使用模式
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def demo_ep_ttm_factor():
    """演示EP_ttm因子的计算和使用"""
    
    print("=" * 60)
    print("EP_ttm因子计算演示 - 使用现有工具")
    print("=" * 60)
    
    try:
        # 方式1: 使用便捷函数（推荐）
        print("\n方式1: 使用便捷函数（自动加载数据）")
        from factors.generator.financial.ep_ttm_factor import calculate_ep_ttm
        
        # 直接计算，自动加载所需数据
        ep_factor = calculate_ep_ttm()
        print(f"EP_ttm因子计算成功: {ep_factor.shape}")
        print(f"样本统计: 均值={ep_factor.mean():.4f}, 标准差={ep_factor.std():.4f}")
        
        # 方式2: 使用因子类
        print("\n方式2: 使用因子类")
        from factors.generator.financial.ep_ttm_factor import EP_ttm_Factor
        
        factor_calculator = EP_ttm_Factor()
        print(f"因子名称: {factor_calculator.name}")
        print(f"因子类别: {factor_calculator.category}")  
        print(f"因子描述: {factor_calculator.description}")
        
        # 计算因子（自动加载数据）
        ep_factor_custom = factor_calculator.calculate()
        print(f"因子计算完成: {ep_factor_custom.shape}")
        
        # 方式3: 展示使用的现有工具
        print("\n方式3: 使用的现有工具")
        print("  - FinancialReportProcessor.calculate_ttm(): TTM计算")
        print("  - DataProcessingMixin._expand_and_align_data(): 数据对齐") 
        print("  - FactorDataLoader: 数据加载")
        print("  - MultiIndexHelper.align_data(): 索引对齐")
        print("  - FactorBase.preprocess(): 去极值和标准化")
        
        # 方式4: 与测试系统集成
        print("\n方式4: 集成测试示例")
        print("因子可直接用于单因子测试:")
        print(f"  factor.name = '{factor_calculator.name}'")
        print(f"  factor.category = '{factor_calculator.category}'") 
        print("  可调用: factor.calculate()")
        print("  可测试: SingleFactorTestPipeline().run('EP_ttm')")
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()


def create_mock_financial_data():
    """创建模拟财务数据"""
    # 创建时间序列（季度报告期）
    report_dates = pd.date_range('2023-03-31', '2024-06-30', freq='Q')
    stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [report_dates, stock_codes], 
        names=['ReportDates', 'StockCodes']
    )
    
    # 模拟净利润数据（单位：亿元）
    np.random.seed(42)
    net_income_data = np.random.normal(5, 2, len(index))  # 季度净利润
    net_income_data = np.maximum(net_income_data, 0.1)  # 确保为正
    
    # 创建DataFrame
    financial_data = pd.DataFrame({
        'net_income': net_income_data
    }, index=index)
    
    return financial_data


def create_mock_market_cap_data():
    """创建模拟市值数据"""
    # 创建交易日序列
    trading_dates = pd.date_range('2023-01-01', '2024-08-31', freq='B')  # 工作日
    stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [trading_dates, stock_codes], 
        names=['TradingDates', 'StockCodes']
    )
    
    # 模拟市值数据（单位：亿元）
    np.random.seed(123)
    market_cap_data = np.random.normal(1000, 300, len(index))  # 总市值
    market_cap_data = np.maximum(market_cap_data, 100)  # 确保合理范围
    
    # 创建Series
    market_cap_series = pd.Series(market_cap_data, index=index, name='market_cap')
    
    return market_cap_series


def show_architecture_benefits():
    """展示新架构的优势"""
    print("\n" + "=" * 60)
    print("新架构设计优势")
    print("=" * 60)
    
    print("\n1. 清晰的职责分离:")
    print("   - EP_ttm_Factor: 专门计算EP_ttm因子")
    print("   - FactorBase: 提供通用功能（预处理、数据验证）")
    print("   - calculate_ep_ttm(): 便捷函数接口")
    
    print("\n2. 明确的依赖关系:")
    print("   - 输入：财务数据 + 市值数据")
    print("   - 处理：TTM计算 -> 数据对齐 -> EP计算 -> 预处理")
    print("   - 输出：标准化的因子Series")
    
    print("\n3. 易于测试和维护:")
    print("   - 单个因子一个文件")
    print("   - 清晰的计算步骤")
    print("   - 完整的数据验证")
    print("   - 便捷函数和类两种使用方式")
    
    print("\n4. 与现有系统兼容:")
    print("   - 继承FactorBase，可用于SingleFactorTestPipeline")
    print("   - 标准的MultiIndex数据格式")
    print("   - 统一的预处理流程")


if __name__ == "__main__":
    # 运行演示
    demo_ep_ttm_factor()
    show_architecture_benefits()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)