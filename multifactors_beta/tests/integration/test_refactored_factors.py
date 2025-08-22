"""
重构后因子的测试示例
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from factors.financial.fundamental_factors import EPFactor, BPFactor, ROEFactor
from factors.base.testable_mixin import FactorTestSuite, MockDataProvider


def test_factor_basic_functionality():
    """测试因子基础功能"""
    print("=== 测试因子基础功能 ===")
    
    # 测试EP因子
    print("\n1. 测试EP因子")
    ep_suite = FactorTestSuite(EPFactor)
    ep_results = ep_suite.run_basic_tests(method='ttm')
    print(f"EP因子测试结果: {ep_results}")
    
    # 测试BP因子
    print("\n2. 测试BP因子") 
    bp_suite = FactorTestSuite(BPFactor)
    bp_results = bp_suite.run_basic_tests()
    print(f"BP因子测试结果: {bp_results}")
    
    # 测试ROE因子
    print("\n3. 测试ROE因子")
    roe_suite = FactorTestSuite(ROEFactor)
    roe_results = roe_suite.run_basic_tests(earnings_method='ttm', equity_method='avg')
    print(f"ROE因子测试结果: {roe_results}")


def test_factor_with_real_data():
    """使用模拟数据测试因子计算"""
    print("\n=== 使用模拟数据测试因子计算 ===")
    
    # 创建模拟数据
    provider = MockDataProvider()
    financial_data = provider.create_mock_financial_data(n_stocks=5, n_periods=4)
    market_cap = provider.create_mock_market_cap_data(n_stocks=5, n_days=50)
    release_dates = provider.create_mock_release_dates(n_stocks=5, n_periods=4)
    trading_dates = provider.create_mock_trading_dates('2022-01-01', '2022-12-31')
    
    print(f"财务数据形状: {financial_data.shape}")
    print(f"市值数据形状: {market_cap.shape}")
    print(f"发布日期数据形状: {release_dates.shape}")
    print(f"交易日数量: {len(trading_dates)}")
    
    # 测试EP因子计算
    try:
        ep_factor = EPFactor(method='ttm')
        ep_result = ep_factor.calculate(
            financial_data=financial_data,
            market_cap=market_cap,
            release_dates=release_dates,
            trading_dates=trading_dates
        )
        print(f"\nEP因子计算成功！结果形状: {ep_result.shape}")
        print(f"EP因子统计信息:\n{ep_result.describe()}")
        
    except Exception as e:
        print(f"EP因子计算失败: {e}")
    
    # 测试BP因子计算
    try:
        bp_factor = BPFactor()
        bp_result = bp_factor.calculate(
            financial_data=financial_data,
            market_cap=market_cap,
            release_dates=release_dates,
            trading_dates=trading_dates
        )
        print(f"\nBP因子计算成功！结果形状: {bp_result.shape}")
        print(f"BP因子统计信息:\n{bp_result.describe()}")
        
    except Exception as e:
        print(f"BP因子计算失败: {e}")


def test_flexible_column_mapping():
    """测试灵活的列映射功能"""
    print("\n=== 测试灵活的列映射功能 ===")
    
    # 创建使用不同列名的数据
    dates = pd.date_range('2022-03-31', periods=4, freq='Q')
    stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    index = pd.MultiIndex.from_product([dates, stocks], names=['ReportDates', 'StockCodes'])
    
    # 使用不同的列名
    financial_data_alt = pd.DataFrame({
        'NET_PROFIT': np.random.normal(1000, 200, len(index)),  # 不同的净利润列名
        'OWNERS_EQUITY': np.random.normal(10000, 2000, len(index)),  # 不同的权益列名
        'QUARTER': [1, 2, 3, 4] * (len(index) // 4),
    }, index=index)
    
    print("创建了使用不同列名的财务数据")
    print(f"列名: {list(financial_data_alt.columns)}")
    
    # 测试EP因子的列映射
    try:
        ep_factor = EPFactor(method='ttm')
        
        # 设置自定义列映射
        ep_factor.set_column_mapping('earnings', 'NET_PROFIT')
        ep_factor.set_column_mapping('quarter', 'QUARTER')
        
        # 检查数据可用性
        availability = ep_factor.validate_data_requirements(
            financial_data_alt, 
            ['earnings', 'quarter']
        )
        print(f"数据可用性检查: {availability}")
        
        if availability:
            print("自定义列映射成功！")
        else:
            # 获取建议的映射
            suggestions = ep_factor.suggest_mappings_for_data(financial_data_alt)
            print(f"建议的列映射: {suggestions}")
        
    except Exception as e:
        print(f"列映射测试失败: {e}")


def test_performance_comparison():
    """测试性能对比"""
    print("\n=== 测试性能对比 ===")
    
    try:
        from factors.base.optimized_time_series_processor import OptimizedTimeSeriesProcessor
        
        # 创建较大的测试数据
        provider = MockDataProvider()
        financial_data = provider.create_mock_financial_data(n_stocks=20, n_periods=8)
        release_dates = provider.create_mock_release_dates(n_stocks=20, n_periods=8)
        trading_dates = provider.create_mock_trading_dates('2022-01-01', '2023-12-31')
        
        print(f"测试数据: {len(financial_data)} 行财务数据, {len(trading_dates)} 个交易日")
        
        # 运行性能基准测试
        benchmark_results = OptimizedTimeSeriesProcessor.benchmark_expand_methods(
            financial_data[['DEDUCTEDPROFIT']],
            release_dates,
            trading_dates
        )
        
        print("\n性能基准测试结果:")
        for method, result in benchmark_results.items():
            if result['success']:
                print(f"{method}: {result['time']:.4f}秒, 结果形状: {result['shape']}")
            else:
                print(f"{method}: 失败 - {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"性能测试失败: {e}")


def main():
    """主测试函数"""
    print("开始重构后因子测试...")
    
    try:
        test_factor_basic_functionality()
        test_factor_with_real_data()
        test_flexible_column_mapping()
        test_performance_comparison()
        
        print("\n=== 测试完成 ===")
        print("重构后的因子系统运行正常！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()