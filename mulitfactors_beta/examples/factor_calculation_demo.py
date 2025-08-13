#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算示例
展示如何使用新的因子计算框架
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factors.financial.fundamental_factors import BPFactor, EPFactor, ROEFactor


def main():
    """主函数"""
    print("=" * 60)
    print("多因子计算框架示例")
    print("=" * 60)
    
    # 设置路径
    project_path = Path(r"E:\Documents\PythonProject\StockProject\MultiFactors\mulitfactors_beta")
    data_path = Path(r"E:\Documents\PythonProject\StockProject\StockData")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    try:
        # 加载预处理的辅助数据
        auxiliary_path = project_path / "data" / "auxiliary"
        financial_data = pd.read_pickle(auxiliary_path / "FinancialData_unified.pkl")
        release_dates = pd.read_pickle(auxiliary_path / "ReleaseDates.pkl")
        trading_dates = pd.read_pickle(auxiliary_path / "TradingDates.pkl")
        
        # 加载市值数据
        market_cap = pd.read_pickle(data_path / "MarketCap.pkl")
        if isinstance(market_cap, pd.DataFrame):
            market_cap = market_cap.iloc[:, 0]
            
        print(f"   财务数据形状: {financial_data.shape}")
        print(f"   交易日数量: {len(trading_dates)}")
        print(f"   市值数据点: {len(market_cap)}")
        
    except Exception as e:
        print(f"   数据加载失败: {e}")
        print("   请先运行 python data/prepare_auxiliary_data.py 生成预处理数据")
        return
    
    # 2. 选择测试数据
    print("\n2. 选择测试数据...")
    # 选择部分股票和时间段进行演示
    test_stocks = ['000001', '000002', '000858', '600000', '600036']
    test_start = '2024-01-01'
    test_end = '2024-12-31'
    
    # 过滤数据
    trading_dates_test = trading_dates[(trading_dates >= test_start) & (trading_dates <= test_end)]
    financial_test = financial_data[financial_data.index.get_level_values('StockCodes').isin(test_stocks)]
    release_test = release_dates[release_dates.index.get_level_values('StockCodes').isin(test_stocks)]
    market_test = market_cap[market_cap.index.get_level_values('StockCodes').isin(test_stocks)]
    
    print(f"   测试股票: {test_stocks}")
    print(f"   测试期间: {test_start} 至 {test_end}")
    print(f"   财务数据点: {len(financial_test)}")
    
    # 3. 计算BP因子
    print("\n3. 计算BP因子...")
    bp_factor = BPFactor()
    bp_values = bp_factor.calculate(
        financial_data=financial_test,
        market_cap=market_test,
        release_dates=release_test,
        trading_dates=trading_dates_test
    )
    
    print(f"   BP因子数据点: {len(bp_values)}")
    print(f"   BP因子均值: {bp_values.mean():.4f}")
    print(f"   BP因子标准差: {bp_values.std():.4f}")
    
    # 4. 计算EP_ttm因子
    print("\n4. 计算EP_ttm因子...")
    ep_factor = EPFactor(method='ttm')
    ep_values = ep_factor.calculate(
        financial_data=financial_test,
        market_cap=market_test,
        release_dates=release_test,
        trading_dates=trading_dates_test
    )
    
    print(f"   EP_ttm因子数据点: {len(ep_values)}")
    print(f"   EP_ttm因子均值: {ep_values.mean():.4f}")
    print(f"   EP_ttm因子标准差: {ep_values.std():.4f}")
    
    # 5. 计算ROE_ttm因子
    print("\n5. 计算ROE_ttm因子...")
    roe_factor = ROEFactor(method='ttm')
    roe_values = roe_factor.calculate(
        financial_data=financial_test,
        release_dates=release_test,
        trading_dates=trading_dates_test
    )
    
    print(f"   ROE_ttm因子数据点: {len(roe_values)}")
    print(f"   ROE_ttm因子均值: {roe_values.mean():.4f}")
    print(f"   ROE_ttm因子标准差: {roe_values.std():.4f}")
    
    # 6. 因子相关性分析
    print("\n6. 因子相关性分析...")
    # 将因子合并为DataFrame
    factors_df = pd.DataFrame({
        'BP': bp_values,
        'EP_ttm': ep_values,
        'ROE_ttm': roe_values
    })
    
    # 计算相关系数
    correlation = factors_df.corr()
    print("\n因子相关系数矩阵:")
    print(correlation.round(3))
    
    # 7. 保存结果
    print("\n7. 保存结果...")
    output_path = project_path / "examples" / "factor_results"
    output_path.mkdir(exist_ok=True)
    
    # 保存因子数据
    factors_df.to_csv(output_path / "factor_values.csv")
    correlation.to_csv(output_path / "factor_correlation.csv")
    
    # 保存因子统计
    stats_df = pd.DataFrame({
        'mean': factors_df.mean(),
        'std': factors_df.std(),
        'min': factors_df.min(),
        'max': factors_df.max(),
        'count': factors_df.count()
    })
    stats_df.to_csv(output_path / "factor_statistics.csv")
    
    print(f"   结果保存至: {output_path}")
    
    # 8. 展示部分结果
    print("\n8. 部分结果展示:")
    print("\n最新5个交易日的因子值:")
    print(factors_df.tail())
    
    print("\n各股票的平均因子值:")
    stock_avg = factors_df.groupby(level='StockCodes').mean()
    print(stock_avg.round(4))
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()