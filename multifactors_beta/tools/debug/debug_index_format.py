#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试索引格式问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    
    # 加载财务数据
    financial_path = project_root / "data" / "auxiliary" / "FinancialData_unified.pkl"
    financial_data = pd.read_pickle(financial_path)
    
    print("=== 财务数据索引信息 ===")
    print(f"Index类型: {type(financial_data.index)}")
    print(f"Index名称: {financial_data.index.names}")
    print(f"前5个索引值:")
    for i in range(min(5, len(financial_data))):
        print(f"  {i}: {financial_data.index[i]}")
    
    # 从财务数据中计算净资产并查看其格式
    from factors.generator.financial.pure_financial_factors import PureFinancialFactorCalculator
    calc = PureFinancialFactorCalculator()
    
    # 计算ROE_ttm看看结果索引格式
    roe = calc.calculate_ROE_ttm(financial_data)
    print(f"\n=== ROE因子索引信息 ===")
    print(f"ROE Index类型: {type(roe.index)}")
    print(f"ROE Index名称: {roe.index.names}")
    print(f"ROE 前5个索引值:")
    for i in range(min(5, len(roe))):
        print(f"  {i}: {roe.index[i]}")
    
    # 构造市值数据看看格式
    trading_dates = pd.read_pickle(project_root / "data" / "auxiliary" / "TradingDates.pkl")
    stock_info = pd.read_pickle(project_root / "data" / "auxiliary" / "StockInfo.pkl")
    
    stock_codes = stock_info.index
    recent_dates = trading_dates[-252:]
    
    multi_index = pd.MultiIndex.from_product(
        [recent_dates, stock_codes], 
        names=['date', 'asset']
    )
    
    market_cap = pd.Series(
        np.random.lognormal(22, 1.5, len(multi_index)),
        index=multi_index,
        name='market_cap'
    )
    
    print(f"\n=== 市值数据索引信息 ===")
    print(f"Market_cap Index类型: {type(market_cap.index)}")
    print(f"Market_cap Index名称: {market_cap.index.names}")
    print(f"Market_cap 前5个索引值:")
    for i in range(min(5, len(market_cap))):
        print(f"  {i}: {market_cap.index[i]}")
    
    # 尝试对齐
    try:
        roe_aligned, market_cap_aligned = roe.align(market_cap, join='inner')
        print(f"\n=== 对齐结果 ===")
        print(f"ROE对齐后: {roe_aligned.shape}")
        print(f"Market_cap对齐后: {market_cap_aligned.shape}")
    except Exception as e:
        print(f"\n=== 对齐失败 ===")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()