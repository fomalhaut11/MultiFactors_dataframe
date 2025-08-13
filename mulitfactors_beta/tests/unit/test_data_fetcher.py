#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据获取功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher.data_fetcher import get_stock_data, get_market_data
from core.database import execute_query

def test_data_fetcher():
    """测试数据获取器"""
    print("=" * 60)
    print("测试数据获取功能")
    print("=" * 60)
    
    try:
        # 1. 测试获取交易日期
        print("1. 测试获取交易日期...")
        trading_dates = get_market_data('trading_dates')
        print(f"[OK] 获取交易日期成功，共 {len(trading_dates)} 个交易日")
        if len(trading_dates) > 0:
            print(f"   最早日期: {trading_dates.iloc[0, 0]}")
            print(f"   最新日期: {trading_dates.iloc[-1, 0]}")
        print()
        
        # 2. 测试获取少量价格数据
        print("2. 测试获取价格数据（2025年1月）...")
        price_data = get_stock_data('price', begin_date=20250101, end_date=20250131)
        print(f"[OK] 获取价格数据成功，形状: {price_data.shape}")
        if len(price_data) > 0:
            print(f"   包含股票数: {price_data['code'].nunique()}")
            print(f"   日期范围: {price_data['tradingday'].min()} - {price_data['tradingday'].max()}")
        print()
        
        # 3. 测试SQL执行器
        print("3. 测试SQL执行器...")
        result = execute_query("SELECT COUNT(*) FROM day5", db_name='database')
        print(f"[OK] day5表共有 {result[0][0]} 条记录")
        print()
        
        # 4. 测试获取ST股票
        print("4. 测试获取ST股票...")
        st_data = get_market_data('st_stocks')
        print(f"[OK] 获取ST股票数据成功，共 {len(st_data)} 条记录")
        print()
        
        # 5. 测试获取可交易股票数据
        print("5. 测试获取可交易股票数据...")
        tradable_data = get_stock_data('tradable')
        print(f"[OK] 获取可交易股票数据成功，共 {len(tradable_data)} 条记录")
        print()
        
        print("=" * 60)
        print("所有测试通过！数据获取功能正常工作。")
        print("=" * 60)
        
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_fetcher()