#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取真实交易日期的统一接口

提供多种数据源获取真实的A股交易日期：
1. 优先使用TradingDates.pkl（预先准备的交易日历）
2. 备用Price.pkl中的tradingday列
3. 最后回退到工作日（去除周末）
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_real_trading_dates(start_date: str = '2020-01-01', 
                          end_date: str = '2024-12-31',
                          method: str = 'auto') -> Tuple[pd.DatetimeIndex, str]:
    """
    获取真实交易日期
    
    Parameters:
    -----------
    start_date : 开始日期
    end_date : 结束日期  
    method : 获取方法 ('auto', 'trading_dates_pkl', 'price_pkl', 'business_days')
    
    Returns:
    --------
    (trading_dates, source) : 交易日期序列和数据源说明
    """
    
    def try_trading_dates_pkl():
        """尝试从TradingDates.pkl获取"""
        try:
            trading_dates_path = project_root / "data" / "auxiliary" / "TradingDates.pkl"
            if trading_dates_path.exists():
                dates = pd.read_pickle(trading_dates_path)
                # 过滤到指定范围
                filtered_dates = dates[(dates >= start_date) & (dates <= end_date)]
                return pd.DatetimeIndex(filtered_dates), "TradingDates.pkl"
        except Exception as e:
            print(f"无法加载TradingDates.pkl: {e}")
        return None, None
    
    def try_price_pkl():
        """尝试从Price.pkl获取交易日期"""
        try:
            price_path = project_root / "core" / "cache" / "StockData_price.pkl"
            if price_path.exists():
                price_data = pd.read_pickle(price_path)
                if 'tradingday' in price_data.columns:
                    unique_dates = price_data['tradingday'].unique()
                    unique_dates = pd.to_datetime(unique_dates)
                    # 过滤到指定范围
                    filtered_dates = unique_dates[(unique_dates >= start_date) & (unique_dates <= end_date)]
                    return pd.DatetimeIndex(sorted(filtered_dates)), "Price.pkl"
        except Exception as e:
            print(f"无法从Price.pkl提取交易日期: {e}")
        return None, None
    
    def get_business_days():
        """回退到工作日（去除周末）"""
        dates = pd.date_range(start_date, end_date, freq='B')  # Business Day
        return dates, "Business Days (工作日)"
    
    # 根据方法选择数据源
    if method == 'auto':
        # 自动选择：优先级 TradingDates.pkl > Price.pkl > Business Days
        dates, source = try_trading_dates_pkl()
        if dates is not None:
            return dates, source
            
        dates, source = try_price_pkl()
        if dates is not None:
            return dates, source
            
        return get_business_days()
        
    elif method == 'trading_dates_pkl':
        dates, source = try_trading_dates_pkl()
        if dates is not None:
            return dates, source
        else:
            raise ValueError("无法加载TradingDates.pkl")
            
    elif method == 'price_pkl':
        dates, source = try_price_pkl()
        if dates is not None:
            return dates, source
        else:
            raise ValueError("无法从Price.pkl提取交易日期")
            
    elif method == 'business_days':
        return get_business_days()
        
    else:
        raise ValueError(f"未知方法: {method}")


def analyze_trading_dates_sources():
    """分析不同数据源的交易日期"""
    print("=" * 80)
    print("交易日期数据源分析")
    print("=" * 80)
    
    date_range = ('2020-01-01', '2024-12-31')
    methods = ['trading_dates_pkl', 'price_pkl', 'business_days']
    
    results = {}
    
    for method in methods:
        try:
            dates, source = get_real_trading_dates(*date_range, method=method)
            results[method] = {
                'dates': dates,
                'source': source,
                'count': len(dates),
                'range': f"{dates.min()} to {dates.max()}" if len(dates) > 0 else "无数据"
            }
            print(f"\n{method}:")
            print(f"  数据源: {source}")
            print(f"  交易日数量: {len(dates)}")
            print(f"  日期范围: {results[method]['range']}")
            
        except Exception as e:
            print(f"\n{method}: 失败 - {e}")
            results[method] = {'error': str(e)}
    
    # 对比分析
    if 'trading_dates_pkl' in results and 'business_days' in results:
        pkl_count = results['trading_dates_pkl'].get('count', 0)
        business_count = results['business_days'].get('count', 0)
        if pkl_count > 0 and business_count > 0:
            holiday_count = business_count - pkl_count
            print(f"\n对比分析:")
            print(f"  真实交易日: {pkl_count} 天")
            print(f"  工作日: {business_count} 天")
            print(f"  节假日差异: 约 {holiday_count} 天")
            
    return results


def create_debug_trading_dates(start_date: str = '2020-01-01', 
                             end_date: str = '2024-12-31',
                             preferred_method: str = 'auto') -> pd.DatetimeIndex:
    """
    为调试创建真实交易日期序列
    
    这是一个便捷函数，可以直接在调试脚本中使用
    
    注意：推荐使用新的 utils.trading_dates_utils.get_trading_dates
    """
    dates, source = get_real_trading_dates(start_date, end_date, preferred_method)
    print(f"使用交易日期: {source}")
    print(f"日期范围: {dates.min()} to {dates.max()}")
    print(f"交易日数量: {len(dates)}")
    return dates


if __name__ == "__main__":
    # 分析所有数据源
    analyze_trading_dates_sources()
    
    print(f"\n" + "=" * 80)
    print("推荐使用方式")
    print("=" * 80)
    
    # 获取2024年的真实交易日期
    real_dates = create_debug_trading_dates('2024-01-01', '2024-12-31')
    
    print(f"\n2024年部分交易日期:")
    print(real_dates[:10])  # 前10个
    print("...")
    print(real_dates[-10:])  # 后10个
    
    print(f"\n可以在调试脚本中这样使用:")
    print("from get_real_trading_dates import create_debug_trading_dates")
    print("trading_dates = create_debug_trading_dates('2020-01-01', '2024-12-31')")