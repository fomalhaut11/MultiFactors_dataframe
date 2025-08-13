#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易日期工具模块 - 统一的交易日期获取接口

为所有模块提供统一的真实交易日期，避免在各处硬编码 pd.date_range
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# 全局路径配置
PROJECT_ROOT = Path(__file__).parent.parent
TRADING_DATES_PATH = PROJECT_ROOT / "data" / "auxiliary" / "TradingDates.pkl"
PRICE_DATA_PATH = PROJECT_ROOT / "core" / "cache" / "StockData_price.pkl"


def get_trading_dates(start_date: str = None, 
                     end_date: str = None,
                     fallback_to_business_days: bool = True,
                     source_preference: list = ['pkl', 'price', 'business']) -> pd.DatetimeIndex:
    """
    统一的交易日期获取接口
    
    Parameters:
    -----------
    start_date : 开始日期，如 '2020-01-01'
    end_date : 结束日期，如 '2024-12-31'  
    fallback_to_business_days : 是否在无法获取真实数据时回退到工作日
    source_preference : 数据源优先级列表
    
    Returns:
    --------
    pd.DatetimeIndex : 交易日期序列
    """
    
    for source in source_preference:
        try:
            if source == 'pkl':
                dates = _get_from_pkl(start_date, end_date)
                if dates is not None:
                    logger.debug(f"成功从TradingDates.pkl获取{len(dates)}个交易日")
                    return dates
                    
            elif source == 'price':
                dates = _get_from_price_data(start_date, end_date)
                if dates is not None:
                    logger.debug(f"成功从Price数据获取{len(dates)}个交易日")
                    return dates
                    
            elif source == 'business':
                if fallback_to_business_days:
                    dates = _get_business_days(start_date, end_date)
                    logger.info(f"回退到工作日模式，获取{len(dates)}个工作日")
                    return dates
                    
        except Exception as e:
            logger.debug(f"数据源 {source} 获取失败: {e}")
            continue
    
    # 所有方法都失败了
    raise RuntimeError("无法获取交易日期数据，请检查数据文件是否存在")


def _get_from_pkl(start_date: str = None, end_date: str = None) -> Optional[pd.DatetimeIndex]:
    """从TradingDates.pkl获取交易日期"""
    if not TRADING_DATES_PATH.exists():
        return None
        
    try:
        dates = pd.read_pickle(TRADING_DATES_PATH)
        
        # 过滤日期范围
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]
            
        return pd.DatetimeIndex(dates)
        
    except Exception:
        return None


def _get_from_price_data(start_date: str = None, end_date: str = None) -> Optional[pd.DatetimeIndex]:
    """从Price数据中提取交易日期"""
    if not PRICE_DATA_PATH.exists():
        return None
        
    try:
        price_data = pd.read_pickle(PRICE_DATA_PATH)
        
        if 'tradingday' not in price_data.columns:
            return None
            
        unique_dates = price_data['tradingday'].unique()
        dates = pd.to_datetime(unique_dates)
        
        # 过滤日期范围
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]
            
        return pd.DatetimeIndex(sorted(dates))
        
    except Exception:
        return None


def _get_business_days(start_date: str = None, end_date: str = None) -> pd.DatetimeIndex:
    """获取工作日（回退方案）"""
    start = start_date or '2020-01-01'
    end = end_date or '2024-12-31'
    
    return pd.date_range(start, end, freq='B')  # Business Day


# 便捷函数
def get_recent_trading_dates(days: int = 252) -> pd.DatetimeIndex:
    """获取最近N个交易日"""
    try:
        all_dates = get_trading_dates()
        return all_dates[-days:]
    except Exception:
        # 回退到工作日
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days*2)).strftime('%Y-%m-%d')  # 预留更多天数
        business_days = pd.date_range(start_date, end_date, freq='B')
        return business_days[-days:]


def get_year_trading_dates(year: int) -> pd.DatetimeIndex:
    """获取指定年份的交易日期"""
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    return get_trading_dates(start_date, end_date)


def validate_trading_dates(dates: pd.DatetimeIndex) -> dict:
    """验证交易日期的合理性"""
    result = {
        'total_days': len(dates),
        'date_range': f"{dates.min()} to {dates.max()}",
        'has_weekends': any(dates.dayofweek >= 5),  # 是否包含周末
        'gaps': [],  # 异常的长间隔
    }
    
    # 检查异常间隔（超过7天的间隔可能是长假期）
    if len(dates) > 1:
        gaps = pd.Series(dates[1:]) - pd.Series(dates[:-1])
        long_gaps = gaps[gaps > pd.Timedelta(days=7)]
        result['gaps'] = [(dates[i], dates[i+1], gap.days) for i, gap in enumerate(long_gaps)]
    
    return result


# 向后兼容的别名
create_debug_trading_dates = get_trading_dates


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("交易日期工具测试")
    print("=" * 60)
    
    # 测试不同的获取方式
    test_cases = [
        ("2024年", "2024-01-01", "2024-12-31"),
        ("最近一年", None, None),
        ("2020-2024", "2020-01-01", "2024-12-31"),
    ]
    
    for name, start, end in test_cases:
        try:
            dates = get_trading_dates(start, end)
            validation = validate_trading_dates(dates)
            
            print(f"\n{name}:")
            print(f"  交易日数量: {validation['total_days']}")
            print(f"  日期范围: {validation['date_range']}")
            print(f"  包含周末: {'是' if validation['has_weekends'] else '否'}")
            
            if validation['gaps']:
                print(f"  长假期间隔:")
                for start_gap, end_gap, gap_days in validation['gaps']:
                    print(f"    {start_gap} -> {end_gap} ({gap_days}天)")
                    
        except Exception as e:
            print(f"\n{name}: 获取失败 - {e}")
    
    print(f"\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)