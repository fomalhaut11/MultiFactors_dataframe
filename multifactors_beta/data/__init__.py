#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data模块统一接口

提供标准化的数据访问接口，保持文件交互设计的同时，
提供格式验证、错误处理和缓存功能

Author: MultiFactors Team
Date: 2025-08-29
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np
import logging

# 核心接口
from .data_bridge import DataBridge, get_data_bridge
from .schemas import (
    validate_price_data, validate_financial_data, validate_factor_format,
    convert_to_factor_format, DataValidator, DataConverter
)

# 更新器
from .fetcher.data_fetcher import StockDataFetcher
from .fetcher.incremental_price_updater import IncrementalPriceUpdater
from .fetcher.incremental_financial_updater import IncrementalFinancialUpdater

# 处理器
from .processor.data_processing_pipeline import DataProcessingPipeline
from .processor.price_processor import PriceDataProcessor
from .processor.financial_processor import FinancialDataProcessor
from .processor.return_calculator import ReturnCalculator

logger = logging.getLogger(__name__)

# 全局数据桥接器实例
_global_bridge = None


def get_factor_data(data_type: str, 
                   column: str,
                   begin_date: Optional[int] = None,
                   end_date: Optional[int] = None,
                   validate: bool = True) -> pd.Series:
    """
    获取标准格式的因子数据（主要接口）
    
    这是factors模块的主要数据接口，内部使用文件读取，
    但提供格式验证和错误处理
    
    Parameters
    ----------
    data_type : str
        数据类型 ('price', 'financial')
    column : str
        数据列名
    begin_date : int, optional
        开始日期 (YYYYMMDD格式)
    end_date : int, optional  
        结束日期 (YYYYMMDD格式)
    validate : bool
        是否验证数据格式
        
    Returns
    -------
    pd.Series
        标准因子格式的数据，索引为[TradingDates, StockCodes]
        
    Examples
    --------
    >>> # 获取收盘价因子
    >>> close_prices = get_factor_data('price', 'c')
    >>> 
    >>> # 获取净利润因子
    >>> net_profit = get_factor_data('financial', 'NET_PROFIT')
    """
    global _global_bridge
    
    if _global_bridge is None:
        _global_bridge = get_data_bridge()
    
    try:
        if data_type == 'price':
            return _global_bridge.price_to_factor(
                value_column=column,
                begin_date=begin_date,
                end_date=end_date,
                validate=validate
            )
        elif data_type == 'financial':
            return _global_bridge.financial_to_factor(
                value_column=column,
                validate=validate
            )
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
            
    except Exception as e:
        logger.error(f"获取因子数据失败 [{data_type}.{column}]: {e}")
        raise


def get_raw_data(data_type: str, **kwargs) -> pd.DataFrame:
    """
    获取原始数据（DataFrame格式）
    
    Parameters
    ----------
    data_type : str
        数据类型 ('price', 'financial', 'trading_dates', 'stock_info')
    **kwargs
        其他参数
        
    Returns
    -------
    pd.DataFrame
        原始数据
    """
    global _global_bridge
    
    if _global_bridge is None:
        _global_bridge = get_data_bridge()
    
    if data_type == 'price':
        return _global_bridge.get_price_data(**kwargs)
    elif data_type == 'financial':
        return _global_bridge.get_financial_data(**kwargs)
    elif data_type == 'trading_dates':
        return _global_bridge.get_trading_dates(**kwargs)
    elif data_type == 'stock_info':
        return _global_bridge.get_stock_info(**kwargs)
    elif data_type == 'release_dates':
        return _global_bridge.get_release_dates(**kwargs)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def validate_data_pipeline() -> bool:
    """
    验证数据管道完整性
    
    Returns
    -------
    bool
        数据管道是否正常
    """
    try:
        bridge = get_data_bridge()
        
        # 检查核心辅助数据文件
        required_files = [
            'TradingDates.pkl',
            'StockInfo.pkl', 
            'FinancialData_unified.pkl',
            'ReleaseDates.pkl'
        ]
        
        for file_name in required_files:
            file_path = bridge.auxiliary_path / file_name
            if not file_path.exists():
                logger.error(f"缺少核心数据文件: {file_path}")
                return False
                
        logger.info("数据管道验证通过")
        return True
        
    except Exception as e:
        logger.error(f"数据管道验证失败: {e}")
        return False


def update_all_data(data_types: Optional[List[str]] = None,
                   force: bool = False) -> Dict[str, bool]:
    """
    批量更新数据
    
    Parameters
    ----------
    data_types : list, optional
        要更新的数据类型列表，None表示更新所有
    force : bool
        是否强制更新
        
    Returns
    -------
    dict
        更新结果
    """
    from .prepare_auxiliary_data import main as prepare_auxiliary
    from ..scheduled_data_updater import main as update_raw_data
    
    results = {}
    
    try:
        # 1. 更新原始数据
        logger.info("开始更新原始数据...")
        update_raw_data()
        results['raw_data'] = True
        
        # 2. 更新辅助数据
        logger.info("开始更新辅助数据...")
        prepare_auxiliary()
        results['auxiliary_data'] = True
        
    except Exception as e:
        logger.error(f"数据更新失败: {e}")
        results['error'] = str(e)
        
    return results


def get_data_status() -> Dict[str, Any]:
    """
    获取数据状态概览
    
    Returns
    -------
    dict
        数据状态信息
    """
    bridge = get_data_bridge()
    return bridge.get_data_quality_report('all')


def clear_cache():
    """清除全局缓存"""
    global _global_bridge
    if _global_bridge is not None:
        _global_bridge.clear_cache()
        

# 导出接口
__all__ = [
    # 主要接口
    'get_factor_data',           # factors模块主要使用这个
    'get_raw_data',
    'validate_data_pipeline',
    'update_all_data',
    'get_data_status',
    'clear_cache',
    
    # 核心类
    'DataBridge',
    'get_data_bridge',
    
    # 验证工具
    'validate_price_data',
    'validate_financial_data', 
    'validate_factor_format',
    'convert_to_factor_format',
    
    # 更新器
    'StockDataFetcher',
    'IncrementalPriceUpdater',
    'IncrementalFinancialUpdater',
    
    # 处理器
    'DataProcessingPipeline',
    'PriceDataProcessor',
    'FinancialDataProcessor',
    'ReturnCalculator',
]