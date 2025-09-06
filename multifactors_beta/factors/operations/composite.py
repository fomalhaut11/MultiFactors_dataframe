#!/usr/bin/env python3
"""
复合因子模块

提供常用的复合因子构建功能：
- 动量类因子：价格动量、收益动量等
- 波动率因子：历史波动率、VIX等
- 均值回归因子：短长期均值差异等
- 质量评分：多因子质量组合
- 中性化因子：规模、行业中性等

所有函数都接受MultiIndex Series格式的数据，并返回相同格式的结果。

Author: AI Assistant
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from .cross_sectional import cross_rank, cross_zscore, cross_neutralize
from .time_series import rolling_mean, rolling_std, returns
from .combination import linear_combine

logger = logging.getLogger(__name__)


def momentum(price_data: pd.Series, period: int = 20) -> pd.Series:
    """
    动量因子
    
    计算价格变化率的截面排名作为动量信号
    
    Parameters
    ----------
    price_data : pd.Series
        价格数据，MultiIndex格式[TradingDates, StockCodes]
    period : int, default 20
        动量计算期数（交易日）
        
    Returns
    -------
    pd.Series
        动量因子，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算20日动量
    >>> mom20 = momentum(price_data, period=20)
    >>> # 计算60日动量
    >>> mom60 = momentum(price_data, period=60)
    """
    if not isinstance(price_data, pd.Series) or not isinstance(price_data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    try:
        # 计算收益率
        ret = returns(price_data, method='pct_change', periods=period)
        
        # 截面排名
        momentum_factor = cross_rank(ret, pct=True)
        momentum_factor.name = f'momentum_{period}d'
        
        logger.debug(f"动量因子计算完成，期数: {period}")
        return momentum_factor
        
    except Exception as e:
        logger.error(f"动量因子计算失败: {e}")
        raise


def volatility(returns_data: pd.Series, window: int = 60) -> pd.Series:
    """
    波动率因子
    
    计算收益率滚动标准差的截面排名
    
    Parameters
    ----------
    returns_data : pd.Series
        收益率数据，MultiIndex格式[TradingDates, StockCodes]
    window : int, default 60
        滚动窗口大小（交易日数量）
        
    Returns
    -------
    pd.Series
        波动率因子，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算60日波动率因子
    >>> vol60 = volatility(returns_data, window=60)
    >>> # 计算20日波动率因子
    >>> vol20 = volatility(returns_data, window=20)
    """
    if not isinstance(returns_data, pd.Series) or not isinstance(returns_data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    try:
        # 计算滚动标准差
        vol = rolling_std(returns_data, window=window)
        
        # 截面排名
        volatility_factor = cross_rank(vol, pct=True)
        volatility_factor.name = f'volatility_{window}d'
        
        logger.debug(f"波动率因子计算完成，窗口: {window}")
        return volatility_factor
        
    except Exception as e:
        logger.error(f"波动率因子计算失败: {e}")
        raise


def mean_reversion(price_data: pd.Series, 
                   short_window: int = 5, 
                   long_window: int = 20) -> pd.Series:
    """
    均值回归因子
    
    计算短期与长期移动平均的差异，反映价格偏离程度
    
    Parameters
    ----------
    price_data : pd.Series
        价格数据，MultiIndex格式[TradingDates, StockCodes]
    short_window : int, default 5
        短期移动平均窗口
    long_window : int, default 20
        长期移动平均窗口
        
    Returns
    -------
    pd.Series
        均值回归因子，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算5日与20日均线差异
    >>> mr_factor = mean_reversion(price_data, short_window=5, long_window=20)
    >>> # 计算10日与60日均线差异
    >>> mr_factor = mean_reversion(price_data, short_window=10, long_window=60)
    """
    if not isinstance(price_data, pd.Series) or not isinstance(price_data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if short_window >= long_window:
        raise ValueError("短期窗口必须小于长期窗口")
    
    try:
        # 计算短期和长期移动平均
        short_ma = rolling_mean(price_data, window=short_window)
        long_ma = rolling_mean(price_data, window=long_window)
        
        # 计算差异
        diff = short_ma - long_ma
        
        # 标准化
        mean_reversion_factor = cross_zscore(diff)
        mean_reversion_factor.name = f'mean_reversion_{short_window}_{long_window}'
        
        logger.debug(f"均值回归因子计算完成，窗口: {short_window}/{long_window}")
        return mean_reversion_factor
        
    except Exception as e:
        logger.error(f"均值回归因子计算失败: {e}")
        raise


def quality_score(factors: List[pd.Series], 
                  weights: Optional[List[float]] = None) -> pd.Series:
    """
    质量评分因子
    
    将多个质量因子进行组合并标准化
    
    Parameters
    ----------
    factors : List[pd.Series]
        质量因子列表，每个都是MultiIndex格式
    weights : List[float], optional
        因子权重，如果不提供则等权重
        
    Returns
    -------
    pd.Series
        质量评分因子，MultiIndex格式与输入因子相同
        
    Examples
    --------
    >>> # 等权重组合ROE、ROA、净利率
    >>> quality = quality_score([roe_factor, roa_factor, npm_factor])
    >>> # 指定权重组合
    >>> quality = quality_score([roe_factor, roa_factor], weights=[0.6, 0.4])
    """
    if not factors:
        raise ValueError("至少需要一个因子")
    
    for i, factor in enumerate(factors):
        if not isinstance(factor, pd.Series) or not isinstance(factor.index, pd.MultiIndex):
            raise ValueError(f"第{i}个因子必须是MultiIndex Series格式")
    
    try:
        # 线性组合
        combined = linear_combine(factors, weights=weights, method='equal')
        
        # 标准化
        quality_factor = cross_zscore(combined)
        quality_factor.name = 'quality_score'
        
        logger.debug(f"质量评分计算完成，组合{len(factors)}个因子")
        return quality_factor
        
    except Exception as e:
        logger.error(f"质量评分计算失败: {e}")
        raise


def size_neutral(factor: pd.Series, market_cap: pd.Series) -> pd.Series:
    """
    规模中性化因子
    
    去除因子中的市值影响，返回对市值中性化的残差
    
    Parameters
    ----------
    factor : pd.Series
        原始因子，MultiIndex格式[TradingDates, StockCodes]
    market_cap : pd.Series
        市值数据，MultiIndex格式[TradingDates, StockCodes]
        
    Returns
    -------
    pd.Series
        规模中性化后的因子，MultiIndex格式与输入因子相同
        
    Examples
    --------
    >>> # 对PE因子进行规模中性化
    >>> pe_neutral = size_neutral(pe_factor, market_cap)
    >>> # 对ROE因子进行规模中性化
    >>> roe_neutral = size_neutral(roe_factor, market_cap)
    """
    if not isinstance(factor, pd.Series) or not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("因子数据必须是MultiIndex Series格式")
    
    if not isinstance(market_cap, pd.Series) or not isinstance(market_cap.index, pd.MultiIndex):
        raise ValueError("市值数据必须是MultiIndex Series格式")
    
    try:
        # 截面中性化
        neutral_factor = cross_neutralize(factor, market_cap)
        neutral_factor.name = f"{factor.name}_size_neutral" if factor.name else "size_neutral"
        
        logger.debug("规模中性化完成")
        return neutral_factor
        
    except Exception as e:
        logger.error(f"规模中性化失败: {e}")
        raise


# 便捷别名
momentum_factor = momentum
volatility_factor = volatility
mean_reversion_factor = mean_reversion


__all__ = [
    'momentum',
    'volatility', 
    'mean_reversion',
    'quality_score',
    'size_neutral',
    'momentum_factor',
    'volatility_factor',
    'mean_reversion_factor'
]