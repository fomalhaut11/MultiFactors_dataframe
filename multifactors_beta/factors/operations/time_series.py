#!/usr/bin/env python3
"""
时序操作模块

提供时间序列操作功能，即对同一股票在不同时间点的数据进行操作：
- 滚动统计：均值、标准差、相关性等
- 指数加权移动平均
- 时序滞后和差分
- 收益率计算

所有函数都接受MultiIndex Series格式的数据，并返回相同格式的结果。

Author: AI Assistant
Date: 2025-08-26
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List
import logging

logger = logging.getLogger(__name__)


def rolling_mean(data: pd.Series,
                window: int,
                min_periods: Optional[int] = None) -> pd.Series:
    """
    滚动均值
    
    计算每只股票的滚动窗口均值
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    window : int
        滚动窗口大小（交易日数量）
    min_periods : int, optional
        计算所需的最小观测值数量，默认为window
        
    Returns
    -------
    pd.Series
        滚动均值结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算20日移动平均
    >>> ma20 = rolling_mean(price_data, window=20)
    >>> # 计算60日移动平均，最少需要30个观测值
    >>> ma60 = rolling_mean(price_data, window=60, min_periods=30)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if window <= 0:
        raise ValueError("窗口大小必须为正整数")
    
    if min_periods is None:
        min_periods = window
    
    try:
        # 按股票分组，对每只股票计算滚动均值
        result = data.groupby(level=1, group_keys=False).rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"滚动均值计算完成，窗口大小: {window}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"滚动均值计算失败: {e}")
        raise


def rolling_std(data: pd.Series,
               window: int,
               min_periods: Optional[int] = None,
               ddof: int = 1) -> pd.Series:
    """
    滚动标准差
    
    计算每只股票的滚动窗口标准差
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    window : int
        滚动窗口大小（交易日数量）
    min_periods : int, optional
        计算所需的最小观测值数量，默认为window
    ddof : int, default 1
        自由度，用于标准差计算
        
    Returns
    -------
    pd.Series
        滚动标准差结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算20日滚动标准差
    >>> std20 = rolling_std(returns_data, window=20)
    >>> # 计算波动率（收益率的标准差）
    >>> volatility = rolling_std(returns_data, window=60)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if window <= 0:
        raise ValueError("窗口大小必须为正整数")
    
    if min_periods is None:
        min_periods = window
    
    try:
        # 按股票分组，对每只股票计算滚动标准差
        result = data.groupby(level=1, group_keys=False).rolling(
            window=window, 
            min_periods=min_periods
        ).std(ddof=ddof)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"滚动标准差计算完成，窗口大小: {window}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"滚动标准差计算失败: {e}")
        raise


def rolling_corr(data: pd.Series,
                other: pd.Series,
                window: int,
                min_periods: Optional[int] = None) -> pd.Series:
    """
    滚动相关性
    
    计算两个时间序列之间的滚动相关系数
    
    Parameters
    ----------
    data : pd.Series
        第一个时间序列，MultiIndex格式[TradingDates, StockCodes]
    other : pd.Series
        第二个时间序列，MultiIndex格式[TradingDates, StockCodes]
    window : int
        滚动窗口大小（交易日数量）
    min_periods : int, optional
        计算所需的最小观测值数量，默认为window
        
    Returns
    -------
    pd.Series
        滚动相关系数结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算股票收益与市场收益的60日滚动相关性
    >>> beta_proxy = rolling_corr(stock_returns, market_returns, window=60)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("data必须是MultiIndex Series格式")
    
    if not isinstance(other, pd.Series) or not isinstance(other.index, pd.MultiIndex):
        raise ValueError("other必须是MultiIndex Series格式")
    
    if window <= 0:
        raise ValueError("窗口大小必须为正整数")
    
    if min_periods is None:
        min_periods = window
    
    try:
        # 对齐两个序列
        data_aligned, other_aligned = data.align(other, join='inner')
        
        # 按股票分组，计算滚动相关性
        def _rolling_corr_group(group):
            stock_code = group.index[0][1]  # 获取股票代码
            other_group = other_aligned.xs(stock_code, level=1)
            return group.droplevel('StockCodes').rolling(
                window=window, 
                min_periods=min_periods
            ).corr(other_group)
        
        result = data_aligned.groupby(level=1, group_keys=False).apply(_rolling_corr_group)
        
        # 保持原有的名称
        result.name = f"{data.name}_corr_{other.name}" if data.name and other.name else None
        
        logger.debug(f"滚动相关性计算完成，窗口大小: {window}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"滚动相关性计算失败: {e}")
        raise


def ewm(data: pd.Series,
        span: Optional[int] = None,
        alpha: Optional[float] = None,
        halflife: Optional[float] = None,
        adjust: bool = True) -> pd.Series:
    """
    指数加权移动平均
    
    计算每只股票的指数加权移动平均值
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    span : int, optional
        平滑参数，相当于N日EMA中的N
    alpha : float, optional
        直接指定衰减参数
    halflife : float, optional
        半衰期参数
    adjust : bool, default True
        是否进行偏差调整
        
    Returns
    -------
    pd.Series
        指数加权移动平均结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算20日EMA
    >>> ema20 = ewm(price_data, span=20)
    >>> # 使用半衰期为10的EMA
    >>> ema_hl10 = ewm(price_data, halflife=10)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    # 参数验证
    param_count = sum(x is not None for x in [span, alpha, halflife])
    if param_count != 1:
        raise ValueError("必须且只能指定span、alpha、halflife中的一个参数")
    
    try:
        # 按股票分组，对每只股票计算EMA
        ewm_params = {}
        if span is not None:
            ewm_params['span'] = span
        elif alpha is not None:
            ewm_params['alpha'] = alpha
        elif halflife is not None:
            ewm_params['halflife'] = halflife
        
        result = data.groupby(level=1, group_keys=False).ewm(
            adjust=adjust, **ewm_params
        ).mean()
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"指数加权移动平均计算完成，参数: {ewm_params}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"指数加权移动平均计算失败: {e}")
        raise


def lag(data: pd.Series,
        periods: int = 1) -> pd.Series:
    """
    时序滞后
    
    将时间序列向前或向后移动指定期数
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    periods : int, default 1
        滞后期数，正数表示向前滞后，负数表示向后移动
        
    Returns
    -------
    pd.Series
        滞后后的结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 向前滞后1期
    >>> lagged_factor = lag(factor_data, periods=1)
    >>> # 向后移动2期
    >>> led_factor = lag(factor_data, periods=-2)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    try:
        # 按股票分组，对每只股票进行滞后
        result = data.groupby(level=1, group_keys=False).shift(periods)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"时序滞后完成，滞后期数: {periods}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"时序滞后失败: {e}")
        raise


def diff(data: pd.Series,
         periods: int = 1) -> pd.Series:
    """
    时序差分
    
    计算时间序列的差分（当前值减去滞后值）
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    periods : int, default 1
        差分的滞后期数
        
    Returns
    -------
    pd.Series
        差分结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 一阶差分
    >>> diff1 = diff(price_data, periods=1)
    >>> # 计算变化量（等同于一阶差分）
    >>> change = diff(factor_data)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if periods <= 0:
        raise ValueError("差分期数必须为正整数")
    
    try:
        # 按股票分组，对每只股票进行差分
        result = data.groupby(level=1, group_keys=False).diff(periods)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"时序差分完成，差分期数: {periods}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"时序差分失败: {e}")
        raise


def returns(data: pd.Series,
           method: str = 'pct_change',
           periods: int = 1) -> pd.Series:
    """
    收益率计算
    
    计算时间序列的收益率（百分比变化或对数收益率）
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    method : str, default 'pct_change'
        收益率计算方法：
        - 'pct_change': 百分比变化 (P_t - P_{t-1}) / P_{t-1}
        - 'log': 对数收益率 ln(P_t / P_{t-1})
    periods : int, default 1
        收益率的时间跨度
        
    Returns
    -------
    pd.Series
        收益率结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算日收益率
    >>> daily_ret = returns(price_data)
    >>> # 计算对数收益率
    >>> log_ret = returns(price_data, method='log')
    >>> # 计算5日收益率
    >>> ret_5d = returns(price_data, periods=5)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if method not in ['pct_change', 'log']:
        raise ValueError("method必须是'pct_change'或'log'")
    
    if periods <= 0:
        raise ValueError("periods必须为正整数")
    
    try:
        if method == 'pct_change':
            # 百分比变化
            result = data.groupby(level=1, group_keys=False).pct_change(periods)
        else:  # method == 'log'
            # 对数收益率
            result = data.groupby(level=1, group_keys=False).apply(
                lambda x: np.log(x / x.shift(periods))
            )
        
        # 保持原有的名称
        result.name = f"{data.name}_returns" if data.name else "returns"
        
        logger.debug(f"收益率计算完成，方法: {method}, 期数: {periods}, 数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"收益率计算失败: {e}")
        raise


# 便捷别名
ma = rolling_mean
std = rolling_std
corr = rolling_corr
ema = ewm
shift = lag
pct_change = lambda data, periods=1: returns(data, method='pct_change', periods=periods)
log_returns = lambda data, periods=1: returns(data, method='log', periods=periods)