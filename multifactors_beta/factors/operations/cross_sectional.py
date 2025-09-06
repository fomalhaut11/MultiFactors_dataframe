#!/usr/bin/env python3
"""
截面操作模块

提供截面（横截面）操作功能，即对同一时间点上的不同股票进行操作：
- 排序和排名
- 标准化和归一化
- 分位数计算
- 去极值处理
- 因子中性化

所有函数都接受MultiIndex Series格式的数据，并返回相同格式的结果。

Author: AI Assistant
Date: 2025-08-26
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def cross_rank(data: pd.Series, 
               method: str = 'average', 
               ascending: bool = True,
               pct: bool = False) -> pd.Series:
    """
    截面排序
    
    对每个交易日的股票因子值进行排序，返回排名
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    method : str, default 'average'
        处理并列值的方法：'average', 'min', 'max', 'first', 'dense'
    ascending : bool, default True
        是否升序排列
    pct : bool, default False
        是否返回百分位数排名（0-1之间）
        
    Returns
    -------
    pd.Series
        排名结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 对因子值进行截面排序
    >>> ranked = cross_rank(factor_data, method='average')
    >>> # 返回百分位数排名
    >>> pct_ranked = cross_rank(factor_data, pct=True)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    try:
        # 按交易日分组，对每组进行排序
        result = data.groupby(level=0).rank(
            method=method, 
            ascending=ascending, 
            pct=pct
        )
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"截面排序完成，数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"截面排序失败: {e}")
        raise


def cross_zscore(data: pd.Series, 
                 robust: bool = False) -> pd.Series:
    """
    截面标准化
    
    对每个交易日的股票因子值进行标准化，使其均值为0，标准差为1
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    robust : bool, default False
        是否使用稳健统计量（中位数和MAD）进行标准化
        
    Returns
    -------
    pd.Series
        标准化后的结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 普通标准化
    >>> zscore = cross_zscore(factor_data)
    >>> # 稳健标准化
    >>> robust_zscore = cross_zscore(factor_data, robust=True)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    try:
        def _zscore_group(group):
            """对单个日期的数据进行标准化"""
            if len(group.dropna()) <= 1:
                return group * np.nan
            
            if robust:
                # 使用稳健统计量
                median = group.median()
                mad = (group - median).abs().median()
                if mad == 0:
                    return group * 0  # 如果MAD为0，返回0
                return (group - median) / (1.4826 * mad)  # 1.4826是正态分布下的调整系数
            else:
                # 使用均值和标准差
                mean = group.mean()
                std = group.std()
                if std == 0:
                    return group * 0  # 如果标准差为0，返回0
                return (group - mean) / std
        
        # 按交易日分组进行标准化
        result = data.groupby(level=0).apply(_zscore_group)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"截面标准化完成，数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"截面标准化失败: {e}")
        raise


def cross_percentile(data: pd.Series,
                    q: Union[float, List[float]] = 0.5) -> pd.Series:
    """
    截面分位数计算
    
    计算每个交易日上各股票因子值的分位数位置
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    q : float or list of float, default 0.5
        分位数值，0-1之间。如果提供列表，返回对应的多个分位数
        
    Returns
    -------
    pd.Series
        分位数结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 计算中位数位置
    >>> median_pct = cross_percentile(factor_data, q=0.5)
    >>> # 计算90%分位数位置
    >>> p90_pct = cross_percentile(factor_data, q=0.9)
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if isinstance(q, (list, tuple)) and len(q) > 1:
        raise ValueError("当前版本只支持单个分位数值")
    
    if isinstance(q, (list, tuple)):
        q = q[0]
    
    if not 0 <= q <= 1:
        raise ValueError("分位数值必须在0-1之间")
    
    try:
        def _percentile_group(group):
            """计算单个日期数据的分位数位置"""
            if len(group.dropna()) == 0:
                return group * np.nan
            
            # 计算每个值在该日期所有股票中的百分位数位置
            return group.rank(pct=True)
        
        result = data.groupby(level=0).apply(_percentile_group)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"截面分位数计算完成，数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"截面分位数计算失败: {e}")
        raise


def cross_winsorize(data: pd.Series,
                   limits: tuple = (0.025, 0.025),
                   inclusive: tuple = (True, True)) -> pd.Series:
    """
    截面去极值
    
    对每个交易日的因子值进行截面去极值处理
    
    Parameters
    ----------
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    limits : tuple of float, default (0.025, 0.025)
        下限和上限的分位数，例如(0.025, 0.025)表示去除2.5%的极端值
    inclusive : tuple of bool, default (True, True)
        边界值是否包含在内
        
    Returns
    -------
    pd.Series
        去极值后的结果，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 标准去极值（去除上下2.5%）
    >>> winsorized = cross_winsorize(factor_data)
    >>> # 只去除上端1%的极值
    >>> winsorized = cross_winsorize(factor_data, limits=(0, 0.01))
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise ValueError("limits必须是包含两个元素的元组")
    
    if not all(0 <= x <= 1 for x in limits):
        raise ValueError("limits中的值必须在0-1之间")
    
    try:
        def _winsorize_group(group):
            """对单个日期的数据进行去极值"""
            if len(group.dropna()) <= 2:
                return group
            
            # 计算分位数
            lower_quantile = group.quantile(limits[0])
            upper_quantile = group.quantile(1 - limits[1])
            
            # 进行截断
            result = group.copy()
            if inclusive[0]:
                result[result <= lower_quantile] = lower_quantile
            else:
                result[result < lower_quantile] = lower_quantile
            
            if inclusive[1]:
                result[result >= upper_quantile] = upper_quantile
            else:
                result[result > upper_quantile] = upper_quantile
            
            return result
        
        result = data.groupby(level=0).apply(_winsorize_group)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"截面去极值完成，数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"截面去极值失败: {e}")
        raise


def cross_neutralize(data: pd.Series,
                    control_factors: Union[pd.Series, List[pd.Series], pd.DataFrame],
                    method: str = 'ols') -> pd.Series:
    """
    截面中性化
    
    对每个交易日，将因子对控制变量进行回归，返回残差作为中性化后的因子
    
    Parameters
    ----------
    data : pd.Series
        输入因子数据，MultiIndex格式[TradingDates, StockCodes]
    control_factors : pd.Series, list of pd.Series, or pd.DataFrame
        控制变量，用于中性化。可以是单个因子、多个因子列表或DataFrame
    method : str, default 'ols'
        回归方法，目前支持'ols'（普通最小二乘）
        
    Returns
    -------
    pd.Series
        中性化后的残差，MultiIndex格式与输入数据相同
        
    Examples
    --------
    >>> # 对市值中性化
    >>> neutral_factor = cross_neutralize(factor, market_cap)
    >>> # 对多个控制变量中性化
    >>> neutral_factor = cross_neutralize(factor, [market_cap, beta])
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是MultiIndex Series格式")
    
    # 处理控制变量格式
    if isinstance(control_factors, pd.Series):
        controls = pd.DataFrame({'control_0': control_factors})
    elif isinstance(control_factors, list):
        controls = pd.DataFrame({f'control_{i}': factor for i, factor in enumerate(control_factors)})
    elif isinstance(control_factors, pd.DataFrame):
        controls = control_factors
    else:
        raise ValueError("控制变量必须是Series、Series列表或DataFrame")
    
    try:
        def _neutralize_group(group_data):
            """对单个日期的数据进行中性化"""
            date = group_data.index[0][0]  # 获取日期
            
            # 获取该日期的控制变量数据
            try:
                group_controls = controls.loc[controls.index.get_level_values('TradingDates') == date]
            except:
                logger.warning(f"日期 {date} 的控制变量数据缺失")
                return group_data * np.nan
            
            # 对齐数据
            common_stocks = group_data.index.intersection(group_controls.index)
            if len(common_stocks) <= len(group_controls.columns):
                logger.warning(f"日期 {date} 的有效数据点过少")
                return group_data * np.nan
            
            y = group_data.loc[common_stocks].values
            X = group_controls.loc[common_stocks].values
            
            # 添加常数项
            X = np.column_stack([np.ones(len(X)), X])
            
            # 去除NaN
            valid_mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if valid_mask.sum() <= X.shape[1]:
                return group_data * np.nan
            
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]
            
            try:
                # OLS回归
                beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                residuals = y - X @ beta
                
                # 构造结果Series
                result = pd.Series(np.nan, index=group_data.index)
                result.loc[common_stocks] = residuals
                
                return result
                
            except np.linalg.LinAlgError:
                logger.warning(f"日期 {date} 的回归计算失败")
                return group_data * np.nan
        
        # 按日期分组进行中性化
        result = data.groupby(level=0).apply(_neutralize_group)
        
        # 展平MultiIndex（因为groupby apply会增加一层索引）
        if isinstance(result.index, pd.MultiIndex) and result.index.nlevels > 2:
            result.index = result.index.droplevel(0)
        
        # 保持原有的名称
        result.name = data.name
        
        logger.debug(f"截面中性化完成，数据形状: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"截面中性化失败: {e}")
        raise


# 便捷别名
rank = cross_rank
zscore = cross_zscore
percentile = cross_percentile
winsorize = cross_winsorize
neutralize = cross_neutralize