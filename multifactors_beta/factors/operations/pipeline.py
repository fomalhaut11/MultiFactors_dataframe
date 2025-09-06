#!/usr/bin/env python3
"""
因子管道操作模块

提供链式因子处理功能，支持流畅的因子处理工作流。

通过FactorPipeline类，用户可以链式调用各种因子操作，
类似于pandas的method chaining方式。

Author: AI Assistant
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
import logging

from .cross_sectional import (
    cross_rank, cross_zscore, cross_percentile, 
    cross_winsorize, cross_neutralize
)
from .time_series import (
    rolling_mean, rolling_std, rolling_corr,
    ewm, lag, diff, returns
)
from .combination import linear_combine, orthogonalize, residualize

logger = logging.getLogger(__name__)


class FactorPipeline:
    """
    因子管道操作类，支持链式调用
    
    通过链式调用的方式对因子数据进行一系列处理，
    包括去极值、标准化、滚动计算、排序等操作。
    
    Examples
    --------
    >>> # 标准的因子处理流程
    >>> result = (FactorPipeline(raw_factor)
    ...           .winsorize((0.01, 0.01))
    ...           .zscore()
    ...           .rolling_mean(20)
    ...           .rank(pct=True)
    ...           .get())
    
    >>> # 复杂的多步骤处理
    >>> result = (FactorPipeline(price_data)
    ...           .returns(method='pct_change', periods=5)
    ...           .rolling_std(window=60)
    ...           .winsorize()
    ...           .zscore(robust=True)
    ...           .neutralize(market_cap)
    ...           .get())
    """
    
    def __init__(self, data: pd.Series):
        """
        初始化管道
        
        Parameters
        ----------
        data : pd.Series
            初始因子数据，MultiIndex格式[TradingDates, StockCodes]
        """
        if not isinstance(data, pd.Series) or not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入数据必须是MultiIndex Series格式")
        
        self.data = data.copy()
        self.operations = []  # 记录操作历史
    
    def pipe(self, func, *args, **kwargs):
        """
        应用任意函数到当前数据
        
        Parameters
        ----------
        func : callable
            要应用的函数
        *args, **kwargs
            函数的参数
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        try:
            self.data = func(self.data, *args, **kwargs)
            self.operations.append(f"{func.__name__}({args}, {kwargs})")
            return self
        except Exception as e:
            logger.error(f"管道操作失败 - {func.__name__}: {e}")
            raise
    
    def winsorize(self, limits: tuple = (0.025, 0.025), 
                  inclusive: tuple = (True, True)):
        """
        去极值处理
        
        Parameters
        ----------
        limits : tuple, default (0.025, 0.025)
            下限和上限的分位数
        inclusive : tuple, default (True, True)
            边界值是否包含在内
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = cross_winsorize(self.data, limits=limits, inclusive=inclusive)
        self.operations.append(f"winsorize(limits={limits})")
        return self
    
    def zscore(self, robust: bool = False):
        """
        标准化处理
        
        Parameters
        ----------
        robust : bool, default False
            是否使用稳健统计量
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = cross_zscore(self.data, robust=robust)
        self.operations.append(f"zscore(robust={robust})")
        return self
    
    def rank(self, method: str = 'average', ascending: bool = True, pct: bool = False):
        """
        截面排序
        
        Parameters
        ----------
        method : str, default 'average'
            处理并列值的方法
        ascending : bool, default True
            是否升序排列
        pct : bool, default False
            是否返回百分位数排名
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = cross_rank(self.data, method=method, ascending=ascending, pct=pct)
        self.operations.append(f"rank(method={method}, pct={pct})")
        return self
    
    def percentile(self, q: float = 0.5):
        """
        分位数计算
        
        Parameters
        ----------
        q : float, default 0.5
            分位数值
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = cross_percentile(self.data, q=q)
        self.operations.append(f"percentile(q={q})")
        return self
    
    def neutralize(self, control_factors: Union[pd.Series, List[pd.Series], pd.DataFrame]):
        """
        中性化处理
        
        Parameters
        ----------
        control_factors : pd.Series, list of pd.Series, or pd.DataFrame
            控制变量
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = cross_neutralize(self.data, control_factors)
        self.operations.append("neutralize")
        return self
    
    def rolling_mean(self, window: int, min_periods: Optional[int] = None):
        """
        滚动均值
        
        Parameters
        ----------
        window : int
            滚动窗口大小
        min_periods : int, optional
            最小观测值数量
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = rolling_mean(self.data, window=window, min_periods=min_periods)
        self.operations.append(f"rolling_mean(window={window})")
        return self
    
    def rolling_std(self, window: int, min_periods: Optional[int] = None):
        """
        滚动标准差
        
        Parameters
        ----------
        window : int
            滚动窗口大小
        min_periods : int, optional
            最小观测值数量
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = rolling_std(self.data, window=window, min_periods=min_periods)
        self.operations.append(f"rolling_std(window={window})")
        return self
    
    def ewm(self, span: Optional[int] = None, alpha: Optional[float] = None, 
            halflife: Optional[float] = None):
        """
        指数加权移动平均
        
        Parameters
        ----------
        span : int, optional
            平滑参数
        alpha : float, optional
            衰减参数
        halflife : float, optional
            半衰期参数
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = ewm(self.data, span=span, alpha=alpha, halflife=halflife)
        self.operations.append(f"ewm(span={span}, alpha={alpha}, halflife={halflife})")
        return self
    
    def lag(self, periods: int = 1):
        """
        滞后处理
        
        Parameters
        ----------
        periods : int, default 1
            滞后期数
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = lag(self.data, periods=periods)
        self.operations.append(f"lag(periods={periods})")
        return self
    
    def diff(self, periods: int = 1):
        """
        差分处理
        
        Parameters
        ----------
        periods : int, default 1
            差分期数
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = diff(self.data, periods=periods)
        self.operations.append(f"diff(periods={periods})")
        return self
    
    def returns(self, method: str = 'pct_change', periods: int = 1):
        """
        收益率计算
        
        Parameters
        ----------
        method : str, default 'pct_change'
            收益率计算方法
        periods : int, default 1
            收益率时间跨度
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = returns(self.data, method=method, periods=periods)
        self.operations.append(f"returns(method={method}, periods={periods})")
        return self
    
    def orthogonalize(self, base_factors: Union[pd.Series, List[pd.Series], pd.DataFrame]):
        """
        正交化处理
        
        Parameters
        ----------
        base_factors : pd.Series, list of pd.Series, or pd.DataFrame
            基础因子
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = orthogonalize(self.data, base_factors)
        self.operations.append("orthogonalize")
        return self
    
    def residualize(self, control_factors: Union[pd.Series, List[pd.Series], pd.DataFrame]):
        """
        残差化处理
        
        Parameters
        ----------
        control_factors : pd.Series, list of pd.Series, or pd.DataFrame
            控制因子
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = residualize(self.data, control_factors)
        self.operations.append("residualize")
        return self
    
    def fillna(self, value=0):
        """
        填充缺失值
        
        Parameters
        ----------
        value : scalar, optional
            填充值
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = self.data.fillna(value)
        self.operations.append(f"fillna(value={value})")
        return self
    
    def dropna(self):
        """
        删除缺失值
        
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = self.data.dropna()
        self.operations.append("dropna")
        return self
    
    def clip(self, lower=None, upper=None):
        """
        截断处理
        
        Parameters
        ----------
        lower : scalar, optional
            下限
        upper : scalar, optional
            上限
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = self.data.clip(lower=lower, upper=upper)
        self.operations.append(f"clip(lower={lower}, upper={upper})")
        return self
    
    def apply(self, func, *args, **kwargs):
        """
        应用自定义函数（按组）
        
        Parameters
        ----------
        func : callable
            应用的函数
        *args, **kwargs
            函数参数
            
        Returns
        -------
        FactorPipeline
            返回自身以支持链式调用
        """
        self.data = self.data.groupby(level=0).apply(func, *args, **kwargs)
        # 展平可能的多层索引
        if isinstance(self.data.index, pd.MultiIndex) and self.data.index.nlevels > 2:
            self.data.index = self.data.index.droplevel(0)
        self.operations.append(f"apply({func.__name__})")
        return self
    
    def get(self) -> pd.Series:
        """
        获取最终结果
        
        Returns
        -------
        pd.Series
            处理后的因子数据
        """
        logger.info(f"管道操作完成，共执行{len(self.operations)}步操作")
        return self.data
    
    def get_operations(self) -> List[str]:
        """
        获取操作历史
        
        Returns
        -------
        List[str]
            操作历史列表
        """
        return self.operations.copy()
    
    def __repr__(self):
        return f"FactorPipeline(operations={len(self.operations)}, shape={self.data.shape})"


def pipeline(data: pd.Series) -> FactorPipeline:
    """
    创建因子管道
    
    Parameters
    ----------
    data : pd.Series
        初始因子数据，MultiIndex格式
        
    Returns
    -------
    FactorPipeline
        因子管道对象
        
    Examples
    --------
    >>> # 创建管道并进行处理
    >>> result = (pipeline(raw_factor)
    ...           .winsorize()
    ...           .zscore()
    ...           .get())
    """
    return FactorPipeline(data)


__all__ = [
    'FactorPipeline',
    'pipeline'
]