"""
Alpha191 运算工具函数

提供Alpha191因子计算中常用的时间序列和截面运算
"""
import pandas as pd
import numpy as np
from typing import Union, Optional


def ts_rank(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    时间序列排名 - 计算过去window期内的排名百分比
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据，索引为时间，列为股票代码
    window : int
        时间窗口
        
    Returns
    -------
    pd.DataFrame
        排名结果，范围[0,1]
    """
    return data.rolling(window).rank(pct=True)


def ts_mean(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列均值"""
    return data.rolling(window).mean()


def ts_std(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列标准差"""
    return data.rolling(window).std()


def ts_max(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列最大值"""
    return data.rolling(window).max()


def ts_min(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列最小值"""
    return data.rolling(window).min()


def delta(data: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    差分运算 - 计算period期的差值
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    period : int
        差分周期
        
    Returns
    -------
    pd.DataFrame
        差分结果
    """
    return data.diff(periods=period)


def rank(data: pd.DataFrame, axis: int = 1, pct: bool = True) -> pd.DataFrame:
    """
    截面排名
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    axis : int
        排名方向，1为按行(跨股票)，0为按列(跨时间)
    pct : bool
        是否返回百分比排名
        
    Returns
    -------
    pd.DataFrame
        排名结果
    """
    return data.rank(axis=axis, pct=pct)


def scale(data: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """
    标准化/缩放，使得绝对值之和为1
    
    Parameters
    ---------- 
    data : pd.DataFrame
        输入数据
    axis : int
        标准化方向，1为按行，0为按列
        
    Returns
    -------
    pd.DataFrame
        标准化结果
    """
    abs_sum = data.abs().sum(axis=axis, keepdims=True)
    return data / (abs_sum + 1e-8)  # 避免除零


def correlation(x: pd.DataFrame, y: pd.DataFrame, 
                window: int) -> pd.DataFrame:
    """
    滚动相关系数
    
    Parameters
    ----------
    x, y : pd.DataFrame
        输入数据
    window : int
        滚动窗口
        
    Returns
    -------
    pd.DataFrame
        相关系数
    """
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, 
               window: int) -> pd.DataFrame:
    """
    滚动协方差
    
    Parameters
    ----------
    x, y : pd.DataFrame
        输入数据
    window : int
        滚动窗口
        
    Returns
    -------
    pd.DataFrame
        协方差
    """
    return x.rolling(window).cov(y)


def ts_sum(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列求和"""
    return data.rolling(window).sum()


def ts_product(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列乘积"""
    return data.rolling(window).apply(np.prod)


def ts_argmax(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列最大值位置"""
    return data.rolling(window).apply(np.argmax)


def ts_argmin(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列最小值位置"""
    return data.rolling(window).apply(np.argmin)


def decay_linear(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    线性衰减加权均值
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    window : int
        时间窗口
        
    Returns
    -------
    pd.DataFrame
        线性衰减加权均值
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    
    def linear_decay(x):
        if len(x) < window:
            return np.nan
        return np.dot(x.values[-window:], weights)
    
    return data.rolling(window).apply(linear_decay)


def sign(data: pd.DataFrame) -> pd.DataFrame:
    """符号函数"""
    return np.sign(data)


def log(data: pd.DataFrame) -> pd.DataFrame:
    """自然对数"""
    return np.log(data.abs())  # 避免负数问题


def abs_func(data: pd.DataFrame) -> pd.DataFrame:
    """绝对值"""
    return data.abs()


def power(data: pd.DataFrame, exp: float) -> pd.DataFrame:
    """幂运算"""
    return np.power(data.abs(), exp) * np.sign(data)


# 条件运算
def condition(condition: pd.DataFrame, x: pd.DataFrame, 
              y: pd.DataFrame) -> pd.DataFrame:
    """
    条件选择运算
    
    Parameters
    ----------
    condition : pd.DataFrame
        条件数据，True/False
    x : pd.DataFrame
        条件为True时的值
    y : pd.DataFrame
        条件为False时的值
        
    Returns
    -------
    pd.DataFrame
        条件选择结果
    """
    return np.where(condition, x, y)