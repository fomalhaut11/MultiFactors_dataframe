"""
移动平均指标计算模块

提供各种移动平均线的计算功能：
- 简单移动平均(SMA)
- 加权移动平均(WMA)  
- 指数移动平均(EMA)

Author: AI Assistant
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


class MovingAverageCalculator:
    """移动平均计算器"""
    
    @staticmethod
    def simple_moving_average(data: Union[pd.Series, np.ndarray], window: int) -> Union[pd.Series, np.ndarray]:
        """
        计算简单移动平均(SMA)
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            价格序列数据
        window : int
            移动平均窗口大小
            
        Returns:
        --------
        pd.Series or np.ndarray
            简单移动平均结果
        """
        if isinstance(data, pd.Series):
            return data.rolling(window=window, min_periods=1).mean()
        else:
            # numpy实现
            sma = np.zeros_like(data)
            for i in range(len(data)):
                if i < window:
                    sma[i] = np.mean(data[:i+1])
                else:
                    sma[i] = np.mean(data[i-window+1:i+1])
            return sma
    
    @staticmethod
    def weighted_moving_average(
        data: Union[pd.Series, np.ndarray], 
        weights: Union[pd.Series, np.ndarray], 
        window: int
    ) -> Union[pd.Series, np.ndarray]:
        """
        计算加权移动平均(WMA)
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            价格序列数据
        weights : pd.Series or np.ndarray
            权重序列
        window : int
            移动平均窗口大小
            
        Returns:
        --------
        pd.Series or np.ndarray
            加权移动平均结果
        """
        if len(data) != len(weights):
            raise ValueError("Data and weights must have the same length")
            
        if isinstance(data, pd.Series):
            result = pd.Series(index=data.index, dtype=float)
            for i in range(len(data)):
                if i < window:
                    start = 0
                else:
                    start = i - window + 1
                end = i + 1
                
                data_slice = data.iloc[start:end]
                weight_slice = weights.iloc[start:end]
                
                if weight_slice.sum() > 0:
                    result.iloc[i] = (data_slice * weight_slice).sum() / weight_slice.sum()
                else:
                    result.iloc[i] = data.iloc[i]
            return result
        else:
            # numpy实现
            wma = np.zeros_like(data, dtype=float)
            for i in range(len(data)):
                if i < window:
                    start = 0
                else:
                    start = i - window + 1
                end = i + 1
                
                weight_sum = np.sum(weights[start:end])
                if weight_sum > 0:
                    wma[i] = np.sum(data[start:end] * weights[start:end]) / weight_sum
                else:
                    wma[i] = data[i]
            return wma
    
    @staticmethod
    def exponential_moving_average(
        data: Union[pd.Series, np.ndarray], 
        span: int
    ) -> Union[pd.Series, np.ndarray]:
        """
        计算指数移动平均(EMA)
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            价格序列数据
        span : int
            EMA的平滑参数
            
        Returns:
        --------
        pd.Series or np.ndarray
            指数移动平均结果
        """
        if isinstance(data, pd.Series):
            return data.ewm(span=span, adjust=False).mean()
        else:
            # numpy实现
            alpha = 2 / (span + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            return ema