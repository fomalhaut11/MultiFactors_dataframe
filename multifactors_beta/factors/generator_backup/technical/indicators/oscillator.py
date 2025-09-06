"""
振荡器指标计算模块

提供各种振荡器技术指标的计算功能：
- RSI (相对强弱指标)
- MACD (指数平滑移动平均线)
- 布林带 (Bollinger Bands)
- 滚动Z-score

Author: AI Assistant  
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """常用技术指标计算"""
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算布林带
        
        Parameters:
        -----------
        data : pd.Series
            价格序列数据
        window : int, default 20
            移动平均窗口大小
        num_std : float, default 2.0
            标准差倍数
            
        Returns:
        --------
        tuple of pd.Series
            (middle_band, upper_band, lower_band)
        """
        middle_band = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        upper_band = middle_band + num_std * rolling_std
        lower_band = middle_band - num_std * rolling_std
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        计算相对强弱指标(RSI)
        
        Parameters:
        -----------
        data : pd.Series
            价格序列数据
        window : int, default 14
            RSI计算窗口
            
        Returns:
        --------
        pd.Series
            RSI指标值 (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        data: pd.Series,
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算MACD指标
        
        Parameters:
        -----------
        data : pd.Series
            价格序列数据
        fast_window : int, default 12
            快速EMA周期
        slow_window : int, default 26
            慢速EMA周期
        signal_window : int, default 9
            信号线EMA周期
            
        Returns:
        --------
        tuple of pd.Series
            (macd_line, signal_line, macd_histogram)
        """
        ema_fast = data.ewm(span=fast_window, adjust=False).mean()
        ema_slow = data.ewm(span=slow_window, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        return macd_line, signal_line, macd_histogram
    
    @staticmethod
    def rolling_zscore(
        data: Union[pd.Series, np.ndarray],
        window: int,
        cap: float = 5.0
    ) -> Union[pd.Series, np.ndarray]:
        """
        计算滚动Z-score
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            输入数据序列
        window : int
            滚动窗口大小
        cap : float, default 5.0
            Z-score截断值
            
        Returns:
        --------
        pd.Series or np.ndarray
            滚动Z-score结果
        """
        if isinstance(data, pd.Series):
            rolling_mean = data.rolling(window=window, min_periods=1).mean()
            rolling_std = data.rolling(window=window, min_periods=1).std()
            
            zscore = (data - rolling_mean) / (rolling_std + 1e-8)
            zscore = zscore.clip(-cap, cap)
            
            return zscore
        else:
            # numpy实现
            zscore = np.zeros_like(data)
            for i in range(len(data)):
                if i < window:
                    start = 0
                else:
                    start = i - window + 1
                    
                window_data = data[start:i+1]
                mean_val = np.mean(window_data)
                std_val = np.std(window_data) + 1e-8
                
                zscore[i] = (data[i] - mean_val) / std_val
                zscore[i] = np.clip(zscore[i], -cap, cap)
                
            return zscore