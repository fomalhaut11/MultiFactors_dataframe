"""
波动率指标计算模块

提供各种波动率指标的计算功能：
- 历史波动率 (多种计算方法)
- 滚动标准差
- Parkinson波动率
- Garman-Klass波动率  
- Yang-Zhang波动率

Author: AI Assistant
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from typing import Union, Literal
import logging

logger = logging.getLogger(__name__)


class VolatilityCalculator:
    """波动率计算器"""
    
    @staticmethod
    def historical_volatility(
        prices: pd.DataFrame,
        window: int = 20,
        method: Literal["simple", "parkinson", "garman_klass", "yang_zhang"] = "simple",
        annualize: bool = True
    ) -> pd.Series:
        """
        计算历史波动率
        
        Parameters:
        -----------
        prices : pd.DataFrame
            价格数据，必须包含'close'列，其他方法可能需要'high', 'low', 'open'
        window : int, default 20
            计算窗口大小
        method : str, default 'simple'
            计算方法：
            - 'simple': 基于收盘价对数收益率
            - 'parkinson': 基于最高最低价
            - 'garman_klass': 综合OHLC价格
            - 'yang_zhang': Yang-Zhang估计器
        annualize : bool, default True
            是否年化波动率
            
        Returns:
        --------
        pd.Series
            历史波动率序列
        """
        if method == "simple":
            returns = np.log(prices['close'] / prices['close'].shift(1))
            volatility = returns.rolling(window=window).std()
            
        elif method == "parkinson":
            # Parkinson's volatility (uses high/low)
            hl_ratio = np.log(prices['high'] / prices['low'])
            volatility = hl_ratio.rolling(window=window).apply(
                lambda x: np.sqrt(np.sum(x**2) / (4 * window * np.log(2)))
            )
            
        elif method == "garman_klass":
            # Garman-Klass volatility
            hl_ratio = np.log(prices['high'] / prices['low'])
            co_ratio = np.log(prices['close'] / prices['open'])
            
            volatility = pd.Series(index=prices.index)
            for i in range(window, len(prices)):
                hl_sum = np.sum(hl_ratio.iloc[i-window:i]**2) / (2 * window)
                co_sum = np.sum(co_ratio.iloc[i-window:i]**2) * (2 * np.log(2) - 1) / window
                volatility.iloc[i] = np.sqrt(hl_sum - co_sum)
                
        elif method == "yang_zhang":
            # Yang-Zhang volatility
            k = 0.34 / (1 + (window + 1) / (window - 1))
            
            # Overnight volatility
            overnight = np.log(prices['open'] / prices['close'].shift(1))
            overnight_var = overnight.rolling(window=window).var()
            
            # Open-to-close volatility
            open_close = np.log(prices['close'] / prices['open'])
            open_close_var = open_close.rolling(window=window).var()
            
            # Rogers-Satchell volatility
            rs = (np.log(prices['high'] / prices['close']) * 
                  np.log(prices['high'] / prices['open']) +
                  np.log(prices['low'] / prices['close']) * 
                  np.log(prices['low'] / prices['open']))
            rs_var = rs.rolling(window=window).mean()
            
            volatility = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if annualize:
            volatility = volatility * np.sqrt(252)
            
        return volatility
    
    @staticmethod
    def rolling_std(data: Union[pd.Series, np.ndarray], window: int) -> Union[pd.Series, np.ndarray]:
        """
        计算滚动标准差
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            输入数据序列
        window : int
            滚动窗口大小
            
        Returns:
        --------
        pd.Series or np.ndarray
            滚动标准差结果
        """
        if isinstance(data, pd.Series):
            return data.rolling(window=window, min_periods=1).std()
        else:
            # numpy实现
            rolling_std = np.zeros_like(data)
            for i in range(len(data)):
                if i < window:
                    rolling_std[i] = np.std(data[:i+1], ddof=1) if i > 0 else 0
                else:
                    rolling_std[i] = np.std(data[i-window+1:i+1], ddof=1)
            return rolling_std