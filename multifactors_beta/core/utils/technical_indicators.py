"""技术指标计算工具模块"""
import pandas as pd
import numpy as np
from typing import Union, Literal, Tuple
import logging

logger = logging.getLogger(__name__)


class MovingAverageCalculator:
    """移动平均计算器"""
    
    @staticmethod
    def simple_moving_average(data: Union[pd.Series, np.ndarray], window: int) -> Union[pd.Series, np.ndarray]:
        """
        计算简单移动平均(SMA)
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
        prices : DataFrame必须包含'close'列，其他方法可能需要'high', 'low', 'open'
        window : 计算窗口
        method : 计算方法
        annualize : 是否年化
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
        
        Returns:
        --------
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
        
        Returns:
        --------
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