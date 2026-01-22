"""
振荡器指标计算模块

提供各种振荡器技术指标的计算功能：
- RSI (相对强弱指标)
- MACD (指数平滑移动平均线)
- 布林带 (Bollinger Bands)
- 滚动Z-score
- KDJ (随机指标)
- CCI (商品通道指标)
- Williams %R (威廉指标)
- ATR (真实波动幅度均值)

Author: AI Assistant
Date: 2025-09-03
Updated: 2025-01
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

    @staticmethod
    def kdj(high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            n: int = 9,
            m1: int = 3,
            m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算KDJ随机指标

        KDJ指标基于最高价、最低价和收盘价计算

        Parameters
        ----------
        high : pd.Series
            最高价
        low : pd.Series
            最低价
        close : pd.Series
            收盘价
        n : int, default 9
            RSV计算周期
        m1 : int, default 3
            K值平滑周期
        m2 : int, default 3
            D值平滑周期

        Returns
        -------
        tuple of pd.Series
            (K, D, J)
            K: 快速随机指标
            D: 慢速随机指标
            J: J指标 = 3*K - 2*D
        """
        # 计算N日内最低价和最高价
        lowest_low = low.rolling(window=n, min_periods=1).min()
        highest_high = high.rolling(window=n, min_periods=1).max()

        # RSV (Raw Stochastic Value)
        rsv = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # K值 (使用EMA平滑)
        k = rsv.ewm(span=m1, adjust=False).mean()

        # D值 (K的EMA)
        d = k.ewm(span=m2, adjust=False).mean()

        # J值
        j = 3 * k - 2 * d

        return k, d, j

    @staticmethod
    def cci(high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 20) -> pd.Series:
        """
        计算商品通道指标 (CCI - Commodity Channel Index)

        CCI = (典型价格 - MA) / (0.015 * 平均偏差)

        Parameters
        ----------
        high : pd.Series
            最高价
        low : pd.Series
            最低价
        close : pd.Series
            收盘价
        window : int, default 20
            计算周期

        Returns
        -------
        pd.Series
            CCI指标值
        """
        # 典型价格
        typical_price = (high + low + close) / 3

        # 移动平均
        ma = typical_price.rolling(window=window).mean()

        # 平均偏差 (Mean Deviation)
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        # CCI
        cci = (typical_price - ma) / (0.015 * mean_deviation + 1e-10)

        return cci

    @staticmethod
    def williams_r(high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   window: int = 14) -> pd.Series:
        """
        计算威廉指标 (Williams %R)

        Williams %R = (最高价 - 收盘价) / (最高价 - 最低价) * -100

        范围: -100 到 0
        - 接近 0: 超买
        - 接近 -100: 超卖

        Parameters
        ----------
        high : pd.Series
            最高价
        low : pd.Series
            最低价
        close : pd.Series
            收盘价
        window : int, default 14
            计算周期

        Returns
        -------
        pd.Series
            Williams %R值 (-100 到 0)
        """
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()

        williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100

        return williams_r

    @staticmethod
    def atr(high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 14) -> pd.Series:
        """
        计算真实波动幅度均值 (ATR - Average True Range)

        ATR是衡量市场波动性的指标

        Parameters
        ----------
        high : pd.Series
            最高价
        low : pd.Series
            最低价
        close : pd.Series
            收盘价
        window : int, default 14
            计算周期

        Returns
        -------
        pd.Series
            ATR值
        """
        # 前一日收盘价
        prev_close = close.shift(1)

        # 真实波幅的三个组成部分
        tr1 = high - low  # 当日高低差
        tr2 = (high - prev_close).abs()  # 当日高点与昨日收盘差
        tr3 = (low - prev_close).abs()  # 当日低点与昨日收盘差

        # 真实波幅
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR (使用EMA)
        atr = tr.ewm(span=window, adjust=False).mean()

        return atr

    @staticmethod
    def dmi(high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算动向指标 (DMI - Directional Movement Index)

        包含 +DI, -DI 和 ADX

        Parameters
        ----------
        high : pd.Series
            最高价
        low : pd.Series
            最低价
        close : pd.Series
            收盘价
        window : int, default 14
            计算周期

        Returns
        -------
        tuple of pd.Series
            (+DI, -DI, ADX)
        """
        # 计算方向变动
        up_move = high.diff()
        down_move = -low.diff()

        # +DM和-DM
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # ATR
        atr = TechnicalIndicators.atr(high, low, close, window)

        # +DI和-DI
        plus_di = 100 * plus_dm.ewm(span=window, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=window, adjust=False).mean() / (atr + 1e-10)

        # DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)

        # ADX (DX的移动平均)
        adx = dx.ewm(span=window, adjust=False).mean()

        return plus_di, minus_di, adx

    @staticmethod
    def stochastic_rsi(close: pd.Series,
                       rsi_window: int = 14,
                       stoch_window: int = 14,
                       k_smooth: int = 3,
                       d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        计算随机RSI (Stochastic RSI)

        将RSI值标准化到0-100范围

        Parameters
        ----------
        close : pd.Series
            收盘价
        rsi_window : int, default 14
            RSI计算周期
        stoch_window : int, default 14
            随机指标周期
        k_smooth : int, default 3
            K线平滑周期
        d_smooth : int, default 3
            D线平滑周期

        Returns
        -------
        tuple of pd.Series
            (Stoch RSI K, Stoch RSI D)
        """
        # 计算RSI
        rsi = TechnicalIndicators.rsi(close, rsi_window)

        # 计算随机RSI
        lowest_rsi = rsi.rolling(window=stoch_window, min_periods=1).min()
        highest_rsi = rsi.rolling(window=stoch_window, min_periods=1).max()

        stoch_rsi = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)

        # K线和D线
        stoch_rsi_k = stoch_rsi.rolling(window=k_smooth).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()

        return stoch_rsi_k, stoch_rsi_d