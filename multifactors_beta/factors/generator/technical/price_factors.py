"""
价格相关技术因子计算模块
"""
import pandas as pd
import numpy as np
from typing import Optional, Union
import logging

from ...base.factor_base import FactorBase
from core.utils import MovingAverageCalculator, TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumFactor(FactorBase):
    """动量因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name=f'Momentum_{window}', category='technical')
        self.window = window
        self.description = f"Price momentum over {window} days"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算动量因子
        动量 = (P_t - P_{t-n}) / P_{t-n}
        """
        # 获取收盘价
        close_price = price_data['close']
        
        # 计算动量
        momentum = close_price.groupby(level='StockCodes').pct_change(periods=self.window)
        
        # 预处理
        momentum = self.preprocess(momentum)
        
        return momentum


class ReversalFactor(FactorBase):
    """反转因子"""
    
    def __init__(self, window: int = 5):
        super().__init__(name=f'Reversal_{window}', category='technical')
        self.window = window
        self.description = f"Short-term reversal over {window} days"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算反转因子（短期反转）
        反转 = -1 * (P_t - P_{t-n}) / P_{t-n}
        """
        # 获取收盘价
        close_price = price_data['close']
        
        # 计算反转（负的短期收益率）
        reversal = -1 * close_price.groupby(level='StockCodes').pct_change(periods=self.window)
        
        # 预处理
        reversal = self.preprocess(reversal)
        
        return reversal


class MovingAverageFactor(FactorBase):
    """移动平均因子"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name=f'MA_{short_window}_{long_window}', category='technical')
        self.short_window = short_window
        self.long_window = long_window
        self.description = f"Moving average factor (MA{short_window}/MA{long_window})"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算移动平均因子
        MA因子 = MA(短期) / MA(长期) - 1
        """
        # 获取收盘价
        close_price = price_data['close']
        
        # 计算移动平均
        ma_short = close_price.groupby(level='StockCodes').apply(
            lambda x: MovingAverageCalculator.simple_moving_average(x, self.short_window)
        )
        ma_long = close_price.groupby(level='StockCodes').apply(
            lambda x: MovingAverageCalculator.simple_moving_average(x, self.long_window)
        )
        
        # 计算MA因子
        ma_factor = ma_short / ma_long - 1
        
        # 预处理
        ma_factor = self.preprocess(ma_factor)
        
        return ma_factor


class RSIFactor(FactorBase):
    """RSI因子"""
    
    def __init__(self, window: int = 14):
        super().__init__(name=f'RSI_{window}', category='technical')
        self.window = window
        self.description = f"Relative Strength Index over {window} days"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算RSI因子
        """
        # 获取收盘价
        close_price = price_data['close']
        
        # 计算RSI
        rsi = close_price.groupby(level='StockCodes').apply(
            lambda x: TechnicalIndicators.rsi(x, window=self.window)
        )
        
        # 预处理
        rsi = self.preprocess(rsi, standardize=False)  # RSI已经在0-100范围内
        
        return rsi


class BollingerBandsFactor(FactorBase):
    """布林带因子"""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(name=f'BollingerBands_{window}_{num_std}', category='technical')
        self.window = window
        self.num_std = num_std
        self.description = f"Bollinger Bands position (window={window}, std={num_std})"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算布林带位置因子
        BB位置 = (Price - Middle) / (Upper - Lower)
        """
        # 获取收盘价
        close_price = price_data['close']
        
        # 计算布林带
        def _calc_bb_position(price_series):
            middle, upper, lower = TechnicalIndicators.bollinger_bands(
                price_series, 
                window=self.window, 
                num_std=self.num_std
            )
            # 计算位置
            bb_position = (price_series - middle) / (upper - lower).replace(0, np.nan)
            return bb_position
        
        bb_factor = close_price.groupby(level='StockCodes').apply(_calc_bb_position)
        
        # 预处理
        bb_factor = self.preprocess(bb_factor)
        
        return bb_factor


class GapReturnFactor(FactorBase):
    """跳空收益率因子"""
    
    def __init__(self):
        super().__init__(name='GapReturn', category='technical')
        self.description = "Overnight gap return"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算跳空收益率
        跳空收益 = log(Open_t * AdjFactor_t / (Close_{t-1} * AdjFactor_{t-1}))
        """
        # 获取所需数据
        open_price = price_data['open']
        close_price = price_data['close']
        adj_factor = price_data['adjfactor']
        
        # 计算调整后的价格
        adj_open = open_price * adj_factor
        adj_close = close_price * adj_factor
        
        # 计算前一日收盘价
        prev_adj_close = adj_close.groupby(level='StockCodes').shift(1)
        
        # 计算跳空收益率
        gap_return = np.log(adj_open / prev_adj_close)
        
        # 预处理
        gap_return = self.preprocess(gap_return)
        
        return gap_return


class PricePositionFactor(FactorBase):
    """价格位置因子"""
    
    def __init__(self, window: int = 252):
        super().__init__(name=f'PricePosition_{window}', category='technical')
        self.window = window
        self.description = f"Price position in {window}-day range"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算价格位置因子
        位置 = (Price - Min) / (Max - Min)
        """
        # 获取收盘价
        close_price = price_data['close']
        
        # 计算滚动最高最低
        rolling_max = close_price.groupby(level='StockCodes').rolling(
            window=self.window, min_periods=self.window//2
        ).max()
        rolling_min = close_price.groupby(level='StockCodes').rolling(
            window=self.window, min_periods=self.window//2
        ).min()
        
        # 整理索引
        rolling_max.index = rolling_max.index.droplevel(0)
        rolling_min.index = rolling_min.index.droplevel(0)
        
        # 计算位置
        price_position = (close_price - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)
        
        # 预处理
        price_position = self.preprocess(price_position, standardize=False)  # 已经在[0,1]范围
        
        return price_position