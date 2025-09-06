#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标因子模块 - 振荡器类指标

实现各种技术分析振荡器指标，包括MACD、RSI、Williams %R、CCI等
这些指标通常用于判断超买超卖状态和趋势转换点
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Literal
import logging
from functools import lru_cache

from ...base.factor_base import FactorBase
from .indicators import TechnicalIndicators, MovingAverageCalculator

logger = logging.getLogger(__name__)


class OscillatorFactorBase(FactorBase):
    """振荡器因子基类"""
    
    def __init__(self, name: str, category: str = 'technical'):
        super().__init__(name=name, category=category)
        
    def _get_price_series(self, 
                         price_data: pd.DataFrame, 
                         price_column: str = 'close',
                         handle_adjfactor: bool = True) -> pd.Series:
        """获取复权价格序列"""
        price = price_data[price_column]
        
        if handle_adjfactor and 'adjfactor' in price_data.columns:
            adj_factor = price_data['adjfactor']
            adjusted_price = price * adj_factor
        else:
            adjusted_price = price
            
        return adjusted_price
    
    def _validate_ohlc_data(self, price_data: pd.DataFrame) -> bool:
        """验证OHLC数据是否完整"""
        required_cols = ['open', 'high', 'low', 'close']
        return all(col in price_data.columns for col in required_cols)


class MACDFactor(OscillatorFactorBase):
    """MACD因子 - 指数平滑移动平均收敛发散指标"""
    
    def __init__(self, 
                 fast_window: int = 12,
                 slow_window: int = 26,
                 signal_window: int = 9,
                 output_type: Literal['macd', 'signal', 'histogram'] = 'macd'):
        
        name = f'MACD_{fast_window}_{slow_window}_{signal_window}_{output_type}'
        super().__init__(name=name)
        
        self.fast_window = fast_window
        self.slow_window = slow_window  
        self.signal_window = signal_window
        self.output_type = output_type
        self.description = f"MACD {output_type} line ({fast_window}/{slow_window}/{signal_window})"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD指标（基于后复权收盘价）
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_window)
        MACD Histogram = MACD Line - Signal Line
        
        Parameters:
        -----------
        price_data : DataFrame
            包含 close 和 adjfactor 列的价格数据
        """
        # 获取收盘价和除权因子
        close_price = price_data['close']
        adj_factor = price_data['adjfactor']
        
        # 计算后复权收盘价
        adj_close = close_price * adj_factor
        
        # 按股票分组计算MACD并进行时序移位
        def _calc_macd_with_shift(stock_prices):
            macd_line, signal_line, macd_histogram = TechnicalIndicators.macd(
                stock_prices,
                fast_window=self.fast_window,
                slow_window=self.slow_window, 
                signal_window=self.signal_window
            )
            
            if self.output_type == 'macd':
                result = macd_line
            elif self.output_type == 'signal':
                result = signal_line
            elif self.output_type == 'histogram':
                result = macd_histogram
            else:
                result = macd_line
            
            return result.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        result = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_macd_with_shift)
        
        # 预处理
        result = self.preprocess(result)
        
        return result


class RSIFactor(OscillatorFactorBase):
    """RSI因子 - 相对强弱指标"""
    
    def __init__(self, window: int = 14):
        super().__init__(name=f'RSI_{window}')
        self.window = window
        self.description = f"Relative Strength Index over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI指标（基于后复权收盘价）
        
        RSI = 100 - 100 / (1 + RS)
        其中 RS = 平均上涨幅度 / 平均下跌幅度
        
        Parameters:
        -----------
        price_data : DataFrame
            包含 close 和 adjfactor 列的价格数据
        """
        # 获取收盘价和除权因子
        close_price = price_data['close']
        adj_factor = price_data['adjfactor']
        
        # 计算后复权收盘价
        adj_close = close_price * adj_factor
        
        # 按股票分组计算RSI并进行时序移位
        def _calc_rsi_with_shift(x):
            rsi = TechnicalIndicators.rsi(x, window=self.window)
            return rsi.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        rsi = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_rsi_with_shift)
        
        # 预处理（RSI已经在0-100范围内，不需要标准化）
        rsi = self.preprocess(rsi, standardize=False)
        
        return rsi


class WilliamsRFactor(OscillatorFactorBase):
    """Williams %R因子 - 威廉指标"""
    
    def __init__(self, window: int = 14):
        super().__init__(name=f'WilliamsR_{window}')
        self.window = window
        self.description = f"Williams %R over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Williams %R指标
        
        %R = (Highest High - Close) / (Highest High - Lowest Low) * (-100)
        """
        if not self._validate_ohlc_data(price_data):
            logger.warning("Williams %R需要OHLC数据，回退到使用收盘价")
            close = self._get_price_series(price_data, 'close')
            high = close
            low = close
        else:
            high = self._get_price_series(price_data, 'high')
            low = self._get_price_series(price_data, 'low')
        
        close = self._get_price_series(price_data, 'close')
        
        def _calc_williams_r(stock_data):
            """计算单只股票的Williams %R"""
            stock_high, stock_low, stock_close = stock_data
            
            # 计算滚动最高最低价
            highest_high = stock_high.rolling(window=self.window, min_periods=1).max()
            lowest_low = stock_low.rolling(window=self.window, min_periods=1).min()
            
            # 计算Williams %R
            williams_r = (highest_high - stock_close) / (highest_high - lowest_low) * (-100)
            
            # 处理除零错误
            williams_r = williams_r.where(highest_high != lowest_low, -50)
            
            return williams_r
        
        # 按股票分组计算
        grouped_data = pd.concat([high, low, close], axis=1, keys=['high', 'low', 'close'])
        williams_r = grouped_data.groupby(level='StockCodes').apply(
            lambda x: _calc_williams_r((x['high'], x['low'], x['close']))
        )
        
        # 预处理（Williams %R已经在-100到0范围内）
        williams_r = self.preprocess(williams_r, standardize=False)
        
        return williams_r


class CCIFactor(OscillatorFactorBase):
    """CCI因子 - 商品通道指标"""
    
    def __init__(self, window: int = 20, constant: float = 0.015):
        super().__init__(name=f'CCI_{window}')
        self.window = window
        self.constant = constant
        self.description = f"Commodity Channel Index over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算CCI指标
        
        TP = (High + Low + Close) / 3  (典型价格)
        CCI = (TP - SMA(TP)) / (constant * Mean Deviation)
        """
        if not self._validate_ohlc_data(price_data):
            logger.warning("CCI需要OHLC数据，使用收盘价作为典型价格")
            typical_price = self._get_price_series(price_data, 'close')
        else:
            high = self._get_price_series(price_data, 'high')
            low = self._get_price_series(price_data, 'low') 
            close = self._get_price_series(price_data, 'close')
            typical_price = (high + low + close) / 3
        
        def _calc_cci(stock_tp):
            """计算单只股票的CCI"""
            # 计算典型价格的简单移动平均
            sma_tp = stock_tp.rolling(window=self.window, min_periods=1).mean()
            
            # 计算平均偏差
            def _mean_deviation(values):
                if len(values) < 2:
                    return np.nan
                mean_val = values.mean()
                return np.abs(values - mean_val).mean()
            
            mean_dev = stock_tp.rolling(window=self.window, min_periods=1).apply(_mean_deviation, raw=True)
            
            # 计算CCI
            cci = (stock_tp - sma_tp) / (self.constant * mean_dev)
            
            # 处理除零错误
            cci = cci.where(mean_dev > 1e-8, 0)
            
            return cci
        
        # 按股票分组计算
        cci = typical_price.groupby(level='StockCodes', group_keys=False).apply(_calc_cci)
        
        # 预处理
        cci = self.preprocess(cci)
        
        return cci


class StochasticOscillatorFactor(OscillatorFactorBase):
    """随机振荡器因子 - Stochastic Oscillator"""
    
    def __init__(self, 
                 k_window: int = 14,
                 d_window: int = 3,
                 output_type: Literal['%K', '%D'] = '%K'):
        
        name = f'Stoch{output_type.replace("%", "")}_{k_window}_{d_window}'
        super().__init__(name=name)
        
        self.k_window = k_window
        self.d_window = d_window
        self.output_type = output_type
        self.description = f"Stochastic Oscillator {output_type} ({k_window}/{d_window})"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算随机振荡器
        
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA(%K, d_window)
        """
        if not self._validate_ohlc_data(price_data):
            logger.warning("Stochastic需要OHLC数据，回退到使用收盘价")
            close = self._get_price_series(price_data, 'close')
            high = close
            low = close
        else:
            high = self._get_price_series(price_data, 'high')
            low = self._get_price_series(price_data, 'low')
            close = self._get_price_series(price_data, 'close')
        
        def _calc_stochastic(stock_data):
            """计算单只股票的随机振荡器"""
            stock_high, stock_low, stock_close = stock_data
            
            # 计算滚动最高最低价
            lowest_low = stock_low.rolling(window=self.k_window, min_periods=1).min()
            highest_high = stock_high.rolling(window=self.k_window, min_periods=1).max()
            
            # 计算%K
            k_percent = (stock_close - lowest_low) / (highest_high - lowest_low) * 100
            k_percent = k_percent.where(highest_high != lowest_low, 50)  # 处理除零错误
            
            if self.output_type == '%K':
                return k_percent
            else:  # %D
                d_percent = k_percent.rolling(window=self.d_window, min_periods=1).mean()
                return d_percent
        
        # 按股票分组计算
        grouped_data = pd.concat([high, low, close], axis=1, keys=['high', 'low', 'close'])
        result = grouped_data.groupby(level='StockCodes').apply(
            lambda x: _calc_stochastic((x['high'], x['low'], x['close']))
        )
        
        # 预处理（已经在0-100范围内）
        result = self.preprocess(result, standardize=False)
        
        return result


class ROCFactor(OscillatorFactorBase):
    """ROC因子 - 变化率指标"""
    
    def __init__(self, window: int = 12):
        super().__init__(name=f'ROC_{window}')
        self.window = window
        self.description = f"Rate of Change over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ROC指标
        
        ROC = (Close - Close[n days ago]) / Close[n days ago] * 100
        """
        close_price = self._get_price_series(price_data, 'close')
        
        def _calc_roc(stock_prices):
            """计算单只股票的ROC"""
            lagged_price = stock_prices.shift(self.window)
            roc = (stock_prices - lagged_price) / lagged_price * 100
            return roc
        
        # 按股票分组计算
        roc = close_price.groupby(level='StockCodes', group_keys=False).apply(_calc_roc)
        
        # 预处理
        roc = self.preprocess(roc)
        
        return roc


class UltimateOscillatorFactor(OscillatorFactorBase):
    """终极振荡器因子 - Ultimate Oscillator"""
    
    def __init__(self, 
                 short_window: int = 7,
                 medium_window: int = 14, 
                 long_window: int = 28):
        
        name = f'UltimateOsc_{short_window}_{medium_window}_{long_window}'
        super().__init__(name=name)
        
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.description = f"Ultimate Oscillator ({short_window}/{medium_window}/{long_window})"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算终极振荡器
        
        BP = Close - min(Low, Prior Close)
        TR = max(High, Prior Close) - min(Low, Prior Close)
        UO = 100 * [(4*Avg7) + (2*Avg14) + Avg28] / (4+2+1)
        """
        if not self._validate_ohlc_data(price_data):
            logger.warning("Ultimate Oscillator需要OHLC数据，使用收盘价近似计算")
            close = self._get_price_series(price_data, 'close')
            high = close
            low = close
        else:
            high = self._get_price_series(price_data, 'high')
            low = self._get_price_series(price_data, 'low')
            close = self._get_price_series(price_data, 'close')
        
        def _calc_ultimate_oscillator(stock_data):
            """计算单只股票的终极振荡器"""
            stock_high, stock_low, stock_close = stock_data
            
            # 计算前一日收盘价
            prior_close = stock_close.shift(1)
            
            # 计算买压(BP)和真实波动幅度(TR)
            bp = stock_close - pd.concat([stock_low, prior_close], axis=1).min(axis=1)
            tr = pd.concat([
                stock_high,
                prior_close
            ], axis=1).max(axis=1) - pd.concat([
                stock_low, 
                prior_close
            ], axis=1).min(axis=1)
            
            # 计算三个周期的平均值
            def _calc_average(bp_series, tr_series, window):
                bp_sum = bp_series.rolling(window=window, min_periods=1).sum()
                tr_sum = tr_series.rolling(window=window, min_periods=1).sum()
                return bp_sum / tr_sum.where(tr_sum > 1e-8, 1)
            
            avg_short = _calc_average(bp, tr, self.short_window)
            avg_medium = _calc_average(bp, tr, self.medium_window) 
            avg_long = _calc_average(bp, tr, self.long_window)
            
            # 计算终极振荡器
            ultimate_osc = 100 * (4 * avg_short + 2 * avg_medium + avg_long) / 7
            
            return ultimate_osc
        
        # 按股票分组计算
        grouped_data = pd.concat([high, low, close], axis=1, keys=['high', 'low', 'close'])
        result = grouped_data.groupby(level='StockCodes').apply(
            lambda x: _calc_ultimate_oscillator((x['high'], x['low'], x['close']))
        )
        
        # 预处理（已经在0-100范围内）
        result = self.preprocess(result, standardize=False)
        
        return result


class MultiOscillatorFactory:
    """多振荡器因子批量生成器"""
    
    def __init__(self):
        self.factor_configs = {
            'MACD': {
                'class': MACDFactor,
                'params': [
                    {'fast_window': 12, 'slow_window': 26, 'signal_window': 9, 'output_type': 'macd'},
                    {'fast_window': 12, 'slow_window': 26, 'signal_window': 9, 'output_type': 'signal'},
                    {'fast_window': 12, 'slow_window': 26, 'signal_window': 9, 'output_type': 'histogram'},
                ]
            },
            'RSI': {
                'class': RSIFactor,
                'params': [
                    {'window': 6},
                    {'window': 14},
                    {'window': 21}
                ]
            },
            'WilliamsR': {
                'class': WilliamsRFactor,
                'params': [
                    {'window': 14},
                    {'window': 21}
                ]
            },
            'CCI': {
                'class': CCIFactor,
                'params': [
                    {'window': 20},
                    {'window': 14}
                ]
            },
            'Stochastic': {
                'class': StochasticOscillatorFactor,
                'params': [
                    {'k_window': 14, 'd_window': 3, 'output_type': '%K'},
                    {'k_window': 14, 'd_window': 3, 'output_type': '%D'},
                ]
            },
            'ROC': {
                'class': ROCFactor,
                'params': [
                    {'window': 12},
                    {'window': 20}
                ]
            },
            'UltimateOsc': {
                'class': UltimateOscillatorFactor,
                'params': [
                    {'short_window': 7, 'medium_window': 14, 'long_window': 28}
                ]
            }
        }
    
    def generate_oscillator_factors(self, 
                                  price_data: pd.DataFrame,
                                  factor_types: List[str] = None) -> Dict[str, pd.Series]:
        """
        批量生成振荡器因子
        
        Parameters:
        -----------
        price_data : DataFrame with MultiIndex [TradingDates, StockCodes]
        factor_types : 要生成的因子类型列表
        
        Returns:
        --------
        Dict[str, pd.Series] : 因子名称到因子值的映射
        """
        if factor_types is None:
            factor_types = ['MACD', 'RSI', 'WilliamsR', 'CCI']
            
        results = {}
        
        for factor_type in factor_types:
            if factor_type not in self.factor_configs:
                logger.warning(f"Unknown factor type: {factor_type}")
                continue
                
            factor_class = self.factor_configs[factor_type]['class']
            param_list = self.factor_configs[factor_type]['params']
            
            for params in param_list:
                try:
                    # 创建因子实例
                    factor = factor_class(**params)
                    
                    # 计算因子
                    factor_result = factor.calculate(price_data)
                    
                    # 保存结果
                    results[factor.name] = factor_result
                    
                    logger.info(f"Generated {factor.name}: {factor_result.count()} valid observations")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {factor_type} factor with params {params}: {e}")
                    continue
        
        return results
    
    def list_available_factors(self) -> Dict[str, List[str]]:
        """列出所有可用的振荡器因子"""
        available = {}
        
        for factor_type, config in self.factor_configs.items():
            factor_class = config['class']
            param_list = config['params']
            
            factor_names = []
            for params in param_list:
                try:
                    factor = factor_class(**params)
                    factor_names.append(factor.name)
                except Exception:
                    continue
                    
            available[factor_type] = factor_names
        
        return available