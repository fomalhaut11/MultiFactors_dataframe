#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动量因子实现
包含经典动量、反转、趋势等技术分析因子
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

from ...base.factor_base import FactorBase
from ...base.data_processing_mixin import DataProcessingMixin
from ...operations.time_series import rolling_returns, lag_data
from ...operations.cross_sectional import rank_data, zscore_data

logger = logging.getLogger(__name__)


class MomentumFactor(FactorBase, DataProcessingMixin):
    """
    经典动量因子
    计算过去n期收益率，排除最近m期以避免短期反转效应
    """
    
    def __init__(self, lookback_days: int = 252, skip_days: int = 22):
        """
        Parameters:
        -----------
        lookback_days : int
            动量计算的回看天数，默认252天（1年）
        skip_days : int
            跳过的最近天数，默认22天（1月），避免短期反转
        """
        super().__init__(name=f'Momentum_{lookback_days}_{skip_days}', category='momentum')
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.description = f"过去{lookback_days}天动量因子，跳过最近{skip_days}天"
        
    def calculate(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        计算动量因子
        
        Parameters:
        -----------
        price_data : pd.Series
            价格数据，MultiIndex[TradingDates, StockCodes]
            
        Returns:
        --------
        pd.Series : 动量因子值
        """
        try:
            # 计算对数收益率
            log_prices = np.log(price_data)
            
            # 计算过去lookback_days天的累积收益率，但跳过最近skip_days天
            # 结束日期是skip_days天前
            end_lag = lag_data(log_prices, self.skip_days)
            # 开始日期是lookback_days + skip_days天前
            start_lag = lag_data(log_prices, self.lookback_days + self.skip_days)
            
            # 计算动量：(P_t-skip / P_t-lookback-skip) - 1
            momentum = end_lag - start_lag
            
            # 转换为截面排名
            momentum_rank = momentum.groupby(level='TradingDates').apply(
                lambda x: rank_data(x)
            ).droplevel(0)
            
            return momentum_rank.fillna(0.5)  # 缺失值填充为中性值
            
        except Exception as e:
            logger.error(f"计算动量因子失败: {e}")
            return pd.Series(dtype=float, index=price_data.index)


class ShortTermReversalFactor(FactorBase, DataProcessingMixin):
    """
    短期反转因子
    基于短期价格反转效应
    """
    
    def __init__(self, lookback_days: int = 22):
        """
        Parameters:
        -----------
        lookback_days : int
            反转计算的回看天数，默认22天（1月）
        """
        super().__init__(name=f'ShortReversal_{lookback_days}', category='reversal')
        self.lookback_days = lookback_days
        self.description = f"过去{lookback_days}天短期反转因子"
        
    def calculate(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """计算短期反转因子"""
        try:
            # 计算过去n天的累积收益率
            returns = rolling_returns(price_data, self.lookback_days)
            
            # 反转因子：收益率取负号
            reversal = -returns
            
            # 截面标准化
            reversal_zscore = reversal.groupby(level='TradingDates').apply(
                lambda x: zscore_data(x)
            ).droplevel(0)
            
            return reversal_zscore.fillna(0.0)
            
        except Exception as e:
            logger.error(f"计算短期反转因子失败: {e}")
            return pd.Series(dtype=float, index=price_data.index)


class LongTermReversalFactor(FactorBase, DataProcessingMixin):
    """
    长期均值回归因子
    基于长期价格均值回归效应
    """
    
    def __init__(self, lookback_days: int = 504):  # 2年
        """
        Parameters:
        -----------
        lookback_days : int
            长期回归计算的回看天数，默认504天（2年）
        """
        super().__init__(name=f'LongReversal_{lookback_days}', category='reversal')
        self.lookback_days = lookback_days
        self.description = f"过去{lookback_days}天长期均值回归因子"
        
    def calculate(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """计算长期均值回归因子"""
        try:
            # 计算对数价格
            log_prices = np.log(price_data)
            
            # 计算过去长期的平均价格
            long_term_avg = log_prices.groupby(level='StockCodes').rolling(
                window=self.lookback_days, min_periods=self.lookback_days//2
            ).mean().droplevel(0)
            
            # 当前价格相对于长期均值的偏离
            deviation = log_prices - long_term_avg
            
            # 均值回归因子：价格偏离取负号（高价格预期回归，给予负分）
            reversal = -deviation
            
            # 截面标准化
            reversal_zscore = reversal.groupby(level='TradingDates').apply(
                lambda x: zscore_data(x)
            ).droplevel(0)
            
            return reversal_zscore.fillna(0.0)
            
        except Exception as e:
            logger.error(f"计算长期均值回归因子失败: {e}")
            return pd.Series(dtype=float, index=price_data.index)


class TrendStrengthFactor(FactorBase, DataProcessingMixin):
    """
    趋势强度因子
    衡量价格趋势的持续性和强度
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 60):
        """
        Parameters:
        -----------
        short_window : int
            短期移动平均窗口，默认20天
        long_window : int
            长期移动平均窗口，默认60天
        """
        super().__init__(name=f'TrendStrength_{short_window}_{long_window}', category='momentum')
        self.short_window = short_window
        self.long_window = long_window
        self.description = f"趋势强度因子，基于{short_window}和{long_window}天移动平均"
        
    def calculate(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """计算趋势强度因子"""
        try:
            # 计算短期和长期移动平均
            ma_short = price_data.groupby(level='StockCodes').rolling(
                window=self.short_window, min_periods=self.short_window//2
            ).mean().droplevel(0)
            
            ma_long = price_data.groupby(level='StockCodes').rolling(
                window=self.long_window, min_periods=self.long_window//2
            ).mean().droplevel(0)
            
            # 趋势强度：短期MA相对于长期MA的偏离程度
            trend_strength = (ma_short - ma_long) / ma_long
            
            # 加入价格相对于短期MA的位置
            price_position = (price_data - ma_short) / ma_short
            
            # 综合趋势强度
            combined_trend = trend_strength + 0.5 * price_position
            
            # 截面标准化
            trend_zscore = combined_trend.groupby(level='TradingDates').apply(
                lambda x: zscore_data(x)
            ).droplevel(0)
            
            return trend_zscore.fillna(0.0)
            
        except Exception as e:
            logger.error(f"计算趋势强度因子失败: {e}")
            return pd.Series(dtype=float, index=price_data.index)


class PriceMomentumFactor(FactorBase, DataProcessingMixin):
    """
    价格动量因子（Jegadeesh and Titman风格）
    经典的12-1动量策略
    """
    
    def __init__(self):
        """标准的12-1动量因子"""
        super().__init__(name='PriceMomentum_12_1', category='momentum')
        self.description = "经典12-1价格动量因子"
        
    def calculate(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """计算12-1价格动量因子"""
        try:
            # 计算对数收益率
            log_prices = np.log(price_data)
            
            # 12个月前的价格（跳过最近1个月）
            price_12m_ago = lag_data(log_prices, 252 + 22)  # 12个月 + 1个月
            price_1m_ago = lag_data(log_prices, 22)  # 1个月前
            
            # 计算11个月的累积收益率（月份2-12）
            momentum_12_1 = price_1m_ago - price_12m_ago
            
            # 截面排名
            momentum_rank = momentum_12_1.groupby(level='TradingDates').apply(
                lambda x: rank_data(x)
            ).droplevel(0)
            
            return momentum_rank.fillna(0.5)
            
        except Exception as e:
            logger.error(f"计算12-1动量因子失败: {e}")
            return pd.Series(dtype=float, index=price_data.index)


class VolatilityAdjustedMomentumFactor(FactorBase, DataProcessingMixin):
    """
    波动率调整的动量因子
    对动量进行波动率风险调整
    """
    
    def __init__(self, return_window: int = 252, vol_window: int = 60):
        """
        Parameters:
        -----------
        return_window : int
            收益率计算窗口，默认252天
        vol_window : int
            波动率计算窗口，默认60天
        """
        super().__init__(name=f'VolAdjMom_{return_window}_{vol_window}', category='momentum')
        self.return_window = return_window
        self.vol_window = vol_window
        self.description = f"波动率调整动量因子，{return_window}天收益/{vol_window}天波动率"
        
    def calculate(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """计算波动率调整的动量因子"""
        try:
            # 计算日收益率
            returns = price_data.groupby(level='StockCodes').pct_change()
            
            # 计算累积收益率（跳过最近22天）
            cumulative_returns = rolling_returns(price_data, self.return_window, skip_days=22)
            
            # 计算波动率
            volatility = returns.groupby(level='StockCodes').rolling(
                window=self.vol_window, min_periods=self.vol_window//2
            ).std().droplevel(0) * np.sqrt(252)  # 年化波动率
            
            # 波动率调整的动量：收益率 / 波动率
            risk_adjusted_momentum = cumulative_returns / volatility
            
            # 截面标准化
            adjusted_momentum_zscore = risk_adjusted_momentum.groupby(level='TradingDates').apply(
                lambda x: zscore_data(x)
            ).droplevel(0)
            
            return adjusted_momentum_zscore.fillna(0.0)
            
        except Exception as e:
            logger.error(f"计算波动率调整动量因子失败: {e}")
            return pd.Series(dtype=float, index=price_data.index)