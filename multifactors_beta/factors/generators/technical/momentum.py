"""
动量因子计算器

提供各种动量因子的计算功能

包含:
- 价格动量 (1M/3M/6M/12M)
- 残差动量
- 行业动量
- 52周高点动量
"""

from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MomentumCalculator:
    """
    动量因子计算器

    计算各种时间周期的价格动量和相关衍生因子

    所有输入输出均使用MultiIndex格式 [TradingDates, StockCodes]

    Examples
    --------
    >>> from factors.generators.technical import MomentumCalculator
    >>>
    >>> # 计算1个月动量
    >>> mom_1m = MomentumCalculator.calculate_momentum_1m(close_prices)
    >>>
    >>> # 计算12个月动量（跳过最近1个月）
    >>> mom_12m = MomentumCalculator.calculate_momentum_12m_skip1m(close_prices)
    """

    # 交易日常量
    DAYS_1M = 21
    DAYS_3M = 63
    DAYS_6M = 126
    DAYS_12M = 252

    @staticmethod
    def calculate_momentum(close: pd.Series,
                          period: int = 20,
                          log_return: bool = False) -> pd.Series:
        """
        计算价格动量

        动量 = (当前价格 - N期前价格) / N期前价格

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        period : int
            回看周期（交易日），默认20
        log_return : bool
            是否使用对数收益率，默认False

        Returns
        -------
        pd.Series
            动量值，MultiIndex[TradingDates, StockCodes]

        Examples
        --------
        >>> momentum = MomentumCalculator.calculate_momentum(close, period=21)
        """
        if log_return:
            # 对数收益率
            momentum = close.groupby(level='StockCodes').apply(
                lambda x: np.log(x / x.shift(period))
            )
        else:
            # 简单收益率
            momentum = close.groupby(level='StockCodes').pct_change(period)

        return momentum

    @staticmethod
    def calculate_momentum_1m(close: pd.Series) -> pd.Series:
        """
        计算1个月动量（约21个交易日）

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            1个月动量
        """
        return MomentumCalculator.calculate_momentum(close, MomentumCalculator.DAYS_1M)

    @staticmethod
    def calculate_momentum_3m(close: pd.Series) -> pd.Series:
        """
        计算3个月动量（约63个交易日）

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            3个月动量
        """
        return MomentumCalculator.calculate_momentum(close, MomentumCalculator.DAYS_3M)

    @staticmethod
    def calculate_momentum_6m(close: pd.Series) -> pd.Series:
        """
        计算6个月动量（约126个交易日）

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            6个月动量
        """
        return MomentumCalculator.calculate_momentum(close, MomentumCalculator.DAYS_6M)

    @staticmethod
    def calculate_momentum_12m(close: pd.Series) -> pd.Series:
        """
        计算12个月动量（约252个交易日）

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            12个月动量
        """
        return MomentumCalculator.calculate_momentum(close, MomentumCalculator.DAYS_12M)

    @staticmethod
    def calculate_momentum_12m_skip1m(close: pd.Series) -> pd.Series:
        """
        计算12个月动量（跳过最近1个月）

        这是经典的动量因子，排除短期反转效应

        动量 = P(t-21) / P(t-252) - 1

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            12个月动量（跳过最近1个月）
        """
        def _calculate_skip_momentum(group):
            """计算跳过短期的动量"""
            # t-21期价格
            price_1m_ago = group.shift(MomentumCalculator.DAYS_1M)
            # t-252期价格
            price_12m_ago = group.shift(MomentumCalculator.DAYS_12M)
            # 动量
            return price_1m_ago / price_12m_ago - 1

        momentum = close.groupby(level='StockCodes').apply(_calculate_skip_momentum)

        # 处理MultiIndex
        if isinstance(momentum.index, pd.MultiIndex) and momentum.index.nlevels > 2:
            momentum = momentum.droplevel(0)

        return momentum

    @staticmethod
    def calculate_momentum_52w_high(close: pd.Series,
                                    high: pd.Series) -> pd.Series:
        """
        计算52周高点动量

        52周高点动量 = 当前价格 / 52周最高价 - 1

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        high : pd.Series
            最高价，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            52周高点动量
        """
        # 52周最高价
        high_52w = high.groupby(level='StockCodes').rolling(
            MomentumCalculator.DAYS_12M, min_periods=1
        ).max()

        # 处理MultiIndex
        if isinstance(high_52w.index, pd.MultiIndex) and high_52w.index.nlevels > 2:
            high_52w = high_52w.droplevel(0)

        # 距离52周高点的幅度
        momentum = close / high_52w - 1

        return momentum

    @staticmethod
    def calculate_acceleration(close: pd.Series,
                              short_period: int = 21,
                              long_period: int = 63) -> pd.Series:
        """
        计算动量加速度

        动量加速度 = 短期动量 - 长期动量

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        short_period : int
            短期周期，默认21（1个月）
        long_period : int
            长期周期，默认63（3个月）

        Returns
        -------
        pd.Series
            动量加速度
        """
        short_mom = MomentumCalculator.calculate_momentum(close, short_period)
        long_mom = MomentumCalculator.calculate_momentum(close, long_period)

        acceleration = short_mom - long_mom

        return acceleration

    @staticmethod
    def calculate_momentum_consistency(close: pd.Series,
                                      period: int = 252,
                                      sub_period: int = 21) -> pd.Series:
        """
        计算动量一致性

        衡量在period期间内，正收益的子期数占比

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        period : int
            总周期，默认252（12个月）
        sub_period : int
            子周期，默认21（1个月）

        Returns
        -------
        pd.Series
            动量一致性（0-1之间）
        """
        # 计算子周期收益率
        sub_returns = close.groupby(level='StockCodes').pct_change(sub_period)

        # 正收益标记
        positive = (sub_returns > 0).astype(float)

        # 计算period内的正收益子周期占比
        n_sub_periods = period // sub_period

        consistency = positive.groupby(level='StockCodes').rolling(
            period, min_periods=n_sub_periods
        ).mean()

        if isinstance(consistency.index, pd.MultiIndex) and consistency.index.nlevels > 2:
            consistency = consistency.droplevel(0)

        return consistency

    @staticmethod
    def calculate_risk_adjusted_momentum(close: pd.Series,
                                        period: int = 252) -> pd.Series:
        """
        计算风险调整后动量

        风险调整动量 = 期间收益率 / 期间波动率

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        period : int
            周期，默认252

        Returns
        -------
        pd.Series
            风险调整后动量
        """
        # 日收益率
        daily_returns = close.groupby(level='StockCodes').pct_change()

        # 期间总收益
        total_return = close.groupby(level='StockCodes').pct_change(period)

        # 期间波动率
        volatility = daily_returns.groupby(level='StockCodes').rolling(
            period, min_periods=period // 2
        ).std()

        if isinstance(volatility.index, pd.MultiIndex) and volatility.index.nlevels > 2:
            volatility = volatility.droplevel(0)

        # 年化调整
        volatility_annualized = volatility * np.sqrt(252)

        # 风险调整动量
        risk_adj_momentum = total_return / volatility_annualized

        return risk_adj_momentum
