"""
反转因子计算器

提供短期和中期反转因子的计算功能

包含:
- 短期反转 (5-20天)
- 中期反转 (1-3个月)
- 异常成交量反转
"""

from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ReversalCalculator:
    """
    反转因子计算器

    计算各种时间周期的反转因子

    反转因子的逻辑是：近期表现差的股票未来可能表现好（均值回归）

    所有输入输出均使用MultiIndex格式 [TradingDates, StockCodes]

    Examples
    --------
    >>> from factors.generators.technical import ReversalCalculator
    >>>
    >>> # 计算短期反转
    >>> reversal_5d = ReversalCalculator.calculate_short_term_reversal(close, 5)
    >>>
    >>> # 计算异常成交量反转
    >>> vol_reversal = ReversalCalculator.calculate_volume_reversal(close, volume)
    """

    # 交易日常量
    DAYS_1W = 5
    DAYS_2W = 10
    DAYS_1M = 21
    DAYS_3M = 63

    @staticmethod
    def calculate_short_term_reversal(close: pd.Series,
                                      period: int = 5) -> pd.Series:
        """
        计算短期反转因子

        短期反转 = -1 * 过去N日收益率

        负号表示：过去收益高的股票，反转因子值低

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        period : int
            回看周期，默认5天

        Returns
        -------
        pd.Series
            短期反转因子
        """
        # 过去N日收益率
        past_return = close.groupby(level='StockCodes').pct_change(period)

        # 反转因子（取负）
        reversal = -1 * past_return

        return reversal

    @staticmethod
    def calculate_reversal_1w(close: pd.Series) -> pd.Series:
        """
        计算1周反转因子

        Parameters
        ----------
        close : pd.Series
            收盘价

        Returns
        -------
        pd.Series
            1周反转因子
        """
        return ReversalCalculator.calculate_short_term_reversal(
            close, ReversalCalculator.DAYS_1W
        )

    @staticmethod
    def calculate_reversal_2w(close: pd.Series) -> pd.Series:
        """
        计算2周反转因子

        Parameters
        ----------
        close : pd.Series
            收盘价

        Returns
        -------
        pd.Series
            2周反转因子
        """
        return ReversalCalculator.calculate_short_term_reversal(
            close, ReversalCalculator.DAYS_2W
        )

    @staticmethod
    def calculate_reversal_1m(close: pd.Series) -> pd.Series:
        """
        计算1个月反转因子

        Parameters
        ----------
        close : pd.Series
            收盘价

        Returns
        -------
        pd.Series
            1个月反转因子
        """
        return ReversalCalculator.calculate_short_term_reversal(
            close, ReversalCalculator.DAYS_1M
        )

    @staticmethod
    def calculate_medium_term_reversal(close: pd.Series,
                                       period: int = 63) -> pd.Series:
        """
        计算中期反转因子

        中期反转因子考虑更长的均值回归周期

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        period : int
            回看周期，默认63天（3个月）

        Returns
        -------
        pd.Series
            中期反转因子
        """
        return ReversalCalculator.calculate_short_term_reversal(close, period)

    @staticmethod
    def calculate_volume_reversal(close: pd.Series,
                                 volume: pd.Series,
                                 return_period: int = 5,
                                 volume_period: int = 20) -> pd.Series:
        """
        计算成交量加权反转因子

        在高成交量时期的价格变动更可能反转

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        volume : pd.Series
            成交量，MultiIndex[TradingDates, StockCodes]
        return_period : int
            收益率计算周期，默认5天
        volume_period : int
            成交量参考周期，默认20天

        Returns
        -------
        pd.Series
            成交量加权反转因子
        """
        # 计算收益率
        returns = close.groupby(level='StockCodes').pct_change(return_period)

        # 计算相对成交量（当前vs平均）
        avg_volume = volume.groupby(level='StockCodes').rolling(
            volume_period, min_periods=1
        ).mean()

        if isinstance(avg_volume.index, pd.MultiIndex) and avg_volume.index.nlevels > 2:
            avg_volume = avg_volume.droplevel(0)

        relative_volume = volume / avg_volume

        # 成交量加权反转
        # 高成交量时期的下跌更可能反转
        volume_reversal = -1 * returns * relative_volume

        return volume_reversal

    @staticmethod
    def calculate_abnormal_volume_reversal(close: pd.Series,
                                          volume: pd.Series,
                                          return_period: int = 5,
                                          volume_zscore_threshold: float = 2.0) -> pd.Series:
        """
        计算异常成交量反转因子

        只在成交量异常高的时期计算反转

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        volume : pd.Series
            成交量，MultiIndex[TradingDates, StockCodes]
        return_period : int
            收益率计算周期，默认5天
        volume_zscore_threshold : float
            成交量Z分数阈值，默认2.0

        Returns
        -------
        pd.Series
            异常成交量反转因子
        """
        # 计算收益率
        returns = close.groupby(level='StockCodes').pct_change(return_period)

        # 计算成交量Z分数
        volume_mean = volume.groupby(level='StockCodes').rolling(60).mean()
        volume_std = volume.groupby(level='StockCodes').rolling(60).std()

        if isinstance(volume_mean.index, pd.MultiIndex) and volume_mean.index.nlevels > 2:
            volume_mean = volume_mean.droplevel(0)
            volume_std = volume_std.droplevel(0)

        volume_zscore = (volume - volume_mean) / volume_std

        # 只在高成交量时期计算反转
        abnormal_volume_mask = volume_zscore > volume_zscore_threshold

        # 异常成交量反转
        abnormal_reversal = -1 * returns
        abnormal_reversal = abnormal_reversal.where(abnormal_volume_mask, 0)

        return abnormal_reversal

    @staticmethod
    def calculate_industry_adjusted_reversal(close: pd.Series,
                                            industry: pd.Series,
                                            period: int = 5) -> pd.Series:
        """
        计算行业调整反转因子

        去除行业收益后的个股反转

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        industry : pd.Series
            行业分类，MultiIndex[TradingDates, StockCodes]
        period : int
            回看周期，默认5天

        Returns
        -------
        pd.Series
            行业调整反转因子
        """
        # 计算个股收益率
        stock_returns = close.groupby(level='StockCodes').pct_change(period)

        # 计算行业平均收益
        # 首先按日期和行业分组
        combined = pd.DataFrame({
            'return': stock_returns,
            'industry': industry
        })

        industry_returns = combined.groupby(
            [combined.index.get_level_values(0), 'industry']
        )['return'].transform('mean')

        # 行业调整后的收益
        adjusted_returns = stock_returns - industry_returns

        # 反转
        industry_adj_reversal = -1 * adjusted_returns

        return industry_adj_reversal

    @staticmethod
    def calculate_max_drawdown_reversal(close: pd.Series,
                                        period: int = 21) -> pd.Series:
        """
        计算最大回撤反转因子

        经历大回撤的股票可能有反转机会

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        period : int
            回看周期，默认21天

        Returns
        -------
        pd.Series
            最大回撤反转因子（正值表示回撤大）
        """
        def _calculate_max_drawdown(prices):
            """计算滚动最大回撤"""
            rolling_max = prices.rolling(period, min_periods=1).max()
            drawdown = (prices - rolling_max) / rolling_max
            return drawdown.rolling(period, min_periods=1).min()

        max_drawdown = close.groupby(level='StockCodes').apply(_calculate_max_drawdown)

        if isinstance(max_drawdown.index, pd.MultiIndex) and max_drawdown.index.nlevels > 2:
            max_drawdown = max_drawdown.droplevel(0)

        # 回撤越大（越负），反转因子越高
        reversal = -1 * max_drawdown

        return reversal
