"""
成交量因子计算器

提供各种成交量相关因子的计算功能

包含:
- VWAP (成交量加权平均价格)
- OBV (能量潮指标)
- MFI (资金流量指标)
- 成交量比率
- 换手率因子
"""

from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VolumeCalculator:
    """
    成交量因子计算器

    计算各种成交量相关的技术因子

    所有输入输出均使用MultiIndex格式 [TradingDates, StockCodes]

    Examples
    --------
    >>> from factors.generators.technical import VolumeCalculator
    >>>
    >>> # 计算VWAP
    >>> vwap = VolumeCalculator.calculate_vwap(high, low, close, volume, period=20)
    >>>
    >>> # 计算OBV
    >>> obv = VolumeCalculator.calculate_obv(close, volume)
    """

    @staticmethod
    def calculate_vwap(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series,
                      period: int = 20) -> pd.Series:
        """
        计算成交量加权平均价格 (VWAP)

        VWAP = sum(典型价格 * 成交量) / sum(成交量)
        典型价格 = (最高价 + 最低价 + 收盘价) / 3

        Parameters
        ----------
        high : pd.Series
            最高价，MultiIndex[TradingDates, StockCodes]
        low : pd.Series
            最低价，MultiIndex[TradingDates, StockCodes]
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        volume : pd.Series
            成交量，MultiIndex[TradingDates, StockCodes]
        period : int
            计算周期，默认20

        Returns
        -------
        pd.Series
            VWAP值
        """
        # 典型价格
        typical_price = (high + low + close) / 3

        # 价格*成交量
        pv = typical_price * volume

        # 滚动求和
        pv_sum = pv.groupby(level='StockCodes').rolling(period, min_periods=1).sum()
        vol_sum = volume.groupby(level='StockCodes').rolling(period, min_periods=1).sum()

        # 处理MultiIndex
        if isinstance(pv_sum.index, pd.MultiIndex) and pv_sum.index.nlevels > 2:
            pv_sum = pv_sum.droplevel(0)
            vol_sum = vol_sum.droplevel(0)

        # VWAP
        vwap = pv_sum / vol_sum

        return vwap

    @staticmethod
    def calculate_vwap_deviation(high: pd.Series,
                                low: pd.Series,
                                close: pd.Series,
                                volume: pd.Series,
                                period: int = 20) -> pd.Series:
        """
        计算价格相对VWAP的偏离

        偏离 = (收盘价 - VWAP) / VWAP

        Parameters
        ----------
        high, low, close, volume : pd.Series
            价格和成交量数据
        period : int
            VWAP计算周期

        Returns
        -------
        pd.Series
            VWAP偏离度
        """
        vwap = VolumeCalculator.calculate_vwap(high, low, close, volume, period)
        deviation = (close - vwap) / vwap

        return deviation

    @staticmethod
    def calculate_obv(close: pd.Series,
                     volume: pd.Series) -> pd.Series:
        """
        计算能量潮指标 (OBV - On Balance Volume)

        OBV = 累计(如果收盘价上涨则+成交量，否则-成交量)

        Parameters
        ----------
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        volume : pd.Series
            成交量，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            OBV值
        """
        # 价格变化方向
        price_change = close.groupby(level='StockCodes').diff()

        # 带符号的成交量
        signed_volume = volume * np.sign(price_change)

        # 累计求和
        obv = signed_volume.groupby(level='StockCodes').cumsum()

        return obv

    @staticmethod
    def calculate_obv_ratio(close: pd.Series,
                           volume: pd.Series,
                           period: int = 20) -> pd.Series:
        """
        计算OBV变化率

        OBV变化率 = OBV的N期变化 / abs(OBV的N期前值)

        Parameters
        ----------
        close : pd.Series
            收盘价
        volume : pd.Series
            成交量
        period : int
            计算周期

        Returns
        -------
        pd.Series
            OBV变化率
        """
        obv = VolumeCalculator.calculate_obv(close, volume)

        obv_change = obv.groupby(level='StockCodes').diff(period)
        obv_lag = obv.groupby(level='StockCodes').shift(period).abs()

        obv_ratio = obv_change / obv_lag.replace(0, np.nan)

        return obv_ratio

    @staticmethod
    def calculate_mfi(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     volume: pd.Series,
                     period: int = 14) -> pd.Series:
        """
        计算资金流量指标 (MFI - Money Flow Index)

        MFI是成交量加权的RSI，范围0-100

        Parameters
        ----------
        high : pd.Series
            最高价，MultiIndex[TradingDates, StockCodes]
        low : pd.Series
            最低价，MultiIndex[TradingDates, StockCodes]
        close : pd.Series
            收盘价，MultiIndex[TradingDates, StockCodes]
        volume : pd.Series
            成交量，MultiIndex[TradingDates, StockCodes]
        period : int
            计算周期，默认14

        Returns
        -------
        pd.Series
            MFI值 (0-100)
        """
        # 典型价格
        typical_price = (high + low + close) / 3

        # 资金流量
        money_flow = typical_price * volume

        # 价格变化
        price_change = typical_price.groupby(level='StockCodes').diff()

        # 正向和负向资金流
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)

        # 滚动求和
        positive_sum = positive_flow.groupby(level='StockCodes').rolling(
            period, min_periods=1
        ).sum()
        negative_sum = negative_flow.groupby(level='StockCodes').rolling(
            period, min_periods=1
        ).sum()

        # 处理MultiIndex
        if isinstance(positive_sum.index, pd.MultiIndex) and positive_sum.index.nlevels > 2:
            positive_sum = positive_sum.droplevel(0)
            negative_sum = negative_sum.droplevel(0)

        # MFI
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series,
                              period: int = 20) -> pd.Series:
        """
        计算成交量比率

        成交量比率 = 当日成交量 / N日平均成交量

        Parameters
        ----------
        volume : pd.Series
            成交量，MultiIndex[TradingDates, StockCodes]
        period : int
            平均周期，默认20

        Returns
        -------
        pd.Series
            成交量比率
        """
        avg_volume = volume.groupby(level='StockCodes').rolling(
            period, min_periods=1
        ).mean()

        if isinstance(avg_volume.index, pd.MultiIndex) and avg_volume.index.nlevels > 2:
            avg_volume = avg_volume.droplevel(0)

        volume_ratio = volume / avg_volume

        return volume_ratio

    @staticmethod
    def calculate_volume_ma_deviation(volume: pd.Series,
                                     short_period: int = 5,
                                     long_period: int = 20) -> pd.Series:
        """
        计算成交量均线偏离

        偏离 = 短期成交量均值 / 长期成交量均值 - 1

        Parameters
        ----------
        volume : pd.Series
            成交量
        short_period : int
            短期周期，默认5
        long_period : int
            长期周期，默认20

        Returns
        -------
        pd.Series
            成交量均线偏离
        """
        short_ma = volume.groupby(level='StockCodes').rolling(
            short_period, min_periods=1
        ).mean()
        long_ma = volume.groupby(level='StockCodes').rolling(
            long_period, min_periods=1
        ).mean()

        if isinstance(short_ma.index, pd.MultiIndex) and short_ma.index.nlevels > 2:
            short_ma = short_ma.droplevel(0)
            long_ma = long_ma.droplevel(0)

        deviation = short_ma / long_ma - 1

        return deviation

    @staticmethod
    def calculate_turnover_rate(volume: pd.Series,
                               shares_outstanding: pd.Series) -> pd.Series:
        """
        计算换手率

        换手率 = 成交量 / 流通股本

        Parameters
        ----------
        volume : pd.Series
            成交量（股数），MultiIndex[TradingDates, StockCodes]
        shares_outstanding : pd.Series
            流通股本，MultiIndex[TradingDates, StockCodes]

        Returns
        -------
        pd.Series
            换手率
        """
        turnover = volume / shares_outstanding

        return turnover

    @staticmethod
    def calculate_avg_turnover(volume: pd.Series,
                              shares_outstanding: pd.Series,
                              period: int = 20) -> pd.Series:
        """
        计算平均换手率

        Parameters
        ----------
        volume : pd.Series
            成交量
        shares_outstanding : pd.Series
            流通股本
        period : int
            平均周期

        Returns
        -------
        pd.Series
            平均换手率
        """
        turnover = VolumeCalculator.calculate_turnover_rate(volume, shares_outstanding)

        avg_turnover = turnover.groupby(level='StockCodes').rolling(
            period, min_periods=1
        ).mean()

        if isinstance(avg_turnover.index, pd.MultiIndex) and avg_turnover.index.nlevels > 2:
            avg_turnover = avg_turnover.droplevel(0)

        return avg_turnover

    @staticmethod
    def calculate_price_volume_correlation(close: pd.Series,
                                          volume: pd.Series,
                                          period: int = 20) -> pd.Series:
        """
        计算价格-成交量相关性

        Parameters
        ----------
        close : pd.Series
            收盘价
        volume : pd.Series
            成交量
        period : int
            计算周期

        Returns
        -------
        pd.Series
            价格-成交量相关系数
        """
        # 计算收益率
        returns = close.groupby(level='StockCodes').pct_change()

        # 合并数据计算相关性
        combined = pd.DataFrame({'returns': returns, 'volume': volume})

        def rolling_corr(group):
            return group['returns'].rolling(period).corr(group['volume'])

        correlation = combined.groupby(level='StockCodes').apply(rolling_corr)

        if isinstance(correlation.index, pd.MultiIndex) and correlation.index.nlevels > 2:
            correlation = correlation.droplevel(0)

        return correlation

    @staticmethod
    def calculate_volume_momentum(volume: pd.Series,
                                 period: int = 20) -> pd.Series:
        """
        计算成交量动量

        成交量动量 = 当前成交量 / N期前成交量 - 1

        Parameters
        ----------
        volume : pd.Series
            成交量
        period : int
            回看周期

        Returns
        -------
        pd.Series
            成交量动量
        """
        volume_momentum = volume.groupby(level='StockCodes').pct_change(period)

        return volume_momentum

    @staticmethod
    def calculate_accumulation_distribution(high: pd.Series,
                                           low: pd.Series,
                                           close: pd.Series,
                                           volume: pd.Series) -> pd.Series:
        """
        计算累积/派发指标 (A/D Line)

        CLV = ((收盘-最低) - (最高-收盘)) / (最高-最低)
        A/D = 累计(CLV * 成交量)

        Parameters
        ----------
        high, low, close : pd.Series
            价格数据
        volume : pd.Series
            成交量

        Returns
        -------
        pd.Series
            A/D指标
        """
        # 价格范围
        price_range = high - low

        # 收盘价位置值 (Close Location Value)
        clv = ((close - low) - (high - close)) / price_range.replace(0, np.nan)

        # A/D
        ad = (clv * volume).groupby(level='StockCodes').cumsum()

        return ad
