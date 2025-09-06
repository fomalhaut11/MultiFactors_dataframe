"""
价格相关技术因子计算模块 - 优化版动量因子族
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import logging
from functools import lru_cache

from ...base.factor_base import FactorBase
from ...utils.data_loader import FactorDataLoader
from .indicators import MovingAverageCalculator, TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumFactorBase(FactorBase):
    """动量因子基类 - 提供通用的收益率计算和复权处理"""
    
    def __init__(self, name: str, window: int, category: str = 'technical'):
        super().__init__(name=name, category=category)
        self.window = window
        
    def _get_daily_returns(self, return_type: str = 'o2o') -> pd.Series:
        """
        获取预处理的日收益率数据
        
        Parameters:
        -----------
        return_type : str
            收益率类型 ('o2o' 或 'vwap')
        
        Returns:
        --------
        pd.Series : 日收益率数据（已复权处理）
        """
        try:
            return FactorDataLoader.load_returns('daily', return_type)
        except Exception as e:
            logger.error(f"无法加载预处理的收益率数据: {e}")
            raise RuntimeError("请先运行数据预处理脚本生成收益率数据")
    
    def _calculate_cumulative_returns(self, 
                                    returns: pd.Series, 
                                    window: int,
                                    min_periods: Optional[int] = None) -> pd.Series:
        """
        计算累积收益率（向量化实现）
        
        Parameters:
        -----------
        returns : 日收益率序列
        window : 累积窗口
        min_periods : 最小有效期数
        
        Returns:
        --------
        pd.Series : 累积收益率
        """
        if min_periods is None:
            min_periods = max(1, window // 2)  # 至少要有一半的数据
        
        # 使用rolling sum计算累积对数收益率（比shift方法更高效）
        cumulative_log_returns = returns.groupby(level='StockCodes').rolling(
            window=window, min_periods=min_periods
        ).sum()
        
        # 修复索引
        cumulative_log_returns.index = cumulative_log_returns.index.droplevel(0)
        
        return cumulative_log_returns


class MomentumFactor(MomentumFactorBase):
    """优化版动量因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name=f'Momentum_{window}d', window=window)
        self.description = f"Price momentum over {window} trading days (adjusted for splits & dividends)"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算动量因子
        动量 = log(后复权收盘价_t / 后复权收盘价_{t-window})
        
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
        
        # 计算累积对数收益率并进行时序移位（一次groupby完成两个操作）
        # log(adj_close_t / adj_close_{t-window})，然后向前移位1期
        def _calc_momentum_with_shift(x):
            momentum = np.log(x / x.shift(self.window))
            return momentum.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        momentum = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_momentum_with_shift)
        
        # 预处理（去极值、标准化）
        momentum = self.preprocess(momentum)
        
        return momentum


class MultiPeriodMomentumFactory:
    """多周期动量因子批量生成器 - 性能优化版"""
    
    def __init__(self, periods: List[int] = [5, 10, 20, 60, 120]):
        """
        Parameters:
        -----------
        periods : 动量计算周期列表（交易日）
        """
        self.periods = sorted(periods)  # 确保从短到长排序
        
    def generate_momentum_factors(self, 
                                price_data: pd.DataFrame,
                                factor_type: str = 'standard') -> Dict[str, pd.Series]:
        """
        批量生成多周期动量因子
        
        Parameters:
        -----------
        price_data : DataFrame with MultiIndex [TradingDates, StockCodes]
        factor_type : 'standard', 'residual', 'risk_adjusted'
        
        Returns:
        --------
        Dict[str, pd.Series] : 因子名称到因子值的映射
        """
        # 加载预处理的日收益率数据（避免重复计算）
        try:
            daily_returns = FactorDataLoader.load_returns('daily', 'o2o')
            logger.info(f"加载日收益率数据成功: {daily_returns.shape}")
        except Exception as e:
            logger.error(f"无法加载预处理的收益率数据: {e}")
            raise RuntimeError("请先运行数据预处理脚本生成收益率数据")
        
        results = {}
        
        # 批量计算不同周期的动量
        for period in self.periods:
            factor_name = f'Momentum_{period}d'
            
            if factor_type == 'standard':
                # 标准动量
                momentum = temp_factor._calculate_cumulative_returns(daily_returns, period)
                momentum = temp_factor.preprocess(momentum)
                
            elif factor_type == 'residual':
                # 残差动量（去除市场效应）
                momentum = self._calculate_residual_momentum(daily_returns, period)
                
            elif factor_type == 'risk_adjusted':
                # 风险调整动量
                momentum = self._calculate_risk_adjusted_momentum(daily_returns, period)
                
            results[factor_name] = momentum
            logger.info(f"Generated {factor_name}: {momentum.count()} valid observations")
        
        return results
    
    def _calculate_residual_momentum(self, 
                                   daily_returns: pd.Series, 
                                   period: int) -> pd.Series:
        """计算残差动量（去除市场beta）"""
        # 计算个股动量 - 使用具体实现类
        temp_factor = MomentumFactor(window=period)
        stock_momentum = temp_factor._calculate_cumulative_returns(daily_returns, period)
        
        # 计算市场动量（等权重）
        market_returns = daily_returns.groupby(level='TradingDates').mean()
        market_momentum = market_returns.rolling(window=period).sum()
        
        # 将市场动量对齐到个股数据
        market_momentum_aligned = stock_momentum.index.to_frame()['TradingDates'].map(market_momentum).values
        
        # 计算beta（使用60日滚动窗口）
        def _calc_beta(stock_returns):
            beta_window = min(60, period * 2)  # beta计算窗口
            stock_ret_series = stock_returns.reindex(market_returns.index, method='ffill')
            
            rolling_cov = stock_ret_series.rolling(beta_window).cov(market_returns)
            rolling_var = market_returns.rolling(beta_window).var()
            beta = rolling_cov / rolling_var
            
            return beta.fillna(1.0)  # 默认beta为1
        
        # 按股票计算beta
        betas = daily_returns.groupby(level='StockCodes', group_keys=False).apply(
            lambda x: _calc_beta(x.droplevel(1))
        )
        
        # 计算残差动量
        residual_momentum = stock_momentum - betas * pd.Series(
            market_momentum_aligned, 
            index=stock_momentum.index
        )
        
        # 预处理
        temp_factor = MomentumFactor(window=period)
        residual_momentum = temp_factor.preprocess(residual_momentum)
        
        return residual_momentum
    
    def _calculate_risk_adjusted_momentum(self, 
                                        daily_returns: pd.Series, 
                                        period: int) -> pd.Series:
        """计算风险调整动量（动量/波动率）"""
        # 计算动量
        temp_factor = MomentumFactor(window=period)
        momentum = temp_factor._calculate_cumulative_returns(daily_returns, period)
        
        # 计算同期波动率
        volatility = daily_returns.groupby(level='StockCodes').rolling(
            window=period, min_periods=period//2
        ).std()
        volatility.index = volatility.index.droplevel(0)
        
        # 年化波动率
        volatility = volatility * np.sqrt(252)
        
        # 计算风险调整动量（信息比率的简化版本）
        risk_adjusted_momentum = momentum / volatility.where(volatility > 0.01, np.nan)  # 避免除零
        
        # 预处理
        risk_adjusted_momentum = temp_factor.preprocess(risk_adjusted_momentum)
        
        return risk_adjusted_momentum


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
        反转 = -1 * log(后复权收盘价_t / 后复权收盘价_{t-n})
        
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
        
        # 计算反转（负的短期对数收益率）并进行时序移位（一次groupby完成）
        def _calc_reversal_with_shift(x):
            reversal = -1 * np.log(x / x.shift(self.window))
            return reversal.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        reversal = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_reversal_with_shift)
        
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
        MA因子 = MA_short(后复权收盘价) / MA_long(后复权收盘价) - 1
        
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
        
        # 计算移动平均因子并进行时序移位（一次groupby完成所有操作）
        def _calc_ma_factor_with_shift(x):
            ma_short = MovingAverageCalculator.simple_moving_average(x, self.short_window)
            ma_long = MovingAverageCalculator.simple_moving_average(x, self.long_window)
            ma_factor = ma_short / ma_long - 1
            return ma_factor.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        ma_factor = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_ma_factor_with_shift)
        
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
        基于后复权收盘价计算，避免除权除息影响
        
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
        
        # 计算RSI并进行时序移位（一次groupby完成）
        def _calc_rsi_with_shift(x):
            rsi = TechnicalIndicators.rsi(x, window=self.window)
            return rsi.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        rsi = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_rsi_with_shift)
        
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
        BB位置 = (后复权Price - Middle) / (Upper - Lower)
        
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
        
        # 计算布林带并进行时序移位（一次groupby完成）
        def _calc_bb_position_with_shift(price_series):
            middle, upper, lower = TechnicalIndicators.bollinger_bands(
                price_series, 
                window=self.window, 
                num_std=self.num_std
            )
            # 计算位置
            bb_position = (price_series - middle) / (upper - lower).replace(0, np.nan)
            return bb_position.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        bb_factor = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_bb_position_with_shift)
        
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
        位置 = (后复权Price - Min) / (Max - Min)
        
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
        
        # 计算价格位置并进行时序移位（一次groupby完成所有操作）
        def _calc_position_with_shift(x):
            rolling_max = x.rolling(window=self.window, min_periods=self.window//2).max()
            rolling_min = x.rolling(window=self.window, min_periods=self.window//2).min()
            position = (x - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)
            return position.shift(1)  # 时序移位：T日计算的因子在T+1日可用
        
        price_position = adj_close.groupby(level='StockCodes', group_keys=False).apply(_calc_position_with_shift)
        
        # 预处理
        price_position = self.preprocess(price_position, standardize=False)  # 已经在[0,1]范围
        
        return price_position