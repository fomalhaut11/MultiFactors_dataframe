#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波动率因子计算模块 - 优化版

实现多种波动率相关的技术因子，包括传统波动率和高级波动率模型
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Literal, Tuple, Union, List
import logging
from functools import lru_cache

from ...base.factor_base import FactorBase
from ...utils.data_loader import FactorDataLoader
from .indicators import VolatilityCalculator

logger = logging.getLogger(__name__)


class VolatilityFactorBase(FactorBase):
    """波动率因子基类"""
    
    def __init__(self, name: str, window: int, category: str = 'technical'):
        super().__init__(name=name, category=category)
        self.window = window
        self.description = f"Volatility factor with {window}-day window"
        
    def _get_daily_returns(self, return_type: str = 'o2o') -> pd.Series:
        """获取预处理的日收益率数据"""
        try:
            return FactorDataLoader.load_returns('daily', return_type)
        except Exception as e:
            logger.error(f"无法加载预处理的收益率数据: {e}")
            raise RuntimeError("请先运行数据预处理脚本生成收益率数据")
    
    def _annualize_volatility(self, volatility: pd.Series, trading_days_per_year: int = 252) -> pd.Series:
        """年化波动率"""
        return volatility * np.sqrt(trading_days_per_year)


class HistoricalVolatilityFactor(VolatilityFactorBase):
    """历史波动率因子 - 支持多种计算方法"""
    
    def __init__(self, 
                 window: int = 20,
                 method: Literal["simple", "parkinson", "garman_klass", "yang_zhang"] = "simple"):
        super().__init__(name=f'HistVol_{window}d_{method}', window=window)
        self.method = method
        self.description = f"{method.title()} historical volatility over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算历史波动率
        """
        if self.method == "simple":
            # 简单历史波动率 - 使用预处理的日收益率
            returns = self._get_daily_returns('o2o')
            volatility = returns.groupby(level=1).rolling(
                window=self.window, min_periods=self.window//2
            ).std()
            # 正确恢复MultiIndex格式
            volatility.index = volatility.index.droplevel(0)
            # 确保索引名称正确
            if hasattr(volatility.index, 'names') and len(volatility.index.names) == 2:
                volatility.index.names = ['TradingDates', 'StockCodes']
            
        else:
            # 使用高级方法（需要OHLC数据）
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in price_data.columns for col in required_cols):
                logger.warning(f"方法{self.method}需要OHLC数据，回退到简单方法")
                return self.calculate(price_data, method="simple")
                
            # 按股票分组计算
            def _calc_advanced_vol(stock_data):
                return VolatilityCalculator.historical_volatility(
                    stock_data, 
                    window=self.window,
                    method=self.method,
                    annualize=False  # 后续统一年化
                )
            
            volatility = price_data.groupby(level=1, group_keys=False).apply(_calc_advanced_vol)
        
        # 年化处理
        volatility = self._annualize_volatility(volatility)
        
        # 预处理
        volatility = self.preprocess(volatility)
        
        return volatility


class RealizedVolatilityFactor(VolatilityFactorBase):
    """已实现波动率因子"""
    
    def __init__(self, window: int = 20, intraday_periods: int = None):
        super().__init__(name=f'RealizedVol_{window}d', window=window)
        self.intraday_periods = intraday_periods or window
        self.description = f"Realized volatility over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算已实现波动率
        RV = sqrt(sum(r_t^2))，其中r_t是日内收益率
        """
        # 使用日收益率平方和近似已实现波动率
        returns = self._get_daily_returns('o2o')
        
        # 计算平方收益率
        squared_returns = returns ** 2
        
        # 滚动求和
        realized_var = squared_returns.groupby(level=1).rolling(
            window=self.window, min_periods=self.window//2
        ).sum()
        realized_var.index = realized_var.index.droplevel(0)
        
        # 计算已实现波动率
        realized_vol = np.sqrt(realized_var)
        
        # 年化处理
        realized_vol = self._annualize_volatility(realized_vol)
        
        # 预处理
        realized_vol = self.preprocess(realized_vol)
        
        return realized_vol


class DownsideVolatilityFactor(VolatilityFactorBase):
    """下行波动率因子（下行风险）"""
    
    def __init__(self, window: int = 20, threshold: float = 0.0):
        super().__init__(name=f'DownsideVol_{window}d', window=window)
        self.threshold = threshold
        self.description = f"Downside volatility over {window} days (threshold={threshold})"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算下行波动率
        只考虑低于阈值的收益率的波动
        """
        returns = self._get_daily_returns('o2o')
        
        # 只保留低于阈值的收益率，其他设为0
        downside_returns = returns.where(returns < self.threshold, 0)
        
        # 计算下行波动率
        def _calc_downside_vol(stock_returns):
            downside_vol = stock_returns.rolling(
                window=self.window, min_periods=self.window//2
            ).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
            return downside_vol
        
        downside_vol = downside_returns.groupby(level=1, group_keys=False).apply(_calc_downside_vol)
        
        # 年化处理
        downside_vol = self._annualize_volatility(downside_vol)
        
        # 预处理
        downside_vol = self.preprocess(downside_vol)
        
        return downside_vol


class GARCHVolatilityFactor(VolatilityFactorBase):
    """GARCH波动率因子 - 简化版GARCH(1,1)"""
    
    def __init__(self, window: int = 252, forecast_horizon: int = 1):
        super().__init__(name=f'GARCH_{window}d', window=window)
        self.forecast_horizon = forecast_horizon
        self.description = f"GARCH(1,1) volatility forecast over {window} days"
        
    def _estimate_garch_params(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        估计GARCH(1,1)参数
        简化版本，使用矩估计法
        """
        try:
            # 计算收益率序列统计量
            returns = returns[~np.isnan(returns)]
            if len(returns) < 30:  # 最少需要30个观测
                return 0.01, 0.05, 0.9  # 默认参数
                
            unconditional_var = np.var(returns)
            
            # 简化参数估计（基于经验值）
            omega = 0.01 * unconditional_var  # 常数项
            alpha = 0.05  # ARCH系数
            beta = 0.90   # GARCH系数
            
            # 确保参数约束
            if alpha + beta >= 1.0:
                alpha = 0.05
                beta = 0.90
                
            return omega, alpha, beta
            
        except Exception as e:
            logger.warning(f"GARCH参数估计失败: {e}")
            return 0.01, 0.05, 0.9
    
    def _forecast_garch_volatility(self, 
                                 returns: np.ndarray, 
                                 omega: float, 
                                 alpha: float, 
                                 beta: float) -> float:
        """
        GARCH波动率预测
        """
        try:
            returns = returns[~np.isnan(returns)]
            if len(returns) < 2:
                return np.nan
                
            # 初始化条件方差
            unconditional_var = omega / (1 - alpha - beta)
            conditional_var = unconditional_var
            
            # 递推计算条件方差
            for r in returns[-min(len(returns), 60):]:  # 最多使用最近60个观测
                conditional_var = omega + alpha * (r**2) + beta * conditional_var
            
            # h步预测
            forecast_var = conditional_var
            for _ in range(self.forecast_horizon - 1):
                forecast_var = omega + (alpha + beta) * forecast_var
                
            return np.sqrt(forecast_var) if forecast_var > 0 else np.nan
            
        except Exception as e:
            logger.warning(f"GARCH预测失败: {e}")
            return np.nan
    
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算GARCH波动率
        """
        returns = self._get_daily_returns('o2o')
        
        def _calc_garch_vol(stock_returns):
            """为单只股票计算GARCH波动率"""
            garch_vol = pd.Series(index=stock_returns.index, dtype=float)
            
            for i in range(len(stock_returns)):
                # 获取滚动窗口数据
                if i < self.window:
                    window_data = stock_returns.iloc[:i+1]
                else:
                    window_data = stock_returns.iloc[i-self.window+1:i+1]
                
                if len(window_data) < 30:  # 需要足够的数据
                    garch_vol.iloc[i] = np.nan
                    continue
                
                # 估计GARCH参数
                returns_array = window_data.values
                omega, alpha, beta = self._estimate_garch_params(returns_array)
                
                # 预测波动率
                forecast_vol = self._forecast_garch_volatility(returns_array, omega, alpha, beta)
                garch_vol.iloc[i] = forecast_vol
            
            return garch_vol
        
        # 按股票分组计算
        garch_volatility = returns.groupby(level=1, group_keys=False).apply(_calc_garch_vol)
        
        # 年化处理
        garch_volatility = self._annualize_volatility(garch_volatility)
        
        # 预处理
        garch_volatility = self.preprocess(garch_volatility)
        
        return garch_volatility


class VolatilityRatioFactor(FactorBase):
    """波动率比率因子"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name=f'VolRatio_{short_window}_{long_window}', category='technical')
        self.short_window = short_window
        self.long_window = long_window
        self.description = f"Volatility ratio ({short_window}d/{long_window}d)"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算波动率比率
        VolRatio = Vol(短期) / Vol(长期)
        """
        # 计算收益率
        returns = price_data['close'].groupby(level=1).pct_change()
        
        # 计算短期和长期波动率
        vol_short = returns.groupby(level=1).rolling(
            window=self.short_window, min_periods=max(1, self.short_window//2)
        ).std()
        
        vol_long = returns.groupby(level=1).rolling(
            window=self.long_window, min_periods=max(1, self.long_window//2)
        ).std()
        
        # 修复索引
        vol_short.index = vol_short.index.droplevel(0)
        vol_long.index = vol_long.index.droplevel(0)
        
        # 计算比率
        vol_ratio = vol_short / vol_long.where(vol_long > 1e-8, np.nan)
        
        # 预处理
        vol_ratio = self.preprocess(vol_ratio)
        
        return vol_ratio


class VolatilitySkewFactor(VolatilityFactorBase):
    """波动率偏度因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(name=f'VolSkew_{window}d', window=window)
        self.description = f"Volatility skewness over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算波动率偏度
        衡量收益率分布的非对称性
        """
        returns = self._get_daily_returns('o2o')
        
        # 计算滚动偏度
        def _calc_skew(x):
            """计算偏度"""
            if len(x) < 10:
                return np.nan
            x_clean = x.dropna()
            if len(x_clean) < 10:
                return np.nan
            return x_clean.skew()
        
        vol_skew = returns.groupby(level=1).rolling(
            window=self.window, min_periods=self.window//2
        ).apply(_calc_skew, raw=False)
        
        vol_skew.index = vol_skew.index.droplevel(0)
        
        # 预处理
        vol_skew = self.preprocess(vol_skew)
        
        return vol_skew


class VolatilityKurtosisFactor(VolatilityFactorBase):
    """波动率峰度因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(name=f'VolKurt_{window}d', window=window)
        self.description = f"Volatility kurtosis over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算波动率峰度
        衡量收益率分布的厚尾特征
        """
        returns = self._get_daily_returns('o2o')
        
        # 计算滚动峰度
        def _calc_kurtosis(x):
            """计算峰度"""
            if len(x) < 10:
                return np.nan
            x_clean = x.dropna()
            if len(x_clean) < 10:
                return np.nan
            return x_clean.kurtosis()
        
        vol_kurtosis = returns.groupby(level=1).rolling(
            window=self.window, min_periods=self.window//2
        ).apply(_calc_kurtosis, raw=False)
        
        vol_kurtosis.index = vol_kurtosis.index.droplevel(0)
        
        # 预处理
        vol_kurtosis = self.preprocess(vol_kurtosis)
        
        return vol_kurtosis


class MultiVolatilityFactory:
    """多波动率因子批量生成器"""
    
    def __init__(self, windows: List[int] = [5, 10, 20, 60]):
        self.windows = windows
        
    def generate_volatility_factors(self, 
                                  price_data: pd.DataFrame,
                                  factor_types: List[str] = None) -> Dict[str, pd.Series]:
        """
        批量生成波动率因子
        
        Parameters:
        -----------
        price_data : DataFrame with MultiIndex [TradingDates, StockCodes]
        factor_types : 要生成的因子类型列表
        
        Returns:
        --------
        Dict[str, pd.Series] : 因子名称到因子值的映射
        """
        if factor_types is None:
            factor_types = ['historical', 'realized', 'downside']
            
        results = {}
        
        # 生成不同类型的波动率因子
        for factor_type in factor_types:
            for window in self.windows:
                try:
                    if factor_type == 'historical':
                        factor = HistoricalVolatilityFactor(window=window, method='simple')
                    elif factor_type == 'realized':
                        factor = RealizedVolatilityFactor(window=window)
                    elif factor_type == 'downside':
                        factor = DownsideVolatilityFactor(window=window)
                    elif factor_type == 'garch' and window >= 60:  # GARCH需要更多数据
                        factor = GARCHVolatilityFactor(window=window)
                    else:
                        continue
                        
                    factor_result = factor.calculate(price_data)
                    results[factor.name] = factor_result
                    
                    logger.info(f"Generated {factor.name}: {factor_result.count()} valid observations")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {factor_type} volatility factor (window={window}): {e}")
                    continue
        
        # 生成波动率比率因子
        if 'ratio' in factor_types:
            try:
                ratio_factor = VolatilityRatioFactor(short_window=5, long_window=20)
                ratio_result = ratio_factor.calculate(price_data)
                results[ratio_factor.name] = ratio_result
                logger.info(f"Generated {ratio_factor.name}: {ratio_result.count()} valid observations")
            except Exception as e:
                logger.error(f"Failed to generate volatility ratio factor: {e}")
        
        return results


# 保持向后兼容的原始VolatilityFactor类
class VolatilityFactor(HistoricalVolatilityFactor):
    """
    原始波动率因子类（向后兼容）
    """
    
    def __init__(self, **kwargs):
        window = kwargs.get('window', 20)
        super().__init__(window=window, method='simple')
        # 保持原有的name以维持兼容性
        self.name = kwargs.get('name', 'VolatilityFactor')
        
    def calculate(self, data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
        """
        兼容原有接口的计算方法
        """
        # 更新窗口大小
        self.window = window
        return super().calculate(data, **kwargs)