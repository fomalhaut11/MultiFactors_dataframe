"""
Beta风险因子计算模块
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, List, Tuple
import logging

from ...base.factor_base import FactorBase
from ...utils.data_loader import FactorDataLoader

logger = logging.getLogger(__name__)


class BetaFactor(FactorBase):
    """Beta因子"""
    
    def __init__(self, 
                 window: int = 252,
                 min_periods: Optional[int] = None):
        super().__init__(name=f'Beta_{window}', category='risk')
        self.window = window
        self.min_periods = min_periods or window // 2
        self.description = f"Market beta over {window} days"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 benchmark_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算市场Beta
        
        Parameters:
        -----------
        price_data : 股票价格数据
        benchmark_data : 基准指数数据，包含'close'列
        """
        # 计算股票收益率
        stock_returns = self._get_daily_returns('o2o')
        
        # 计算基准收益率
        benchmark_returns = np.log(
            benchmark_data['close'] / benchmark_data['close'].shift(1)
        )
        
        # 将收益率转换为宽表格式
        stock_returns_wide = stock_returns.unstack(level='StockCodes')
        
        # 对齐数据
        aligned_dates = stock_returns_wide.index.intersection(benchmark_returns.index)
        stock_returns_wide = stock_returns_wide.loc[aligned_dates]
        benchmark_returns = benchmark_returns.loc[aligned_dates]
        
        # 计算Beta
        betas = self._rolling_regression(
            stock_returns_wide,
            benchmark_returns,
            self.window,
            self.min_periods
        )
        
        # 转换回长表格式
        betas_long = betas.stack()
        betas_long.index.names = ['TradingDates', 'StockCodes']
        
        # 预处理
        betas_long = self.preprocess(betas_long)
        
        return betas_long
    
    def _get_daily_returns(self, return_type: str = 'o2o') -> pd.Series:
        """获取预处理的日收益率数据"""
        try:
            return FactorDataLoader.load_returns('daily', return_type)
        except Exception as e:
            logger.error(f"无法加载预处理的收益率数据: {e}")
            raise RuntimeError("请先运行数据预处理脚本生成收益率数据")
    
    def _rolling_regression(self,
                          stock_returns: pd.DataFrame,
                          benchmark_returns: pd.Series,
                          window: int,
                          min_periods: int) -> pd.DataFrame:
        """滚动回归计算Beta"""
        betas = pd.DataFrame(index=stock_returns.index, 
                           columns=stock_returns.columns,
                           dtype=float)
        
        for i in range(len(stock_returns)):
            if i < min_periods - 1:
                continue
                
            # 确定窗口
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            # 获取窗口数据
            y = benchmark_returns.iloc[start_idx:end_idx].values
            
            for col in stock_returns.columns:
                x = stock_returns[col].iloc[start_idx:end_idx].values
                
                # 剔除NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < min_periods:
                    continue
                    
                x_clean = x[mask]
                y_clean = y[mask]
                
                # OLS回归
                try:
                    X = sm.add_constant(x_clean)
                    model = sm.OLS(y_clean, X)
                    results = model.fit()
                    betas.iloc[i, stock_returns.columns.get_loc(col)] = results.params[1]
                except:
                    continue
                    
        return betas


class WeightedBetaFactor(BetaFactor):
    """加权Beta因子"""
    
    def __init__(self,
                 window: int = 252,
                 half_life: int = 63):
        super().__init__(window=window)
        self.name = f'WeightedBeta_{window}_{half_life}'
        self.half_life = half_life
        self.description = f"Weighted beta (window={window}, half_life={half_life})"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 benchmark_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算加权Beta（越近期权重越大）
        """
        # 计算收益率
        stock_returns = self._get_daily_returns('o2o')
        benchmark_returns = np.log(
            benchmark_data['close'] / benchmark_data['close'].shift(1)
        )
        
        # 转换为宽表
        stock_returns_wide = stock_returns.unstack(level='StockCodes')
        
        # 对齐数据
        aligned_dates = stock_returns_wide.index.intersection(benchmark_returns.index)
        stock_returns_wide = stock_returns_wide.loc[aligned_dates]
        benchmark_returns = benchmark_returns.loc[aligned_dates]
        
        # 生成权重
        weights = self._generate_weights(self.window, self.half_life)
        
        # 计算加权Beta
        weighted_betas = self._weighted_rolling_regression(
            stock_returns_wide,
            benchmark_returns,
            self.window,
            self.min_periods,
            weights
        )
        
        # 转换回长表格式
        betas_long = weighted_betas.stack()
        betas_long.index.names = ['TradingDates', 'StockCodes']
        
        # 预处理
        betas_long = self.preprocess(betas_long)
        
        return betas_long
    
    def _generate_weights(self, window: int, half_life: int) -> np.ndarray:
        """生成指数衰减权重"""
        t = np.arange(window)
        weights = 0.5 ** (t / half_life)
        weights = weights[::-1]  # 反转，使最近的权重最大
        weights = weights / weights.sum()  # 归一化
        return weights
    
    def _weighted_rolling_regression(self,
                                   stock_returns: pd.DataFrame,
                                   benchmark_returns: pd.Series,
                                   window: int,
                                   min_periods: int,
                                   weights: np.ndarray) -> pd.DataFrame:
        """加权滚动回归"""
        betas = pd.DataFrame(index=stock_returns.index,
                           columns=stock_returns.columns,
                           dtype=float)
        
        for i in range(len(stock_returns)):
            if i < min_periods - 1:
                continue
                
            # 确定窗口
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_size = end_idx - start_idx
            
            # 获取窗口数据
            y = benchmark_returns.iloc[start_idx:end_idx].values
            
            # 调整权重
            window_weights = weights[-window_size:]
            
            for col in stock_returns.columns:
                x = stock_returns[col].iloc[start_idx:end_idx].values
                
                # 剔除NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < min_periods:
                    continue
                    
                x_clean = x[mask]
                y_clean = y[mask]
                w_clean = window_weights[mask]
                
                # WLS回归
                try:
                    X = sm.add_constant(x_clean)
                    model = sm.WLS(y_clean, X, weights=w_clean)
                    results = model.fit()
                    betas.iloc[i, stock_returns.columns.get_loc(col)] = results.params[1]
                except:
                    continue
                    
        return betas


class AlphaFactor(FactorBase):
    """Alpha因子（截距项）"""
    
    def __init__(self, window: int = 252):
        super().__init__(name=f'Alpha_{window}', category='risk')
        self.window = window
        self.description = f"Regression alpha over {window} days"
        
    def calculate(self,
                 price_data: pd.DataFrame,
                 benchmark_data: pd.DataFrame,
                 **kwargs) -> pd.Series:
        """
        计算回归Alpha（截距项）
        """
        # 复用Beta计算中的逻辑，但返回截距项
        # 这里简化实现，实际应该和Beta一起计算
        
        # 计算收益率
        stock_returns = self._get_daily_returns('o2o')
        benchmark_returns = np.log(
            benchmark_data['close'] / benchmark_data['close'].shift(1)
        )
        
        # 转换为宽表
        stock_returns_wide = stock_returns.unstack(level='StockCodes')
        
        # 对齐数据
        aligned_dates = stock_returns_wide.index.intersection(benchmark_returns.index)
        stock_returns_wide = stock_returns_wide.loc[aligned_dates]
        benchmark_returns = benchmark_returns.loc[aligned_dates]
        
        # 计算Alpha
        alphas = self._rolling_alpha(
            stock_returns_wide,
            benchmark_returns,
            self.window
        )
        
        # 转换回长表格式
        alphas_long = alphas.stack()
        alphas_long.index.names = ['TradingDates', 'StockCodes']
        
        # 预处理
        alphas_long = self.preprocess(alphas_long)
        
        return alphas_long
    
    def _get_daily_returns(self, return_type: str = 'o2o') -> pd.Series:
        """获取预处理的日收益率数据"""
        try:
            return FactorDataLoader.load_returns('daily', return_type)
        except Exception as e:
            logger.error(f"无法加载预处理的收益率数据: {e}")
            raise RuntimeError("请先运行数据预处理脚本生成收益率数据")
    
    def _rolling_alpha(self,
                      stock_returns: pd.DataFrame,
                      benchmark_returns: pd.Series,
                      window: int) -> pd.DataFrame:
        """滚动回归计算Alpha"""
        alphas = pd.DataFrame(index=stock_returns.index,
                            columns=stock_returns.columns,
                            dtype=float)
        
        for i in range(window-1, len(stock_returns)):
            # 确定窗口
            start_idx = i - window + 1
            end_idx = i + 1
            
            # 获取窗口数据
            y = benchmark_returns.iloc[start_idx:end_idx].values
            
            for col in stock_returns.columns:
                x = stock_returns[col].iloc[start_idx:end_idx].values
                
                # 剔除NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < window // 2:
                    continue
                    
                x_clean = x[mask]
                y_clean = y[mask]
                
                # OLS回归
                try:
                    X = sm.add_constant(x_clean)
                    model = sm.OLS(y_clean, X)
                    results = model.fit()
                    alphas.iloc[i, stock_returns.columns.get_loc(col)] = results.params[0]
                except:
                    continue
                    
        return alphas