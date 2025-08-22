"""
指数加权协方差估计器

实现指数加权移动平均(EWMA)协方差矩阵估计
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, Union

from ..base.exceptions import (
    DataFormatError,
    InsufficientDataError,
    CalculationError,
    InvalidParameterError
)

logger = logging.getLogger(__name__)


class ExponentialWeightedEstimator:
    """
    指数加权协方差估计器
    
    使用指数加权移动平均方法估计时变协方差矩阵，
    对近期观测赋予更高权重，能够更好地捕获波动率聚集效应
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化指数加权估计器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            估计器配置参数
        """
        self.config = config or {}
        
        # 默认配置
        self.decay_factor = self.config.get('decay_factor', 0.94)  # λ参数
        self.span = self.config.get('span', None)  # 半衰期
        self.min_periods = self.config.get('min_periods', 30)
        self.adjust = self.config.get('adjust', True)  # 是否调整偏差
        self.ignore_na = self.config.get('ignore_na', False)
        self.handle_missing = self.config.get('handle_missing', 'drop')
        
        # 如果提供span，转换为decay_factor
        if self.span is not None:
            self.decay_factor = 2.0 / (self.span + 1.0)
        
        # 估计结果
        self.covariance_matrix_ = None
        self.correlation_matrix_ = None
        self.volatilities_ = None
        self.asset_list_ = None
        self.estimation_stats_ = {}
        self.covariance_series_ = None  # 时间序列协方差
        
        self._validate_parameters()
        
        logger.info(f"Initialized ExponentialWeightedEstimator with decay_factor: {self.decay_factor:.4f}")
    
    def fit(self, returns: pd.DataFrame) -> 'ExponentialWeightedEstimator':
        """
        拟合指数加权协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据，index=dates, columns=assets
            
        Returns
        -------
        ExponentialWeightedEstimator
            拟合后的估计器实例
        """
        self._validate_returns(returns)
        
        # 处理缺失值
        clean_returns = self._handle_missing_values(returns)
        
        try:
            # 计算指数加权协方差矩阵
            self.covariance_matrix_ = self._compute_ewm_covariance(clean_returns)
            
            # 计算波动率和相关系数矩阵
            self.volatilities_ = self._extract_volatilities()
            self.correlation_matrix_ = self._compute_correlation_matrix()
            
            # 计算时间序列协方差（可选）
            if self.config.get('compute_time_series', False):
                self.covariance_series_ = self._compute_time_series_covariance(clean_returns)
            
            # 保存资产列表
            self.asset_list_ = clean_returns.columns.tolist()
            
            # 计算估计统计量
            self._calculate_estimation_stats(clean_returns)
            
            logger.info(f"Fitted EWMA covariance matrix for {len(self.asset_list_)} assets "
                       f"using {len(clean_returns)} observations with decay factor: {self.decay_factor:.4f}")
            
        except Exception as e:
            raise CalculationError("exponential weighted covariance estimation", str(e))
        
        return self
    
    def get_covariance_matrix(self, as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        获取协方差矩阵
        
        Parameters
        ----------
        as_of_date : pd.Timestamp, optional
            指定日期的协方差矩阵（如果计算了时间序列）
            
        Returns
        -------
        pd.DataFrame
            协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        if as_of_date is not None and self.covariance_series_ is not None:
            if as_of_date in self.covariance_series_:
                return self.covariance_series_[as_of_date].copy()
            else:
                raise ValueError(f"No covariance matrix available for date {as_of_date}")
        
        return self.covariance_matrix_.copy()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        获取相关系数矩阵
        
        Returns
        -------
        pd.DataFrame
            相关系数矩阵
        """
        if self.correlation_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.correlation_matrix_.copy()
    
    def get_volatilities(self) -> pd.Series:
        """
        获取各资产的波动率
        
        Returns
        -------
        pd.Series
            各资产的波动率
        """
        if self.volatilities_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.volatilities_.copy()
    
    def get_time_series_covariance(self) -> Optional[Dict[pd.Timestamp, pd.DataFrame]]:
        """
        获取时间序列协方差矩阵
        
        Returns
        -------
        Dict[pd.Timestamp, pd.DataFrame] or None
            时间序列协方差矩阵字典
        """
        return self.covariance_series_.copy() if self.covariance_series_ else None
    
    def update_covariance(self, new_returns: pd.Series) -> pd.DataFrame:
        """
        使用新收益率更新协方差矩阵
        
        Parameters
        ----------
        new_returns : pd.Series
            新的收益率观测值
            
        Returns
        -------
        pd.DataFrame
            更新后的协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        # 对齐资产
        common_assets = self.covariance_matrix_.index.intersection(new_returns.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between existing covariance and new returns")
        
        aligned_returns = new_returns.reindex(common_assets, fill_value=0)
        
        # 指数加权更新
        old_cov = self.covariance_matrix_.reindex(index=common_assets, columns=common_assets)
        
        # 计算新的外积
        new_outer = np.outer(aligned_returns.values, aligned_returns.values)
        
        # 更新协方差矩阵
        updated_cov = (self.decay_factor * old_cov.values + 
                      (1 - self.decay_factor) * new_outer)
        
        updated_cov_df = pd.DataFrame(updated_cov, 
                                     index=common_assets, 
                                     columns=common_assets)
        
        # 更新存储的协方差矩阵
        self.covariance_matrix_ = updated_cov_df
        self.volatilities_ = self._extract_volatilities()
        self.correlation_matrix_ = self._compute_correlation_matrix()
        
        return updated_cov_df
    
    def forecast_covariance(self, horizon: int = 1) -> pd.DataFrame:
        """
        预测未来协方差矩阵
        
        Parameters
        ----------
        horizon : int
            预测期限（天数）
            
        Returns
        -------
        pd.DataFrame
            预测的协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        # 简单的多期预测：假设协方差矩阵保持不变
        # 更复杂的预测可以考虑方差的均值回归等
        forecasted_cov = self.covariance_matrix_ * horizon
        
        return forecasted_cov
    
    def _compute_ewm_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算指数加权协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
            
        Returns
        -------
        pd.DataFrame
            指数加权协方差矩阵
        """
        # 使用pandas的ewm方法计算协方差
        ewm_cov = returns.ewm(
            alpha=1 - self.decay_factor,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na
        ).cov()
        
        # 获取最新的协方差矩阵
        latest_date = ewm_cov.index.get_level_values(0)[-1]
        latest_cov = ewm_cov.loc[latest_date]
        
        return latest_cov
    
    def _compute_time_series_covariance(self, returns: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        计算时间序列协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
            
        Returns
        -------
        Dict[pd.Timestamp, pd.DataFrame]
            时间序列协方差矩阵
        """
        # 计算滚动的指数加权协方差
        ewm_cov = returns.ewm(
            alpha=1 - self.decay_factor,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na
        ).cov()
        
        # 组织成字典格式
        covariance_series = {}
        for date in ewm_cov.index.get_level_values(0).unique():
            covariance_series[date] = ewm_cov.loc[date]
        
        return covariance_series
    
    def _extract_volatilities(self) -> pd.Series:
        """提取波动率"""
        volatilities = np.sqrt(np.diag(self.covariance_matrix_.values))
        return pd.Series(volatilities, index=self.covariance_matrix_.index, name='volatility')
    
    def _compute_correlation_matrix(self) -> pd.DataFrame:
        """计算相关系数矩阵"""
        volatilities = self.volatilities_.values
        vol_matrix = np.outer(volatilities, volatilities)
        
        # 避免除零
        correlation = np.divide(
            self.covariance_matrix_.values,
            vol_matrix,
            out=np.zeros_like(self.covariance_matrix_.values),
            where=vol_matrix != 0
        )
        
        # 确保对角线为1
        np.fill_diagonal(correlation, 1.0)
        
        return pd.DataFrame(correlation,
                          index=self.covariance_matrix_.index,
                          columns=self.covariance_matrix_.columns)
    
    def _validate_parameters(self):
        """验证参数"""
        if not 0 < self.decay_factor < 1:
            raise InvalidParameterError('decay_factor', self.decay_factor, "(0, 1)")
        
        if self.span is not None and self.span <= 0:
            raise InvalidParameterError('span', self.span, "> 0")
        
        if self.min_periods < 1:
            raise InvalidParameterError('min_periods', self.min_periods, ">= 1")
    
    def _validate_returns(self, returns: pd.DataFrame):
        """验证收益率数据"""
        if not isinstance(returns, pd.DataFrame):
            raise DataFormatError("pandas DataFrame", f"{type(returns).__name__}")
        
        if len(returns) < self.min_periods:
            raise InsufficientDataError(self.min_periods, len(returns), "return observations")
        
        if len(returns.columns) < 2:
            raise InsufficientDataError(2, len(returns.columns), "assets")
        
        # 检查数据类型
        non_numeric_cols = returns.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            raise DataFormatError(f"numeric data for all columns", 
                                f"non-numeric columns: {non_numeric_cols.tolist()}")
        
        # 检查日期索引
        if not isinstance(returns.index, pd.DatetimeIndex):
            logger.warning("Returns index is not DatetimeIndex, time series features may not work properly")
    
    def _handle_missing_values(self, returns: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if self.handle_missing == 'drop':
            clean_returns = returns.dropna()
        elif self.handle_missing == 'forward_fill':
            clean_returns = returns.fillna(method='ffill')
        elif self.handle_missing == 'interpolate':
            clean_returns = returns.interpolate()
        elif self.handle_missing == 'zero':
            clean_returns = returns.fillna(0)
        else:
            raise ValueError(f"Unknown missing value handling method: {self.handle_missing}")
        
        if len(clean_returns) < self.min_periods:
            raise InsufficientDataError(
                self.min_periods, 
                len(clean_returns), 
                "observations after missing value handling"
            )
        
        return clean_returns
    
    def _calculate_estimation_stats(self, returns: pd.DataFrame):
        """计算估计统计信息"""
        # 计算有效半衰期
        effective_span = 2.0 / self.decay_factor - 1.0
        
        # 计算协方差矩阵的基本统计
        eigenvals = np.linalg.eigvals(self.covariance_matrix_.values)
        condition_number = np.max(eigenvals) / np.min(eigenvals) if np.min(eigenvals) > 0 else np.inf
        
        # 计算与等权协方差的差异
        equal_weight_cov = returns.cov()
        frobenius_diff = np.linalg.norm(
            self.covariance_matrix_.values - equal_weight_cov.values, 'fro'
        )
        
        self.estimation_stats_ = {
            'n_observations': len(returns),
            'n_assets': len(returns.columns),
            'decay_factor': self.decay_factor,
            'effective_span': effective_span,
            'min_periods': self.min_periods,
            'condition_number': condition_number,
            'is_positive_definite': np.all(eigenvals > 1e-8),
            'min_eigenvalue': np.min(eigenvals),
            'max_eigenvalue': np.max(eigenvals),
            'average_volatility': self.volatilities_.mean(),
            'max_volatility': self.volatilities_.max(),
            'min_volatility': self.volatilities_.min(),
            'average_correlation': self._compute_average_correlation(),
            'frobenius_difference_vs_sample': frobenius_diff
        }
    
    def _compute_average_correlation(self) -> float:
        """计算平均相关系数"""
        corr_values = self.correlation_matrix_.values
        n = len(corr_values)
        
        # 提取上三角（排除对角线）
        upper_triangular = corr_values[np.triu_indices(n, k=1)]
        
        return np.mean(upper_triangular)
    
    def get_estimation_stats(self) -> Dict[str, Any]:
        """
        获取估计统计信息
        
        Returns
        -------
        Dict[str, Any]
            估计统计信息
        """
        return self.estimation_stats_.copy()
    
    def predict_portfolio_risk(self, weights: pd.Series) -> float:
        """
        预测组合风险
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
            
        Returns
        -------
        float
            组合波动率
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        # 对齐权重和协方差矩阵
        common_assets = self.covariance_matrix_.index.intersection(weights.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between weights and covariance matrix")
        
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_cov = self.covariance_matrix_.reindex(index=common_assets, columns=common_assets)
        
        # 计算组合方差
        portfolio_variance = np.dot(aligned_weights.values, 
                                  np.dot(aligned_cov.values, aligned_weights.values))
        
        return np.sqrt(portfolio_variance)
    
    def calculate_decay_weights(self, n_periods: int) -> np.ndarray:
        """
        计算指数衰减权重
        
        Parameters
        ----------
        n_periods : int
            时间期数
            
        Returns
        -------
        np.ndarray
            衰减权重数组（最新观测权重最大）
        """
        weights = np.array([self.decay_factor ** i for i in range(n_periods)])
        weights = weights[::-1]  # 反转，使最新观测权重最大
        
        # 标准化权重
        return weights / weights.sum()