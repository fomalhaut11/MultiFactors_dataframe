"""
样本协方差估计器

实现标准的样本协方差矩阵估计
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any

from ..base.exceptions import (
    DataFormatError,
    InsufficientDataError,
    CalculationError
)

logger = logging.getLogger(__name__)


class SampleCovarianceEstimator:
    """
    样本协方差估计器
    
    使用样本协方差矩阵估计资产间的协方差关系
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化样本协方差估计器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            估计器配置参数
        """
        self.config = config or {}
        
        # 默认配置
        self.min_periods = self.config.get('min_periods', 30)
        self.ddof = self.config.get('ddof', 1)  # 自由度调整
        self.center_data = self.config.get('center_data', True)  # 是否中心化
        self.handle_missing = self.config.get('handle_missing', 'drop')  # 缺失值处理
        
        self.covariance_matrix_ = None
        self.correlation_matrix_ = None
        self.asset_list_ = None
        self.estimation_stats_ = {}
        
        logger.info(f"Initialized SampleCovarianceEstimator with config: {self.config}")
    
    def fit(self, returns: pd.DataFrame) -> 'SampleCovarianceEstimator':
        """
        拟合样本协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据，index=dates, columns=assets
            
        Returns
        -------
        SampleCovarianceEstimator
            拟合后的估计器实例
        """
        self._validate_returns(returns)
        
        # 处理缺失值
        clean_returns = self._handle_missing_values(returns)
        
        # 中心化数据（如果需要）
        if self.center_data:
            centered_returns = clean_returns - clean_returns.mean()
        else:
            centered_returns = clean_returns
        
        # 计算样本协方差矩阵
        try:
            self.covariance_matrix_ = centered_returns.cov(ddof=self.ddof)
            
            # 计算相关系数矩阵
            self.correlation_matrix_ = centered_returns.corr()
            
            # 保存资产列表
            self.asset_list_ = clean_returns.columns.tolist()
            
            # 计算估计统计量
            self._calculate_estimation_stats(clean_returns)
            
            logger.info(f"Fitted covariance matrix for {len(self.asset_list_)} assets "
                       f"using {len(clean_returns)} observations")
            
        except Exception as e:
            raise CalculationError("sample covariance estimation", str(e))
        
        return self
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        获取协方差矩阵
        
        Returns
        -------
        pd.DataFrame
            协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
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
            各资产的波动率（标准差）
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        volatilities = np.sqrt(np.diag(self.covariance_matrix_))
        return pd.Series(volatilities, index=self.covariance_matrix_.index, name='volatility')
    
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
    
    def calculate_eigenvalues(self) -> pd.Series:
        """
        计算协方差矩阵的特征值
        
        Returns
        -------
        pd.Series
            特征值（降序排列）
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        eigenvalues = np.linalg.eigvals(self.covariance_matrix_.values)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
        
        return pd.Series(eigenvalues, name='eigenvalues')
    
    def calculate_condition_number(self) -> float:
        """
        计算协方差矩阵的条件数
        
        Returns
        -------
        float
            条件数
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return np.linalg.cond(self.covariance_matrix_.values)
    
    def is_positive_definite(self) -> bool:
        """
        检查协方差矩阵是否正定
        
        Returns
        -------
        bool
            是否正定
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        try:
            eigenvalues = np.linalg.eigvals(self.covariance_matrix_.values)
            return np.all(eigenvalues > 1e-8)
        except:
            return False
    
    def regularize_matrix(self, shrinkage: float = 0.01) -> pd.DataFrame:
        """
        正则化协方差矩阵
        
        Parameters
        ----------
        shrinkage : float
            收缩参数
            
        Returns
        -------
        pd.DataFrame
            正则化后的协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        # 对角化正则化
        n_assets = len(self.covariance_matrix_)
        identity = np.eye(n_assets)
        trace = np.trace(self.covariance_matrix_.values)
        
        regularized = ((1 - shrinkage) * self.covariance_matrix_.values + 
                      shrinkage * (trace / n_assets) * identity)
        
        return pd.DataFrame(regularized, 
                          index=self.covariance_matrix_.index,
                          columns=self.covariance_matrix_.columns)
    
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
    
    def _handle_missing_values(self, returns: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if self.handle_missing == 'drop':
            # 删除包含缺失值的行
            clean_returns = returns.dropna()
            
        elif self.handle_missing == 'forward_fill':
            # 前向填充
            clean_returns = returns.fillna(method='ffill')
            
        elif self.handle_missing == 'interpolate':
            # 线性插值
            clean_returns = returns.interpolate()
            
        elif self.handle_missing == 'mean':
            # 均值填充
            clean_returns = returns.fillna(returns.mean())
            
        else:
            raise ValueError(f"Unknown missing value handling method: {self.handle_missing}")
        
        # 再次检查数据量
        if len(clean_returns) < self.min_periods:
            raise InsufficientDataError(
                self.min_periods, 
                len(clean_returns), 
                "observations after missing value handling"
            )
        
        # 检查是否还有缺失值
        if clean_returns.isnull().any().any():
            remaining_nulls = clean_returns.isnull().sum().sum()
            logger.warning(f"Still have {remaining_nulls} missing values after handling")
        
        return clean_returns
    
    def _calculate_estimation_stats(self, returns: pd.DataFrame):
        """计算估计统计信息"""
        self.estimation_stats_ = {
            'n_observations': len(returns),
            'n_assets': len(returns.columns),
            'estimation_start': returns.index[0] if hasattr(returns.index, 'min') else None,
            'estimation_end': returns.index[-1] if hasattr(returns.index, 'max') else None,
            'condition_number': self.calculate_condition_number(),
            'is_positive_definite': self.is_positive_definite(),
            'min_eigenvalue': self.calculate_eigenvalues().min(),
            'max_eigenvalue': self.calculate_eigenvalues().max(),
            'mean_correlation': self.correlation_matrix_.values[
                np.triu_indices_from(self.correlation_matrix_.values, k=1)
            ].mean(),
            'max_correlation': self.correlation_matrix_.values[
                np.triu_indices_from(self.correlation_matrix_.values, k=1)
            ].max(),
            'min_correlation': self.correlation_matrix_.values[
                np.triu_indices_from(self.correlation_matrix_.values, k=1)
            ].min()
        }