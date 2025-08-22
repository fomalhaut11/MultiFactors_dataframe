"""
Ledoit-Wolf收缩估计器

实现Ledoit-Wolf收缩方法估计协方差矩阵
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, Tuple

from ..base.exceptions import (
    DataFormatError,
    InsufficientDataError,
    CalculationError
)

logger = logging.getLogger(__name__)


class LedoitWolfEstimator:
    """
    Ledoit-Wolf收缩估计器
    
    基于Ledoit和Wolf (2004)的收缩方法估计协方差矩阵
    自动选择最优收缩参数，提高估计的稳定性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Ledoit-Wolf估计器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            估计器配置参数
        """
        self.config = config or {}
        
        # 默认配置
        self.shrinkage_target = self.config.get('shrinkage_target', 'diagonal')  # 收缩目标
        self.min_periods = self.config.get('min_periods', 30)
        self.assume_centered = self.config.get('assume_centered', False)
        self.handle_missing = self.config.get('handle_missing', 'drop')
        
        # 估计结果
        self.covariance_matrix_ = None
        self.shrinkage_ = None
        self.target_matrix_ = None
        self.sample_covariance_ = None
        self.asset_list_ = None
        self.estimation_stats_ = {}
        
        logger.info(f"Initialized LedoitWolfEstimator with config: {self.config}")
    
    def fit(self, returns: pd.DataFrame) -> 'LedoitWolfEstimator':
        """
        拟合Ledoit-Wolf收缩协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据，index=dates, columns=assets
            
        Returns
        -------
        LedoitWolfEstimator
            拟合后的估计器实例
        """
        self._validate_returns(returns)
        
        # 处理缺失值
        clean_returns = self._handle_missing_values(returns)
        
        # 中心化数据
        if not self.assume_centered:
            centered_returns = clean_returns - clean_returns.mean()
        else:
            centered_returns = clean_returns
        
        try:
            # 计算样本协方差矩阵
            self.sample_covariance_ = centered_returns.cov()
            
            # 计算收缩目标矩阵
            self.target_matrix_ = self._compute_shrinkage_target(centered_returns)
            
            # 计算最优收缩参数
            self.shrinkage_ = self._compute_optimal_shrinkage(centered_returns)
            
            # 计算收缩协方差矩阵
            self.covariance_matrix_ = self._compute_shrunk_covariance()
            
            # 保存资产列表
            self.asset_list_ = clean_returns.columns.tolist()
            
            # 计算估计统计量
            self._calculate_estimation_stats(clean_returns)
            
            logger.info(f"Fitted Ledoit-Wolf covariance matrix for {len(self.asset_list_)} assets "
                       f"with shrinkage parameter: {self.shrinkage_:.4f}")
            
        except Exception as e:
            raise CalculationError("Ledoit-Wolf estimation", str(e))
        
        return self
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        获取收缩协方差矩阵
        
        Returns
        -------
        pd.DataFrame
            收缩协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.covariance_matrix_.copy()
    
    def get_sample_covariance(self) -> pd.DataFrame:
        """
        获取样本协方差矩阵
        
        Returns
        -------
        pd.DataFrame
            样本协方差矩阵
        """
        if self.sample_covariance_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.sample_covariance_.copy()
    
    def get_shrinkage_parameter(self) -> float:
        """
        获取收缩参数
        
        Returns
        -------
        float
            最优收缩参数
        """
        if self.shrinkage_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.shrinkage_
    
    def get_target_matrix(self) -> pd.DataFrame:
        """
        获取收缩目标矩阵
        
        Returns
        -------
        pd.DataFrame
            收缩目标矩阵
        """
        if self.target_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.target_matrix_.copy()
    
    def compute_shrinkage_with_parameter(self, shrinkage: float) -> pd.DataFrame:
        """
        使用指定收缩参数计算协方差矩阵
        
        Parameters
        ----------
        shrinkage : float
            收缩参数 [0, 1]
            
        Returns
        -------
        pd.DataFrame
            收缩协方差矩阵
        """
        if self.sample_covariance_ is None or self.target_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        if not 0 <= shrinkage <= 1:
            raise ValueError("Shrinkage parameter must be between 0 and 1")
        
        shrunk_cov = ((1 - shrinkage) * self.sample_covariance_ + 
                     shrinkage * self.target_matrix_)
        
        return shrunk_cov
    
    def _compute_shrinkage_target(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算收缩目标矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            中心化收益率数据
            
        Returns
        -------
        pd.DataFrame
            收缩目标矩阵
        """
        n_assets = len(returns.columns)
        
        if self.shrinkage_target == 'diagonal':
            # 对角矩阵：保持样本方差，相关系数为0
            target = pd.DataFrame(
                np.diag(np.diag(self.sample_covariance_.values)),
                index=returns.columns,
                columns=returns.columns
            )
            
        elif self.shrinkage_target == 'identity':
            # 单位矩阵的倍数
            trace = np.trace(self.sample_covariance_.values)
            scaling = trace / n_assets
            target = pd.DataFrame(
                scaling * np.eye(n_assets),
                index=returns.columns,
                columns=returns.columns
            )
            
        elif self.shrinkage_target == 'single_factor':
            # 单因子模型
            market_variance = np.var(returns.mean(axis=1))
            average_asset_variance = np.mean(np.diag(self.sample_covariance_.values))
            
            target_values = np.full((n_assets, n_assets), market_variance)
            np.fill_diagonal(target_values, average_asset_variance)
            
            target = pd.DataFrame(
                target_values,
                index=returns.columns,
                columns=returns.columns
            )
            
        elif self.shrinkage_target == 'constant_correlation':
            # 常数相关系数模型
            avg_correlation = self._compute_average_correlation(returns)
            asset_variances = np.diag(self.sample_covariance_.values)
            
            target_values = np.outer(np.sqrt(asset_variances), np.sqrt(asset_variances))
            target_values *= avg_correlation
            np.fill_diagonal(target_values, asset_variances)
            
            target = pd.DataFrame(
                target_values,
                index=returns.columns,
                columns=returns.columns
            )
            
        else:
            raise ValueError(f"Unknown shrinkage target: {self.shrinkage_target}")
        
        return target
    
    def _compute_optimal_shrinkage(self, returns: pd.DataFrame) -> float:
        """
        计算最优收缩参数
        
        基于Ledoit-Wolf (2004)的解析公式
        
        Parameters
        ----------
        returns : pd.DataFrame
            中心化收益率数据
            
        Returns
        -------
        float
            最优收缩参数
        """
        X = returns.values
        n_samples, n_features = X.shape
        
        # 计算样本协方差矩阵
        S = self.sample_covariance_.values
        
        # 计算收缩目标
        F = self.target_matrix_.values
        
        # 计算Pi（样本协方差的不确定性）
        Pi = 0
        for i in range(n_samples):
            xi = X[i:i+1].T  # 列向量
            Pi += np.linalg.norm(np.outer(xi, xi) - S, 'fro') ** 2
        Pi /= n_samples
        
        # 计算Rho（样本协方差与目标的协方差）
        Rho = 0
        for i in range(n_samples):
            xi = X[i:i+1].T
            Rho += np.linalg.norm(np.outer(xi, xi) - F, 'fro') ** 2
        Rho /= n_samples
        
        # 计算Gamma（目标与样本协方差的距离）
        Gamma = np.linalg.norm(S - F, 'fro') ** 2
        
        # 计算最优收缩参数
        kappa = (Pi - Rho) / Gamma
        shrinkage = max(0, min(1, kappa / n_samples))
        
        return shrinkage
    
    def _compute_shrunk_covariance(self) -> pd.DataFrame:
        """
        计算收缩协方差矩阵
        
        Returns
        -------
        pd.DataFrame
            收缩协方差矩阵
        """
        shrunk_cov = ((1 - self.shrinkage_) * self.sample_covariance_ + 
                     self.shrinkage_ * self.target_matrix_)
        
        return shrunk_cov
    
    def _compute_average_correlation(self, returns: pd.DataFrame) -> float:
        """
        计算平均相关系数
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
            
        Returns
        -------
        float
            平均相关系数
        """
        corr_matrix = returns.corr().values
        n = len(corr_matrix)
        
        # 提取上三角（排除对角线）
        upper_triangular = corr_matrix[np.triu_indices(n, k=1)]
        
        return np.mean(upper_triangular)
    
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
            clean_returns = returns.dropna()
        elif self.handle_missing == 'forward_fill':
            clean_returns = returns.fillna(method='ffill')
        elif self.handle_missing == 'interpolate':
            clean_returns = returns.interpolate()
        elif self.handle_missing == 'mean':
            clean_returns = returns.fillna(returns.mean())
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
        # 计算协方差矩阵的基本统计
        eigenvals = np.linalg.eigvals(self.covariance_matrix_.values)
        condition_number = np.max(eigenvals) / np.min(eigenvals) if np.min(eigenvals) > 0 else np.inf
        
        self.estimation_stats_ = {
            'n_observations': len(returns),
            'n_assets': len(returns.columns),
            'shrinkage_parameter': self.shrinkage_,
            'shrinkage_target': self.shrinkage_target,
            'condition_number': condition_number,
            'is_positive_definite': np.all(eigenvals > 1e-8),
            'min_eigenvalue': np.min(eigenvals),
            'max_eigenvalue': np.max(eigenvals),
            'frobenius_norm_difference': np.linalg.norm(
                self.covariance_matrix_.values - self.sample_covariance_.values, 'fro'
            ),
            'relative_improvement': self._calculate_relative_improvement()
        }
    
    def _calculate_relative_improvement(self) -> float:
        """
        计算相对于样本协方差的改进
        
        Returns
        -------
        float
            相对改进（条件数的改善）
        """
        try:
            sample_cond = np.linalg.cond(self.sample_covariance_.values)
            shrunk_cond = np.linalg.cond(self.covariance_matrix_.values)
            return (sample_cond - shrunk_cond) / sample_cond
        except:
            return 0.0
    
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