"""
稳健协方差估计器

实现多种稳健协方差估计方法，用于处理异常值和非正态分布
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, Tuple, List
from scipy import stats
from scipy.linalg import sqrtm

from ..base.exceptions import (
    DataFormatError,
    InsufficientDataError,
    CalculationError,
    InvalidParameterError
)

logger = logging.getLogger(__name__)


class RobustCovarianceEstimator:
    """
    稳健协方差估计器
    
    提供多种稳健估计方法：
    1. MCD (Minimum Covariance Determinant)
    2. Huber估计
    3. Tyler估计
    4. 分位数协方差
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化稳健协方差估计器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            估计器配置参数
        """
        self.config = config or {}
        
        # 默认配置
        self.method = self.config.get('method', 'mcd')  # 估计方法
        self.support_fraction = self.config.get('support_fraction', None)  # MCD支持比例
        self.max_iterations = self.config.get('max_iterations', 500)
        self.tolerance = self.config.get('tolerance', 1e-6)
        self.random_state = self.config.get('random_state', 42)
        self.min_periods = self.config.get('min_periods', 30)
        self.handle_missing = self.config.get('handle_missing', 'drop')
        
        # 估计结果
        self.covariance_matrix_ = None
        self.location_ = None  # 稳健位置估计
        self.support_ = None  # 支持点（对于MCD）
        self.outliers_ = None  # 异常值标识
        self.asset_list_ = None
        self.estimation_stats_ = {}
        
        self._validate_parameters()
        
        logger.info(f"Initialized RobustCovarianceEstimator with method: {self.method}")
    
    def fit(self, returns: pd.DataFrame) -> 'RobustCovarianceEstimator':
        """
        拟合稳健协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据，index=dates, columns=assets
            
        Returns
        -------
        RobustCovarianceEstimator
            拟合后的估计器实例
        """
        self._validate_returns(returns)
        
        # 处理缺失值
        clean_returns = self._handle_missing_values(returns)
        
        try:
            # 根据方法选择估计算法
            if self.method == 'mcd':
                self._fit_mcd(clean_returns)
            elif self.method == 'huber':
                self._fit_huber(clean_returns)
            elif self.method == 'tyler':
                self._fit_tyler(clean_returns)
            elif self.method == 'quantile':
                self._fit_quantile_covariance(clean_returns)
            else:
                raise InvalidParameterError('method', self.method, 
                                          "{'mcd', 'huber', 'tyler', 'quantile'}")
            
            # 保存资产列表
            self.asset_list_ = clean_returns.columns.tolist()
            
            # 识别异常值
            self._identify_outliers(clean_returns)
            
            # 计算估计统计量
            self._calculate_estimation_stats(clean_returns)
            
            logger.info(f"Fitted robust covariance matrix for {len(self.asset_list_)} assets "
                       f"using {self.method} method")
            
        except Exception as e:
            raise CalculationError(f"robust covariance estimation ({self.method})", str(e))
        
        return self
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        获取稳健协方差矩阵
        
        Returns
        -------
        pd.DataFrame
            稳健协方差矩阵
        """
        if self.covariance_matrix_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.covariance_matrix_.copy()
    
    def get_location(self) -> pd.Series:
        """
        获取稳健位置估计
        
        Returns
        -------
        pd.Series
            稳健均值估计
        """
        if self.location_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.location_.copy()
    
    def get_outliers(self) -> pd.DataFrame:
        """
        获取异常值标识
        
        Returns
        -------
        pd.DataFrame
            异常值标识，布尔值DataFrame
        """
        if self.outliers_ is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() first.")
        
        return self.outliers_.copy()
    
    def get_support_points(self) -> Optional[np.ndarray]:
        """
        获取支持点（仅对MCD方法）
        
        Returns
        -------
        np.ndarray or None
            支持点索引
        """
        return self.support_.copy() if self.support_ is not None else None
    
    def _fit_mcd(self, returns: pd.DataFrame):
        """
        拟合最小协方差行列式估计器
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        """
        X = returns.values
        n_samples, n_features = X.shape
        
        # 确定支持比例
        if self.support_fraction is None:
            # 默认支持比例，基于样本量和维度
            self.support_fraction = (n_samples + n_features + 1) / (2 * n_samples)
            self.support_fraction = min(1.0, max(0.5, self.support_fraction))
        
        support_size = int(self.support_fraction * n_samples)
        
        # 随机搜索最佳支持集
        np.random.seed(self.random_state)
        best_determinant = np.inf
        best_mean = None
        best_cov = None
        best_support = None
        
        n_trials = min(500, 10 * n_features)
        
        for trial in range(n_trials):
            # 随机选择支持点
            support_indices = np.random.choice(n_samples, support_size, replace=False)
            X_support = X[support_indices]
            
            # 计算支持集的均值和协方差
            mean_support = np.mean(X_support, axis=0)
            cov_support = np.cov(X_support, rowvar=False)
            
            # 计算行列式
            try:
                det = np.linalg.det(cov_support)
                if det > 0 and det < best_determinant:
                    best_determinant = det
                    best_mean = mean_support
                    best_cov = cov_support
                    best_support = support_indices
            except:
                continue
        
        if best_cov is None:
            # 回退到样本协方差
            logger.warning("MCD estimation failed, using sample covariance")
            best_mean = np.mean(X, axis=0)
            best_cov = np.cov(X, rowvar=False)
            best_support = np.arange(n_samples)
        
        # 保存结果
        self.location_ = pd.Series(best_mean, index=returns.columns, name='robust_mean')
        self.covariance_matrix_ = pd.DataFrame(best_cov, 
                                             index=returns.columns, 
                                             columns=returns.columns)
        self.support_ = best_support
    
    def _fit_huber(self, returns: pd.DataFrame):
        """
        拟合Huber稳健估计器
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        """
        X = returns.values
        n_samples, n_features = X.shape
        
        # 初始化
        mean_est = np.mean(X, axis=0)
        cov_est = np.cov(X, rowvar=False)
        
        # Huber常数（通常选择1.345以达到95%的高斯效率）
        huber_c = 1.345
        
        # 迭代求解
        for iteration in range(self.max_iterations):
            old_mean = mean_est.copy()
            old_cov = cov_est.copy()
            
            # 计算马氏距离
            try:
                inv_cov = np.linalg.inv(cov_est)
                distances = np.array([
                    np.sqrt(np.dot(np.dot(x - mean_est, inv_cov), x - mean_est))
                    for x in X
                ])
            except:
                # 如果协方差矩阵奇异，添加正则化
                cov_est += 1e-6 * np.eye(n_features)
                inv_cov = np.linalg.inv(cov_est)
                distances = np.array([
                    np.sqrt(np.dot(np.dot(x - mean_est, inv_cov), x - mean_est))
                    for x in X
                ])
            
            # 计算权重
            weights = np.minimum(1.0, huber_c / distances)
            weights = np.maximum(weights, 1e-8)  # 避免零权重
            
            # 更新均值
            weighted_sum = np.sum(weights.reshape(-1, 1) * X, axis=0)
            weight_sum = np.sum(weights)
            mean_est = weighted_sum / weight_sum
            
            # 更新协方差
            weighted_deviations = weights.reshape(-1, 1) * (X - mean_est)
            cov_est = np.dot(weighted_deviations.T, X - mean_est) / weight_sum
            
            # 检查收敛
            mean_change = np.linalg.norm(mean_est - old_mean)
            cov_change = np.linalg.norm(cov_est - old_cov, 'fro')
            
            if mean_change < self.tolerance and cov_change < self.tolerance:
                logger.info(f"Huber estimation converged after {iteration + 1} iterations")
                break
        
        # 保存结果
        self.location_ = pd.Series(mean_est, index=returns.columns, name='huber_mean')
        self.covariance_matrix_ = pd.DataFrame(cov_est, 
                                             index=returns.columns, 
                                             columns=returns.columns)
    
    def _fit_tyler(self, returns: pd.DataFrame):
        """
        拟合Tyler M-估计器
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        """
        X = returns.values
        n_samples, n_features = X.shape
        
        # 初始化（使用中位数作为位置估计）
        mean_est = np.median(X, axis=0)
        
        # Tyler估计器假设已知位置参数，这里使用稳健的中位数
        centered_X = X - mean_est
        
        # 初始化协方差矩阵
        cov_est = np.cov(centered_X, rowvar=False)
        
        # 迭代求解
        for iteration in range(self.max_iterations):
            old_cov = cov_est.copy()
            
            try:
                inv_cov = np.linalg.inv(cov_est)
                
                # 计算加权协方差
                weighted_cov = np.zeros((n_features, n_features))
                
                for i in range(n_samples):
                    xi = centered_X[i]
                    quad_form = np.dot(np.dot(xi, inv_cov), xi)
                    
                    if quad_form > 1e-12:  # 避免除零
                        weight = n_features / quad_form
                        weighted_cov += weight * np.outer(xi, xi)
                
                cov_est = weighted_cov / n_samples
                
            except np.linalg.LinAlgError:
                # 如果奇异，添加正则化
                cov_est += 1e-6 * np.eye(n_features)
                continue
            
            # 检查收敛
            cov_change = np.linalg.norm(cov_est - old_cov, 'fro')
            if cov_change < self.tolerance:
                logger.info(f"Tyler estimation converged after {iteration + 1} iterations")
                break
        
        # 保存结果
        self.location_ = pd.Series(mean_est, index=returns.columns, name='tyler_median')
        self.covariance_matrix_ = pd.DataFrame(cov_est, 
                                             index=returns.columns, 
                                             columns=returns.columns)
    
    def _fit_quantile_covariance(self, returns: pd.DataFrame):
        """
        拟合分位数协方差估计器
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        """
        # 使用Kendall's tau和正态逆变换
        # 这是一种分布自由的协方差估计方法
        
        X = returns.values
        n_samples, n_features = X.shape
        
        # 计算位置估计（中位数）
        mean_est = np.median(X, axis=0)
        
        # 计算Kendall tau相关系数矩阵
        tau_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    tau_matrix[i, j] = 1.0
                else:
                    # 计算Kendall tau
                    tau, _ = stats.kendalltau(X[:, i], X[:, j])
                    tau_matrix[i, j] = tau_matrix[j, i] = tau
        
        # 转换为协方差矩阵
        # 使用sin变换：sigma_ij = sin(pi/2 * tau_ij) * sigma_i * sigma_j
        marginal_scales = np.array([
            stats.median_abs_deviation(X[:, i], scale='normal') for i in range(n_features)
        ])
        
        cov_est = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    cov_est[i, j] = marginal_scales[i] ** 2
                else:
                    correlation = np.sin(np.pi / 2 * tau_matrix[i, j])
                    cov_est[i, j] = correlation * marginal_scales[i] * marginal_scales[j]
        
        # 保存结果
        self.location_ = pd.Series(mean_est, index=returns.columns, name='quantile_median')
        self.covariance_matrix_ = pd.DataFrame(cov_est, 
                                             index=returns.columns, 
                                             columns=returns.columns)
    
    def _identify_outliers(self, returns: pd.DataFrame):
        """
        识别异常值
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        """
        X = returns.values
        n_samples, n_features = X.shape
        
        # 计算马氏距离
        try:
            inv_cov = np.linalg.inv(self.covariance_matrix_.values)
            distances = np.array([
                np.sqrt(np.dot(np.dot(x - self.location_.values, inv_cov), 
                              x - self.location_.values))
                for x in X
            ])
            
            # 使用卡方分布确定异常值阈值
            threshold = stats.chi2.ppf(0.975, n_features)  # 97.5%分位数
            outlier_mask = distances > np.sqrt(threshold)
            
            # 创建异常值DataFrame
            self.outliers_ = pd.DataFrame(
                False, 
                index=returns.index, 
                columns=returns.columns
            )
            
            # 标记异常观测
            outlier_indices = returns.index[outlier_mask]
            if len(outlier_indices) > 0:
                self.outliers_.loc[outlier_indices] = True
                logger.info(f"Identified {len(outlier_indices)} outlier observations")
            
        except Exception as e:
            logger.warning(f"Failed to identify outliers: {e}")
            self.outliers_ = pd.DataFrame(
                False, 
                index=returns.index, 
                columns=returns.columns
            )
    
    def _validate_parameters(self):
        """验证参数"""
        valid_methods = {'mcd', 'huber', 'tyler', 'quantile'}
        if self.method not in valid_methods:
            raise InvalidParameterError('method', self.method, str(valid_methods))
        
        if self.support_fraction is not None:
            if not 0.5 <= self.support_fraction <= 1.0:
                raise InvalidParameterError('support_fraction', self.support_fraction, "[0.5, 1.0]")
        
        if self.max_iterations < 1:
            raise InvalidParameterError('max_iterations', self.max_iterations, ">= 1")
        
        if self.tolerance <= 0:
            raise InvalidParameterError('tolerance', self.tolerance, "> 0")
    
    def _validate_returns(self, returns: pd.DataFrame):
        """验证收益率数据"""
        if not isinstance(returns, pd.DataFrame):
            raise DataFormatError("pandas DataFrame", f"{type(returns).__name__}")
        
        if len(returns) < self.min_periods:
            raise InsufficientDataError(self.min_periods, len(returns), "return observations")
        
        if len(returns.columns) < 2:
            raise InsufficientDataError(2, len(returns.columns), "assets")
        
        # 对于MCD，需要足够的样本量
        if self.method == 'mcd':
            min_samples_mcd = len(returns.columns) * 2
            if len(returns) < min_samples_mcd:
                raise InsufficientDataError(min_samples_mcd, len(returns), 
                                          "observations for MCD estimation")
    
    def _handle_missing_values(self, returns: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if self.handle_missing == 'drop':
            clean_returns = returns.dropna()
        elif self.handle_missing == 'forward_fill':
            clean_returns = returns.fillna(method='ffill')
        elif self.handle_missing == 'interpolate':
            clean_returns = returns.interpolate()
        elif self.handle_missing == 'median':
            clean_returns = returns.fillna(returns.median())
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
        
        # 计算与样本协方差的差异
        sample_cov = returns.cov()
        frobenius_diff = np.linalg.norm(
            self.covariance_matrix_.values - sample_cov.values, 'fro'
        )
        
        # 计算异常值比例
        outlier_ratio = self.outliers_.any(axis=1).mean() if self.outliers_ is not None else 0
        
        self.estimation_stats_ = {
            'n_observations': len(returns),
            'n_assets': len(returns.columns),
            'method': self.method,
            'support_fraction': getattr(self, 'support_fraction', None),
            'condition_number': condition_number,
            'is_positive_definite': np.all(eigenvals > 1e-8),
            'min_eigenvalue': np.min(eigenvals),
            'max_eigenvalue': np.max(eigenvals),
            'outlier_ratio': outlier_ratio,
            'frobenius_difference_vs_sample': frobenius_diff,
            'estimation_robust': True
        }
        
        if self.support_ is not None:
            self.estimation_stats_['support_size'] = len(self.support_)
    
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