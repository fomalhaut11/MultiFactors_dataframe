"""
协方差模型

基于协方差矩阵的风险建模方法，支持多种估计器
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, List, Union, Tuple
from datetime import datetime

from ..base.risk_model_base import RiskModelBase
from ..base.exceptions import (
    ModelNotFittedError,
    SingularCovarianceError,
    CalculationError,
    InvalidParameterError,
    InsufficientDataError
)
from ..estimators.sample_covariance import SampleCovarianceEstimator
from ..estimators.ledoit_wolf import LedoitWolfEstimator
from ..estimators.exponential_weighted import ExponentialWeightedEstimator
from ..estimators.robust_estimators import RobustCovarianceEstimator

logger = logging.getLogger(__name__)


class CovarianceModel(RiskModelBase):
    """
    协方差风险模型
    
    基于协方差矩阵的风险建模，支持多种协方差估计方法：
    - 样本协方差
    - Ledoit-Wolf收缩
    - 指数加权移动平均
    - 稳健估计器
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化协方差模型
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            模型配置参数
        """
        super().__init__(config)
        
        # 模型特定配置
        self.estimator_method = self.config.get('estimator_method', 'sample')
        self.estimator_config = self.config.get('estimator_config', {})
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self.regularization = self.config.get('regularization', None)
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'monthly')
        
        # 估计器实例
        self.estimator_ = None
        self.covariance_matrix_ = None
        self.returns_data_ = None
        self.asset_universe_ = None
        
        # 创建估计器
        self._create_estimator()
        
        logger.info(f"Initialized CovarianceModel with estimator: {self.estimator_method}")
    
    def fit(self, 
            factor_exposures: pd.DataFrame,
            returns: pd.Series,
            **kwargs) -> 'CovarianceModel':
        """
        拟合协方差模型
        
        Parameters
        ----------
        factor_exposures : pd.DataFrame
            因子暴露度矩阵（用于获取资产列表，可以为None）
        returns : pd.Series  
            股票收益率，MultiIndex(date, stock)
        **kwargs : dict
            其他参数
            
        Returns
        -------
        CovarianceModel
            拟合后的模型实例
        """
        # 验证数据
        self.validate_returns(returns)
        
        # 将Series转换为DataFrame
        returns_df = self._prepare_returns_data(returns)
        
        # 保存原始数据
        self.returns_data_ = returns_df.copy()
        self.asset_universe_ = returns_df.columns.tolist()
        
        # 使用估计器拟合协方差矩阵
        try:
            self.estimator_.fit(returns_df)
            
            # 获取协方差矩阵
            self.covariance_matrix_ = self.estimator_.get_covariance_matrix()
            
            # 应用正则化（如果需要）
            if self.regularization:
                self.covariance_matrix_ = self._apply_regularization(self.covariance_matrix_)
            
            # 验证协方差矩阵
            self._validate_covariance_matrix()
            
            # 更新模型状态
            n_observations = len(returns_df)
            n_assets = len(returns_df.columns)
            self._log_fit_completion(n_observations, n_assets)
            
            # 保存模型参数
            self.model_params = {
                'estimator_method': self.estimator_method,
                'n_assets': n_assets,
                'n_observations': n_observations,
                'condition_number': np.linalg.cond(self.covariance_matrix_.values),
                'min_eigenvalue': np.min(np.linalg.eigvals(self.covariance_matrix_.values)),
                'max_eigenvalue': np.max(np.linalg.eigvals(self.covariance_matrix_.values))
            }
            
            # 添加估计器特定统计
            if hasattr(self.estimator_, 'get_estimation_stats'):
                estimator_stats = self.estimator_.get_estimation_stats()
                self.model_params.update(estimator_stats)
            
            logger.info(f"Successfully fitted CovarianceModel with {n_assets} assets")
            
        except Exception as e:
            raise CalculationError("covariance model fitting", str(e))
        
        return self
    
    def predict_covariance(self, 
                          horizon: int = 1,
                          method: str = 'default') -> pd.DataFrame:
        """
        预测协方差矩阵
        
        Parameters
        ----------
        horizon : int
            预测时间范围（天数）
        method : str
            预测方法
            
        Returns  
        -------
        pd.DataFrame
            预测的协方差矩阵
        """
        self._check_fitted()
        
        if method == 'default':
            # 简单的时间缩放
            forecast_cov = self.covariance_matrix_ * horizon
            
        elif method == 'exponential_decay':
            # 指数衰减预测（适用于EWMA估计器）
            if hasattr(self.estimator_, 'forecast_covariance'):
                forecast_cov = self.estimator_.forecast_covariance(horizon)
            else:
                # 使用默认方法
                forecast_cov = self.covariance_matrix_ * horizon
                
        elif method == 'mean_reversion':
            # 均值回归预测
            forecast_cov = self._forecast_with_mean_reversion(horizon)
            
        else:
            raise InvalidParameterError('method', method, 
                                      "{'default', 'exponential_decay', 'mean_reversion'}")
        
        return forecast_cov
    
    def calculate_portfolio_risk(self, 
                                weights: pd.Series,
                                horizon: int = 1) -> Dict[str, float]:
        """
        计算组合风险
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
        horizon : int
            风险预测时间范围
            
        Returns
        -------
        Dict[str, float]
            风险指标字典
        """
        self._check_fitted()
        self.validate_weights(weights)
        
        # 获取协方差矩阵
        cov_matrix = self.predict_covariance(horizon)
        
        # 对齐权重和协方差矩阵
        common_assets = cov_matrix.index.intersection(weights.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between weights and covariance matrix")
        
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_cov = cov_matrix.reindex(index=common_assets, columns=common_assets)
        
        # 计算组合方差和波动率
        portfolio_variance = np.dot(aligned_weights.values,
                                  np.dot(aligned_cov.values, aligned_weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 计算VaR（假设正态分布）
        var_95 = -1.96 * portfolio_volatility  # 95% VaR
        var_99 = -2.58 * portfolio_volatility  # 99% VaR
        
        # 计算组合beta（如果有市场指数）
        portfolio_beta = self._calculate_portfolio_beta(aligned_weights, aligned_cov)
        
        risk_metrics = {
            'volatility': portfolio_volatility,
            'variance': portfolio_variance,
            'var_95': var_95,
            'var_99': var_99,
            'portfolio_beta': portfolio_beta,
            'diversification_ratio': self.calculate_diversification_ratio(weights)
        }
        
        return risk_metrics
    
    def decompose_risk(self, weights: pd.Series) -> Dict[str, Any]:
        """
        风险分解
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
            
        Returns
        -------
        Dict[str, Any]
            风险分解结果
        """
        self._check_fitted()
        self.validate_weights(weights)
        
        # 获取协方差矩阵
        cov_matrix = self.covariance_matrix_
        
        # 对齐权重和协方差矩阵
        common_assets = cov_matrix.index.intersection(weights.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_cov = cov_matrix.reindex(index=common_assets, columns=common_assets)
        
        # 计算总风险
        total_variance = np.dot(aligned_weights.values,
                              np.dot(aligned_cov.values, aligned_weights.values))
        total_volatility = np.sqrt(total_variance)
        
        # 计算边际风险贡献
        marginal_contrib = np.dot(aligned_cov.values, aligned_weights.values)
        
        # 计算风险贡献
        risk_contributions = aligned_weights.values * marginal_contrib
        
        # 计算风险贡献百分比
        risk_contrib_pct = risk_contributions / total_variance
        
        # 计算资产集中度
        concentration = self._calculate_concentration(aligned_weights)
        
        decomposition = {
            'total_risk': total_volatility,
            'total_variance': total_variance,
            'risk_contributions': pd.Series(risk_contributions, index=common_assets),
            'risk_contrib_pct': pd.Series(risk_contrib_pct, index=common_assets),
            'marginal_contributions': pd.Series(marginal_contrib, index=common_assets),
            'concentration_ratio': concentration,
            'effective_assets': self._calculate_effective_assets(aligned_weights),
            'correlation_impact': self._calculate_correlation_impact(aligned_weights, aligned_cov)
        }
        
        return decomposition
    
    def stress_test(self, 
                   weights: pd.Series,
                   stress_scenarios: Dict[str, Dict]) -> Dict[str, Any]:
        """
        压力测试
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
        stress_scenarios : Dict[str, Dict]
            压力场景，格式：{scenario_name: {'cov_multiplier': float, 'correlation_shift': float}}
            
        Returns
        -------
        Dict[str, Any]
            压力测试结果
        """
        self._check_fitted()
        self.validate_weights(weights)
        
        base_risk = self.calculate_portfolio_risk(weights)
        stress_results = {'base_case': base_risk}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            try:
                # 构造压力协方差矩阵
                stressed_cov = self._create_stressed_covariance(scenario_params)
                
                # 计算压力下的组合风险
                stressed_variance = np.dot(weights.values,
                                         np.dot(stressed_cov.values, weights.values))
                stressed_volatility = np.sqrt(stressed_variance)
                
                stress_results[scenario_name] = {
                    'volatility': stressed_volatility,
                    'volatility_change': stressed_volatility - base_risk['volatility'],
                    'volatility_change_pct': (stressed_volatility / base_risk['volatility'] - 1) * 100
                }
                
            except Exception as e:
                logger.warning(f"Failed to compute stress scenario {scenario_name}: {e}")
                stress_results[scenario_name] = {'error': str(e)}
        
        return stress_results
    
    def _prepare_returns_data(self, returns: pd.Series) -> pd.DataFrame:
        """
        准备收益率数据
        
        Parameters
        ----------
        returns : pd.Series
            MultiIndex收益率数据
            
        Returns
        -------
        pd.DataFrame
            DataFrame格式的收益率数据
        """
        if isinstance(returns.index, pd.MultiIndex):
            # 转换MultiIndex到DataFrame
            returns_df = returns.unstack()
            
            # 处理缺失值
            if self.handle_missing == 'drop':
                returns_df = returns_df.dropna(axis=1, how='any')
            elif self.handle_missing == 'forward_fill':
                returns_df = returns_df.fillna(method='ffill')
            elif self.handle_missing == 'interpolate':
                returns_df = returns_df.interpolate()
                
        else:
            # 假设是单一时间序列
            returns_df = pd.DataFrame(returns)
        
        # 检查数据量
        if len(returns_df) < self.min_observations:
            raise InsufficientDataError(self.min_observations, len(returns_df),
                                      "return observations")
        
        return returns_df
    
    def _create_estimator(self):
        """创建协方差估计器"""
        if self.estimator_method == 'sample':
            self.estimator_ = SampleCovarianceEstimator(self.estimator_config)
        elif self.estimator_method == 'ledoit_wolf':
            self.estimator_ = LedoitWolfEstimator(self.estimator_config)
        elif self.estimator_method == 'exponential_weighted':
            self.estimator_ = ExponentialWeightedEstimator(self.estimator_config)
        elif self.estimator_method == 'robust':
            self.estimator_ = RobustCovarianceEstimator(self.estimator_config)
        else:
            raise InvalidParameterError('estimator_method', self.estimator_method,
                                      "{'sample', 'ledoit_wolf', 'exponential_weighted', 'robust'}")
    
    def _apply_regularization(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """应用正则化"""
        if self.regularization['method'] == 'ridge':
            alpha = self.regularization.get('alpha', 0.01)
            regularized = cov_matrix + alpha * np.eye(len(cov_matrix))
            
        elif self.regularization['method'] == 'eigenvalue_clipping':
            min_eigenvalue = self.regularization.get('min_eigenvalue', 1e-6)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix.values)
            eigenvals = np.maximum(eigenvals, min_eigenvalue)
            regularized_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            regularized = pd.DataFrame(regularized_matrix,
                                     index=cov_matrix.index,
                                     columns=cov_matrix.columns)
        else:
            regularized = cov_matrix
        
        return regularized
    
    def _validate_covariance_matrix(self):
        """验证协方差矩阵"""
        # 检查正定性
        eigenvals = np.linalg.eigvals(self.covariance_matrix_.values)
        if np.any(eigenvals <= 1e-8):
            min_eigenval = np.min(eigenvals)
            logger.warning(f"Covariance matrix has small eigenvalue: {min_eigenval}")
            
            if min_eigenval < -1e-6:
                raise SingularCovarianceError("covariance matrix", 
                                            np.linalg.cond(self.covariance_matrix_.values))
        
        # 检查对称性
        if not np.allclose(self.covariance_matrix_.values, 
                          self.covariance_matrix_.values.T, atol=1e-10):
            logger.warning("Covariance matrix is not symmetric, forcing symmetry")
            symmetric_cov = (self.covariance_matrix_ + self.covariance_matrix_.T) / 2
            self.covariance_matrix_ = symmetric_cov
    
    def _forecast_with_mean_reversion(self, horizon: int) -> pd.DataFrame:
        """均值回归预测"""
        # 简化的均值回归模型
        long_term_vol = self.returns_data_.std() * np.sqrt(252)  # 年化波动率
        current_vol = np.sqrt(np.diag(self.covariance_matrix_)) * np.sqrt(252)
        
        # 半衰期
        half_life = self.config.get('mean_reversion_half_life', 60)
        decay_factor = np.exp(-np.log(2) / half_life * horizon)
        
        # 预测波动率
        forecast_vol = current_vol * decay_factor + long_term_vol * (1 - decay_factor)
        forecast_vol /= np.sqrt(252)  # 转回日频
        
        # 保持相关系数不变，更新方差
        correlation = self.covariance_matrix_.values / np.outer(
            np.sqrt(np.diag(self.covariance_matrix_)), 
            np.sqrt(np.diag(self.covariance_matrix_))
        )
        
        forecast_cov = correlation * np.outer(forecast_vol, forecast_vol)
        
        return pd.DataFrame(forecast_cov,
                          index=self.covariance_matrix_.index,
                          columns=self.covariance_matrix_.columns)
    
    def _calculate_portfolio_beta(self, weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """计算组合beta（相对于等权重市场组合）"""
        # 使用等权重作为市场组合代理
        market_weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        # 计算组合与市场的协方差
        portfolio_market_cov = np.dot(weights.values,
                                    np.dot(cov_matrix.values, market_weights.values))
        
        # 计算市场方差
        market_variance = np.dot(market_weights.values,
                               np.dot(cov_matrix.values, market_weights.values))
        
        # 计算beta
        if market_variance > 0:
            beta = portfolio_market_cov / market_variance
        else:
            beta = 1.0
        
        return beta
    
    def _calculate_concentration(self, weights: pd.Series) -> float:
        """计算权重集中度（Herfindahl指数）"""
        return np.sum(weights.values ** 2)
    
    def _calculate_effective_assets(self, weights: pd.Series) -> float:
        """计算有效资产数量"""
        return 1.0 / np.sum(weights.values ** 2)
    
    def _calculate_correlation_impact(self, weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """计算相关性对风险的影响"""
        # 计算无相关性风险（仅方差项）
        individual_risk = np.sum((weights.values ** 2) * np.diag(cov_matrix.values))
        
        # 计算总风险
        total_risk = np.dot(weights.values, np.dot(cov_matrix.values, weights.values))
        
        # 相关性贡献
        correlation_contribution = total_risk - individual_risk
        
        return correlation_contribution / total_risk if total_risk > 0 else 0
    
    def _create_stressed_covariance(self, scenario_params: Dict) -> pd.DataFrame:
        """创建压力协方差矩阵"""
        stressed_cov = self.covariance_matrix_.copy()
        
        # 波动率乘数
        if 'vol_multiplier' in scenario_params:
            vol_mult = scenario_params['vol_multiplier']
            stressed_cov *= (vol_mult ** 2)
        
        # 相关性冲击
        if 'correlation_shift' in scenario_params:
            corr_shift = scenario_params['correlation_shift']
            
            # 提取波动率
            vols = np.sqrt(np.diag(stressed_cov.values))
            
            # 计算相关系数矩阵
            correlation = stressed_cov.values / np.outer(vols, vols)
            
            # 调整相关系数
            np.fill_diagonal(correlation, 1.0)
            correlation = np.clip(correlation + corr_shift, -0.99, 0.99)
            np.fill_diagonal(correlation, 1.0)
            
            # 重构协方差矩阵
            stressed_cov = pd.DataFrame(
                correlation * np.outer(vols, vols),
                index=stressed_cov.index,
                columns=stressed_cov.columns
            )
        
        return stressed_cov
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        获取当前协方差矩阵
        
        Returns
        -------
        pd.DataFrame
            协方差矩阵
        """
        self._check_fitted()
        return self.covariance_matrix_.copy()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        获取相关系数矩阵
        
        Returns
        -------
        pd.DataFrame
            相关系数矩阵
        """
        self._check_fitted()
        
        vols = np.sqrt(np.diag(self.covariance_matrix_.values))
        correlation = self.covariance_matrix_.values / np.outer(vols, vols)
        
        return pd.DataFrame(correlation,
                          index=self.covariance_matrix_.index,
                          columns=self.covariance_matrix_.columns)
    
    def get_asset_volatilities(self) -> pd.Series:
        """
        获取资产波动率
        
        Returns
        -------
        pd.Series
            各资产波动率
        """
        self._check_fitted()
        
        volatilities = np.sqrt(np.diag(self.covariance_matrix_.values))
        return pd.Series(volatilities, index=self.covariance_matrix_.index, name='volatility')