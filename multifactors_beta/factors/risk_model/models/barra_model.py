"""
Barra多因子风险模型

实现Barra风险模型，包括因子收益估计、因子协方差矩阵和特异性风险
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, List, Union, Tuple
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression

from ..base.risk_model_base import RiskModelBase
from ..base.exceptions import (
    ModelNotFittedError,
    SingularCovarianceError,
    CalculationError,
    InvalidParameterError,
    InsufficientDataError
)
from ..estimators.ledoit_wolf import LedoitWolfEstimator
from ..estimators.exponential_weighted import ExponentialWeightedEstimator

logger = logging.getLogger(__name__)


class BarraModel(RiskModelBase):
    """
    Barra多因子风险模型
    
    实现Barra风险模型框架：
    r_i = sum(X_ik * f_k) + u_i
    
    其中：
    - r_i: 股票i的收益率
    - X_ik: 股票i在因子k上的暴露度
    - f_k: 因子k的收益率
    - u_i: 股票i的特异性收益率
    """
    
    def __init__(self, 
                 style_factors: Optional[List[str]] = None,
                 industry_factors: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化Barra模型
        
        Parameters
        ----------
        style_factors : List[str], optional
            风格因子列表
        industry_factors : List[str], optional
            行业因子列表
        config : Dict[str, Any], optional
            模型配置参数
        """
        super().__init__(config)
        
        # 因子配置
        self.style_factors = style_factors or ['momentum', 'value', 'quality', 'size', 'volatility']
        self.industry_factors = industry_factors or []
        self.all_factors = self.style_factors + self.industry_factors
        
        # 模型特定配置
        self.factor_cov_method = self.config.get('factor_cov_method', 'exponential_weighted')
        self.specific_risk_method = self.config.get('specific_risk_method', 'bayesian_shrinkage')
        self.regression_method = self.config.get('regression_method', 'wls')  # weighted least squares
        self.half_life = self.config.get('half_life', 90)  # 半衰期
        self.newey_west_lags = self.config.get('newey_west_lags', 5)  # Newey-West滞后
        self.volatility_regime_adjust = self.config.get('volatility_regime_adjust', True)
        
        # 估计结果
        self.factor_returns_ = None  # 因子收益率时间序列
        self.factor_covariance_ = None  # 因子协方差矩阵
        self.specific_risk_ = None  # 特异性风险
        self.factor_loadings_ = None  # 因子载荷（最新）
        self.regression_stats_ = None  # 回归统计
        self.r_squared_ = None  # R²统计
        
        # 数据存储
        self.factor_exposures_data_ = None
        self.returns_data_ = None
        self.asset_universe_ = None
        
        logger.info(f"Initialized BarraModel with {len(self.style_factors)} style factors "
                   f"and {len(self.industry_factors)} industry factors")
    
    def fit(self, 
            factor_exposures: pd.DataFrame,
            returns: pd.Series,
            **kwargs) -> 'BarraModel':
        """
        拟合Barra模型
        
        Parameters
        ----------
        factor_exposures : pd.DataFrame
            因子暴露度矩阵，MultiIndex(date, stock) x factors
        returns : pd.Series  
            股票收益率，MultiIndex(date, stock)
        **kwargs : dict
            其他参数
            
        Returns
        -------
        BarraModel
            拟合后的模型实例
        """
        # 验证和对齐数据
        self.validate_factor_exposures(factor_exposures)
        self.validate_returns(returns)
        
        aligned_exposures, aligned_returns = self.align_data(factor_exposures, returns)
        
        # 检查因子完整性
        missing_factors = set(self.all_factors) - set(aligned_exposures.columns)
        if missing_factors:
            logger.warning(f"Missing factors in exposures: {missing_factors}")
            # 使用可用因子
            available_factors = [f for f in self.all_factors if f in aligned_exposures.columns]
            aligned_exposures = aligned_exposures[available_factors]
        
        # 保存数据
        self.factor_exposures_data_ = aligned_exposures.copy()
        self.returns_data_ = aligned_returns.copy()
        self.asset_universe_ = aligned_exposures.columns.intersection(
            aligned_returns.index.get_level_values(1)
        ).unique().tolist()
        
        try:
            # 1. 估计因子收益率
            self.factor_returns_ = self._estimate_factor_returns(aligned_exposures, aligned_returns)
            
            # 2. 估计因子协方差矩阵
            self.factor_covariance_ = self._estimate_factor_covariance(self.factor_returns_)
            
            # 3. 估计特异性风险
            self.specific_risk_ = self._estimate_specific_risk(
                aligned_exposures, aligned_returns, self.factor_returns_
            )
            
            # 4. 计算回归统计
            self._calculate_regression_statistics(aligned_exposures, aligned_returns)
            
            # 5. 保存最新因子载荷
            latest_date = aligned_exposures.index.get_level_values(0)[-1]
            self.factor_loadings_ = aligned_exposures.xs(latest_date, level=0)
            
            # 更新模型状态
            n_observations = len(aligned_exposures.index.get_level_values(0).unique())
            n_assets = len(self.asset_universe_)
            n_factors = len(self.factor_returns_.columns)
            
            self._log_fit_completion(n_observations, n_assets, n_factors)
            
            # 保存模型参数
            self.model_params = {
                'style_factors': self.style_factors,
                'industry_factors': self.industry_factors,
                'n_factors': n_factors,
                'n_assets': n_assets,
                'n_observations': n_observations,
                'factor_cov_method': self.factor_cov_method,
                'specific_risk_method': self.specific_risk_method,
                'avg_r_squared': self.r_squared_.mean() if self.r_squared_ is not None else None,
                'factor_cov_condition_number': np.linalg.cond(self.factor_covariance_.values),
                'min_specific_risk': self.specific_risk_.min(),
                'max_specific_risk': self.specific_risk_.max()
            }
            
            logger.info(f"Successfully fitted BarraModel with average R²: "
                       f"{self.model_params['avg_r_squared']:.3f}")
            
        except Exception as e:
            raise CalculationError("Barra model fitting", str(e))
        
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
        
        # 获取当前因子载荷
        if self.factor_loadings_ is None:
            raise ModelNotFittedError("Factor loadings not available")
        
        # 预测因子协方差矩阵
        if method == 'default':
            forecast_factor_cov = self.factor_covariance_ * horizon
        elif method == 'volatility_scaling':
            forecast_factor_cov = self._forecast_factor_covariance_with_scaling(horizon)
        else:
            forecast_factor_cov = self.factor_covariance_ * horizon
        
        # 预测特异性风险
        forecast_specific_risk = self._forecast_specific_risk(horizon)
        
        # 构建完整协方差矩阵
        # Cov = X * F * X' + D
        # 其中 X 是因子载荷矩阵，F 是因子协方差矩阵，D 是特异性风险对角矩阵
        
        # 获取共同资产
        common_assets = self.factor_loadings_.index.intersection(forecast_specific_risk.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between factor loadings and specific risk")
        
        # 对齐数据
        X = self.factor_loadings_.reindex(common_assets)  # N x K
        F = forecast_factor_cov  # K x K
        D = np.diag(forecast_specific_risk.reindex(common_assets).values)  # N x N
        
        # 计算系统性风险: X * F * X'
        systematic_cov = np.dot(X.values, np.dot(F.values, X.values.T))
        
        # 加上特异性风险
        full_covariance = systematic_cov + D
        
        # 转换为DataFrame
        covariance_matrix = pd.DataFrame(
            full_covariance,
            index=common_assets,
            columns=common_assets
        )
        
        return covariance_matrix
    
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
        
        # 计算总风险
        total_variance = np.dot(aligned_weights.values,
                              np.dot(aligned_cov.values, aligned_weights.values))
        total_volatility = np.sqrt(total_variance)
        
        # 分解系统性风险和特异性风险
        risk_decomp = self._decompose_portfolio_risk(aligned_weights)
        
        # 计算VaR
        var_95 = -1.96 * total_volatility
        var_99 = -2.58 * total_volatility
        
        risk_metrics = {
            'volatility': total_volatility,
            'variance': total_variance,
            'systematic_risk': risk_decomp['systematic_risk'],
            'specific_risk': risk_decomp['specific_risk'],
            'factor_risk_pct': risk_decomp['factor_risk_pct'],
            'specific_risk_pct': risk_decomp['specific_risk_pct'],
            'var_95': var_95,
            'var_99': var_99,
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
        
        # 基本风险分解
        basic_decomp = self._decompose_portfolio_risk(weights)
        
        # 因子风险贡献
        factor_contrib = self._calculate_factor_contributions(weights)
        
        # 资产风险贡献
        asset_contrib = self._calculate_asset_contributions(weights)
        
        decomposition = {
            **basic_decomp,
            'factor_contributions': factor_contrib,
            'asset_contributions': asset_contrib,
            'concentration_analysis': self._analyze_concentration(weights),
            'factor_exposures': self._calculate_portfolio_exposures(weights)
        }
        
        return decomposition
    
    def _estimate_factor_returns(self, 
                                exposures: pd.DataFrame, 
                                returns: pd.Series) -> pd.DataFrame:
        """
        估计因子收益率
        
        使用横截面回归估计每日因子收益率
        """
        factor_returns_list = []
        regression_stats_list = []
        
        # 获取所有日期
        dates = exposures.index.get_level_values(0).unique().sort_values()
        
        for date in dates:
            try:
                # 获取当日数据
                daily_exposures = exposures.xs(date, level=0)
                daily_returns = returns.xs(date, level=0)
                
                # 对齐数据
                common_assets = daily_exposures.index.intersection(daily_returns.index)
                if len(common_assets) < self.min_observations:
                    continue
                
                X = daily_exposures.reindex(common_assets)
                y = daily_returns.reindex(common_assets)
                
                # 去除缺失值
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) < len(X.columns):  # 样本数少于因子数
                    continue
                
                # 回归估计
                if self.regression_method == 'ols':
                    factor_ret, stats = self._ols_regression(X_clean, y_clean)
                elif self.regression_method == 'wls':
                    factor_ret, stats = self._wls_regression(X_clean, y_clean, date)
                else:
                    factor_ret, stats = self._robust_regression(X_clean, y_clean)
                
                factor_returns_list.append(pd.Series(factor_ret, index=X.columns, name=date))
                regression_stats_list.append(stats)
                
            except Exception as e:
                logger.warning(f"Failed to estimate factor returns for {date}: {e}")
                continue
        
        if not factor_returns_list:
            raise CalculationError("factor returns estimation", "No valid regression results")
        
        # 组合结果
        factor_returns_df = pd.DataFrame(factor_returns_list)
        self.regression_stats_ = regression_stats_list
        
        return factor_returns_df
    
    def _ols_regression(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict]:
        """普通最小二乘回归"""
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X.values, y.values)
        
        # 计算统计量
        y_pred = reg.predict(X.values)
        residuals = y.values - y_pred
        mse = np.mean(residuals ** 2)
        r_squared = 1 - np.sum(residuals ** 2) / np.sum((y.values - np.mean(y.values)) ** 2)
        
        stats = {
            'r_squared': r_squared,
            'mse': mse,
            'n_assets': len(X)
        }
        
        return reg.coef_, stats
    
    def _wls_regression(self, X: pd.DataFrame, y: pd.Series, date) -> Tuple[np.ndarray, Dict]:
        """加权最小二乘回归"""
        # 计算权重（基于市值或流动性，这里使用简化版本）
        weights = self._calculate_regression_weights(X.index, date)
        
        # 加权回归
        W = np.diag(weights)
        X_weighted = np.sqrt(W) @ X.values
        y_weighted = np.sqrt(W) @ y.values
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_weighted, y_weighted)
        
        # 统计量
        y_pred = reg.predict(X_weighted)
        residuals = y_weighted - y_pred
        mse = np.mean(residuals ** 2)
        
        # 计算加权R²
        y_mean_weighted = np.average(y.values, weights=weights)
        ss_tot = np.sum(weights * (y.values - y_mean_weighted) ** 2)
        ss_res = np.sum(weights * (y.values - (X.values @ reg.coef_)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        stats = {
            'r_squared': r_squared,
            'mse': mse,
            'n_assets': len(X),
            'weighted': True
        }
        
        return reg.coef_, stats
    
    def _robust_regression(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict]:
        """稳健回归（简化版）"""
        from sklearn.linear_model import HuberRegressor
        
        reg = HuberRegressor(fit_intercept=False, max_iter=100)
        reg.fit(X.values, y.values)
        
        y_pred = reg.predict(X.values)
        residuals = y.values - y_pred
        mse = np.mean(residuals ** 2)
        r_squared = 1 - np.sum(residuals ** 2) / np.sum((y.values - np.mean(y.values)) ** 2)
        
        stats = {
            'r_squared': r_squared,
            'mse': mse,
            'n_assets': len(X),
            'robust': True
        }
        
        return reg.coef_, stats
    
    def _calculate_regression_weights(self, assets: pd.Index, date) -> np.ndarray:
        """计算回归权重（简化版本，实际应基于市值）"""
        # 简化：使用等权重
        return np.ones(len(assets))
    
    def _estimate_factor_covariance(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """估计因子协方差矩阵"""
        if self.factor_cov_method == 'sample':
            factor_cov = factor_returns.cov()
            
        elif self.factor_cov_method == 'exponential_weighted':
            # 使用指数加权
            estimator = ExponentialWeightedEstimator({
                'decay_factor': 2.0 / (self.half_life + 1.0),
                'min_periods': 30
            })
            estimator.fit(factor_returns)
            factor_cov = estimator.get_covariance_matrix()
            
        elif self.factor_cov_method == 'ledoit_wolf':
            # 使用Ledoit-Wolf收缩
            estimator = LedoitWolfEstimator({'shrinkage_target': 'diagonal'})
            estimator.fit(factor_returns)
            factor_cov = estimator.get_covariance_matrix()
            
        else:
            factor_cov = factor_returns.cov()
        
        # Newey-West调整（处理序列相关性）
        if self.newey_west_lags > 0:
            factor_cov = self._newey_west_adjustment(factor_returns, factor_cov)
        
        return factor_cov
    
    def _newey_west_adjustment(self, factor_returns: pd.DataFrame, base_cov: pd.DataFrame) -> pd.DataFrame:
        """Newey-West调整"""
        n_obs, n_factors = factor_returns.shape
        
        # 计算去均值收益率
        demeaned_returns = factor_returns - factor_returns.mean()
        
        # 计算滞后协方差
        adjustment = np.zeros((n_factors, n_factors))
        
        for lag in range(1, self.newey_west_lags + 1):
            # 计算滞后协方差
            lag_cov = np.zeros((n_factors, n_factors))
            
            for i in range(lag, n_obs):
                outer_prod = np.outer(demeaned_returns.iloc[i].values, 
                                    demeaned_returns.iloc[i-lag].values)
                lag_cov += outer_prod + outer_prod.T
            
            lag_cov /= n_obs
            
            # Newey-West权重
            weight = 1 - lag / (self.newey_west_lags + 1)
            adjustment += weight * lag_cov
        
        # 应用调整
        adjusted_cov = base_cov.values + adjustment
        
        return pd.DataFrame(adjusted_cov, index=base_cov.index, columns=base_cov.columns)
    
    def _estimate_specific_risk(self, 
                               exposures: pd.DataFrame, 
                               returns: pd.Series,
                               factor_returns: pd.DataFrame) -> pd.Series:
        """估计特异性风险"""
        specific_returns_dict = {}
        
        # 计算每日特异性收益率
        for date in factor_returns.index:
            try:
                # 获取当日数据
                daily_exposures = exposures.xs(date, level=0)
                daily_returns = returns.xs(date, level=0)
                daily_factor_returns = factor_returns.loc[date]
                
                # 对齐数据
                common_assets = daily_exposures.index.intersection(daily_returns.index)
                X = daily_exposures.reindex(common_assets)
                y = daily_returns.reindex(common_assets)
                
                # 计算预测收益率
                predicted_returns = X @ daily_factor_returns
                
                # 计算特异性收益率
                specific_returns = y - predicted_returns
                specific_returns_dict[date] = specific_returns
                
            except Exception as e:
                logger.warning(f"Failed to calculate specific returns for {date}: {e}")
                continue
        
        # 组合特异性收益率
        specific_returns_df = pd.DataFrame(specific_returns_dict).T
        
        # 估计特异性风险
        if self.specific_risk_method == 'sample':
            specific_risk = specific_returns_df.std()
            
        elif self.specific_risk_method == 'exponential_weighted':
            # 指数加权标准差
            specific_risk = specific_returns_df.ewm(
                alpha=2.0/(self.half_life + 1), min_periods=10
            ).std().iloc[-1]
            
        elif self.specific_risk_method == 'bayesian_shrinkage':
            # 贝叶斯收缩
            specific_risk = self._bayesian_shrinkage_specific_risk(specific_returns_df)
            
        else:
            specific_risk = specific_returns_df.std()
        
        # 去除缺失值
        specific_risk = specific_risk.dropna()
        
        # 确保为正数
        specific_risk = specific_risk.abs()
        
        return specific_risk
    
    def _bayesian_shrinkage_specific_risk(self, specific_returns: pd.DataFrame) -> pd.Series:
        """贝叶斯收缩特异性风险估计"""
        # 计算样本方差
        sample_var = specific_returns.var()
        
        # 计算收缩目标（横截面方差的中位数）
        target_var = sample_var.median()
        
        # 计算收缩参数
        n_obs = len(specific_returns)
        shrinkage = min(1.0, (n_obs - 3) / (n_obs * sample_var.var() / target_var**2 + n_obs - 3))
        
        # 应用收缩
        shrunk_var = (1 - shrinkage) * sample_var + shrinkage * target_var
        
        return np.sqrt(shrunk_var)
    
    def _calculate_regression_statistics(self, exposures: pd.DataFrame, returns: pd.Series):
        """计算回归统计"""
        if self.regression_stats_:
            # 计算R²统计
            r_squared_values = [stat['r_squared'] for stat in self.regression_stats_]
            self.r_squared_ = pd.Series(r_squared_values, 
                                      index=self.factor_returns_.index,
                                      name='r_squared')
    
    def _decompose_portfolio_risk(self, weights: pd.Series) -> Dict[str, float]:
        """分解组合风险为系统性和特异性"""
        # 获取因子载荷和特异性风险
        common_assets = (self.factor_loadings_.index
                        .intersection(weights.index)
                        .intersection(self.specific_risk_.index))
        
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_loadings = self.factor_loadings_.reindex(common_assets)
        aligned_specific_risk = self.specific_risk_.reindex(common_assets)
        
        # 计算因子暴露度
        portfolio_exposures = aligned_loadings.T @ aligned_weights
        
        # 系统性风险
        systematic_variance = portfolio_exposures.T @ self.factor_covariance_ @ portfolio_exposures
        systematic_risk = np.sqrt(systematic_variance)
        
        # 特异性风险
        specific_variance = np.sum((aligned_weights ** 2) * (aligned_specific_risk ** 2))
        specific_risk = np.sqrt(specific_variance)
        
        # 总风险
        total_variance = systematic_variance + specific_variance
        total_risk = np.sqrt(total_variance)
        
        return {
            'total_risk': total_risk,
            'systematic_risk': systematic_risk,
            'specific_risk': specific_risk,
            'factor_risk_pct': systematic_variance / total_variance * 100,
            'specific_risk_pct': specific_variance / total_variance * 100
        }
    
    def _calculate_factor_contributions(self, weights: pd.Series) -> pd.Series:
        """计算各因子的风险贡献"""
        common_assets = (self.factor_loadings_.index
                        .intersection(weights.index)
                        .intersection(self.specific_risk_.index))
        
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_loadings = self.factor_loadings_.reindex(common_assets)
        
        # 计算组合在各因子上的暴露度
        portfolio_exposures = aligned_loadings.T @ aligned_weights
        
        # 计算各因子的边际风险贡献
        marginal_contrib = self.factor_covariance_ @ portfolio_exposures
        
        # 风险贡献 = 暴露度 × 边际贡献
        factor_contributions = portfolio_exposures * marginal_contrib
        
        return factor_contributions
    
    def _calculate_asset_contributions(self, weights: pd.Series) -> pd.Series:
        """计算各资产的风险贡献"""
        # 获取协方差矩阵
        cov_matrix = self.predict_covariance()
        
        # 对齐数据
        common_assets = cov_matrix.index.intersection(weights.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_cov = cov_matrix.reindex(index=common_assets, columns=common_assets)
        
        # 计算边际风险贡献
        marginal_contrib = aligned_cov @ aligned_weights
        
        # 风险贡献
        asset_contributions = aligned_weights * marginal_contrib
        
        return asset_contributions
    
    def _analyze_concentration(self, weights: pd.Series) -> Dict[str, float]:
        """分析权重集中度"""
        abs_weights = weights.abs()
        
        return {
            'herfindahl_index': np.sum(abs_weights ** 2),
            'effective_assets': 1.0 / np.sum(abs_weights ** 2),
            'max_weight': abs_weights.max(),
            'top_5_weight': abs_weights.nlargest(5).sum(),
            'top_10_weight': abs_weights.nlargest(10).sum()
        }
    
    def _calculate_portfolio_exposures(self, weights: pd.Series) -> pd.Series:
        """计算组合因子暴露度"""
        common_assets = self.factor_loadings_.index.intersection(weights.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_loadings = self.factor_loadings_.reindex(common_assets)
        
        portfolio_exposures = aligned_loadings.T @ aligned_weights
        
        return portfolio_exposures
    
    def _forecast_factor_covariance_with_scaling(self, horizon: int) -> pd.DataFrame:
        """带波动率调整的因子协方差预测"""
        # 基础预测
        base_forecast = self.factor_covariance_ * horizon
        
        if not self.volatility_regime_adjust:
            return base_forecast
        
        # 波动率制度调整（简化版）
        recent_vol = self.factor_returns_.tail(20).std()
        long_term_vol = self.factor_returns_.std()
        
        vol_ratio = recent_vol / long_term_vol
        
        # 调整因子
        scaling_factors = np.clip(vol_ratio, 0.5, 2.0)
        
        # 应用调整
        adjusted_forecast = base_forecast * np.outer(scaling_factors, scaling_factors)
        
        return pd.DataFrame(adjusted_forecast,
                          index=base_forecast.index,
                          columns=base_forecast.columns)
    
    def _forecast_specific_risk(self, horizon: int) -> pd.Series:
        """预测特异性风险"""
        # 简单时间缩放
        return self.specific_risk_ * np.sqrt(horizon)
    
    def get_factor_returns(self) -> pd.DataFrame:
        """获取因子收益率"""
        self._check_fitted()
        return self.factor_returns_.copy()
    
    def get_factor_covariance(self) -> pd.DataFrame:
        """获取因子协方差矩阵"""
        self._check_fitted()
        return self.factor_covariance_.copy()
    
    def get_specific_risk(self) -> pd.Series:
        """获取特异性风险"""
        self._check_fitted()
        return self.specific_risk_.copy()
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """获取最新因子载荷"""
        self._check_fitted()
        return self.factor_loadings_.copy()
    
    def get_regression_statistics(self) -> pd.Series:
        """获取回归统计"""
        self._check_fitted()
        return self.r_squared_.copy() if self.r_squared_ is not None else None