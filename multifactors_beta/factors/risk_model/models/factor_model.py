"""
通用因子模型

提供灵活的因子建模框架，支持自定义因子和建模方法
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, List, Union, Tuple
from datetime import datetime

from ..base.risk_model_base import RiskModelBase
from ..base.exceptions import (
    ModelNotFittedError,
    CalculationError,
    InvalidParameterError
)
from .barra_model import BarraModel
from .covariance_model import CovarianceModel

logger = logging.getLogger(__name__)


class FactorModel(RiskModelBase):
    """
    通用因子风险模型
    
    提供灵活的因子建模框架，可以选择不同的建模方法：
    - Barra风格的多因子模型
    - 基于PCA的统计因子模型
    - 混合模型
    """
    
    def __init__(self, 
                 factors: Optional[List[str]] = None,
                 model_type: str = 'barra',
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化因子模型
        
        Parameters
        ----------
        factors : List[str], optional
            因子列表
        model_type : str
            模型类型 {'barra', 'pca', 'hybrid', 'covariance'}
        config : Dict[str, Any], optional
            模型配置参数
        """
        super().__init__(config)
        
        self.factors = factors or []
        self.model_type = model_type
        self.n_pca_factors = self.config.get('n_pca_factors', 10)
        self.pca_var_threshold = self.config.get('pca_var_threshold', 0.9)
        
        # 内部模型实例
        self.underlying_model_ = None
        self.pca_factors_ = None
        self.factor_loadings_ = None
        
        # 创建底层模型
        self._create_underlying_model()
        
        logger.info(f"Initialized FactorModel with type: {model_type}")
    
    def fit(self, 
            factor_exposures: pd.DataFrame,
            returns: pd.Series,
            **kwargs) -> 'FactorModel':
        """
        拟合因子模型
        
        Parameters
        ----------
        factor_exposures : pd.DataFrame
            因子暴露度矩阵
        returns : pd.Series  
            股票收益率
        **kwargs : dict
            其他参数
            
        Returns
        -------
        FactorModel
            拟合后的模型实例
        """
        if self.model_type in ['barra', 'covariance']:
            # 直接使用底层模型
            self.underlying_model_.fit(factor_exposures, returns, **kwargs)
            
        elif self.model_type == 'pca':
            # PCA因子模型
            self._fit_pca_model(factor_exposures, returns, **kwargs)
            
        elif self.model_type == 'hybrid':
            # 混合模型
            self._fit_hybrid_model(factor_exposures, returns, **kwargs)
            
        else:
            raise InvalidParameterError('model_type', self.model_type,
                                      "{'barra', 'pca', 'hybrid', 'covariance'}")
        
        # 更新状态
        self.is_fitted = True
        self.fit_timestamp = datetime.now()
        
        return self
    
    def predict_covariance(self, 
                          horizon: int = 1,
                          method: str = 'default') -> pd.DataFrame:
        """预测协方差矩阵"""
        self._check_fitted()
        
        if self.model_type in ['barra', 'covariance']:
            return self.underlying_model_.predict_covariance(horizon, method)
        elif self.model_type == 'pca':
            return self._predict_pca_covariance(horizon)
        elif self.model_type == 'hybrid':
            return self._predict_hybrid_covariance(horizon)
    
    def calculate_portfolio_risk(self, 
                                weights: pd.Series,
                                horizon: int = 1) -> Dict[str, float]:
        """计算组合风险"""
        self._check_fitted()
        
        if self.model_type in ['barra', 'covariance']:
            return self.underlying_model_.calculate_portfolio_risk(weights, horizon)
        else:
            # 使用协方差矩阵计算
            cov_matrix = self.predict_covariance(horizon)
            
            common_assets = cov_matrix.index.intersection(weights.index)
            aligned_weights = weights.reindex(common_assets, fill_value=0)
            aligned_cov = cov_matrix.reindex(index=common_assets, columns=common_assets)
            
            portfolio_variance = np.dot(aligned_weights.values,
                                      np.dot(aligned_cov.values, aligned_weights.values))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            return {
                'volatility': portfolio_volatility,
                'variance': portfolio_variance,
                'var_95': -1.96 * portfolio_volatility,
                'var_99': -2.58 * portfolio_volatility
            }
    
    def decompose_risk(self, weights: pd.Series) -> Dict[str, Any]:
        """风险分解"""
        self._check_fitted()
        
        if self.model_type in ['barra']:
            return self.underlying_model_.decompose_risk(weights)
        else:
            # 基础风险分解
            cov_matrix = self.predict_covariance()
            
            common_assets = cov_matrix.index.intersection(weights.index)
            aligned_weights = weights.reindex(common_assets, fill_value=0)
            aligned_cov = cov_matrix.reindex(index=common_assets, columns=common_assets)
            
            # 计算风险贡献
            marginal_contrib = aligned_cov @ aligned_weights
            risk_contributions = aligned_weights * marginal_contrib
            
            total_variance = np.dot(aligned_weights.values,
                                  np.dot(aligned_cov.values, aligned_weights.values))
            
            return {
                'total_risk': np.sqrt(total_variance),
                'risk_contributions': risk_contributions,
                'marginal_contributions': marginal_contrib
            }
    
    def _create_underlying_model(self):
        """创建底层模型"""
        if self.model_type == 'barra':
            style_factors = [f for f in self.factors if not f.startswith('industry_')]
            industry_factors = [f for f in self.factors if f.startswith('industry_')]
            
            self.underlying_model_ = BarraModel(
                style_factors=style_factors or None,
                industry_factors=industry_factors or None,
                config=self.config
            )
            
        elif self.model_type == 'covariance':
            self.underlying_model_ = CovarianceModel(self.config)
            
        elif self.model_type in ['pca', 'hybrid']:
            # PCA和混合模型不需要预创建底层模型
            pass
    
    def _fit_pca_model(self, factor_exposures: pd.DataFrame, returns: pd.Series, **kwargs):
        """拟合PCA因子模型"""
        from sklearn.decomposition import PCA
        
        # 验证和对齐数据
        self.validate_returns(returns)
        
        # 将收益率转换为DataFrame
        if isinstance(returns.index, pd.MultiIndex):
            returns_df = returns.unstack()
        else:
            returns_df = pd.DataFrame(returns)
        
        # 去除缺失值
        clean_returns = returns_df.dropna(axis=1, how='any')
        
        # 执行PCA
        pca = PCA(n_components=min(self.n_pca_factors, len(clean_returns.columns)))
        
        # 标准化收益率
        standardized_returns = (clean_returns - clean_returns.mean()) / clean_returns.std()
        
        # 拟合PCA
        factor_scores = pca.fit_transform(standardized_returns.T)  # N x K
        factor_loadings = pca.components_.T  # N x K
        
        # 创建因子收益率时间序列
        factor_returns = pd.DataFrame(
            standardized_returns.values @ factor_loadings,
            index=clean_returns.index,
            columns=[f'PC_{i+1}' for i in range(factor_loadings.shape[1])]
        )
        
        # 保存结果
        self.pca_factors_ = {
            'factor_returns': factor_returns,
            'factor_loadings': pd.DataFrame(
                factor_loadings,
                index=clean_returns.columns,
                columns=[f'PC_{i+1}' for i in range(factor_loadings.shape[1])]
            ),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'pca_model': pca
        }
        
        # 计算协方差矩阵
        self._compute_pca_covariance()
        
        logger.info(f"PCA model fitted with {factor_loadings.shape[1]} factors explaining "
                   f"{pca.explained_variance_ratio_.sum():.1%} of variance")
    
    def _fit_hybrid_model(self, factor_exposures: pd.DataFrame, returns: pd.Series, **kwargs):
        """拟合混合模型（结合显式因子和PCA因子）"""
        # 首先拟合Barra模型（如果有显式因子）
        if len(self.factors) > 0 and factor_exposures is not None:
            style_factors = [f for f in self.factors if not f.startswith('industry_')]
            industry_factors = [f for f in self.factors if f.startswith('industry_')]
            
            barra_model = BarraModel(
                style_factors=style_factors or None,
                industry_factors=industry_factors or None,
                config=self.config
            )
            barra_model.fit(factor_exposures, returns, **kwargs)
            
            # 获取残差收益率
            residual_returns = self._calculate_residual_returns(
                factor_exposures, returns, barra_model
            )
        else:
            barra_model = None
            residual_returns = returns
        
        # 对残差收益率应用PCA
        self._fit_pca_model(None, residual_returns, **kwargs)
        
        # 保存混合模型组件
        self.underlying_model_ = barra_model
        
        logger.info("Hybrid model fitted combining explicit factors and PCA factors")
    
    def _calculate_residual_returns(self, 
                                   factor_exposures: pd.DataFrame,
                                   returns: pd.Series,
                                   barra_model: BarraModel) -> pd.Series:
        """计算残差收益率"""
        factor_returns = barra_model.get_factor_returns()
        residual_returns_dict = {}
        
        for date in factor_returns.index:
            try:
                daily_exposures = factor_exposures.xs(date, level=0)
                daily_returns = returns.xs(date, level=0)
                daily_factor_returns = factor_returns.loc[date]
                
                # 计算预测收益率
                common_assets = daily_exposures.index.intersection(daily_returns.index)
                predicted_returns = daily_exposures.reindex(common_assets) @ daily_factor_returns
                
                # 计算残差
                residuals = daily_returns.reindex(common_assets) - predicted_returns
                residual_returns_dict[date] = residuals
                
            except Exception as e:
                logger.warning(f"Failed to calculate residuals for {date}: {e}")
                continue
        
        # 组合结果
        residual_df = pd.DataFrame(residual_returns_dict).T
        residual_series = residual_df.stack()
        residual_series.index.names = ['date', 'stock']
        
        return residual_series
    
    def _compute_pca_covariance(self):
        """计算PCA协方差矩阵"""
        if self.pca_factors_ is None:
            return
        
        factor_returns = self.pca_factors_['factor_returns']
        factor_loadings = self.pca_factors_['factor_loadings']
        
        # 计算因子协方差矩阵
        factor_cov = factor_returns.cov()
        
        # 重构协方差矩阵: Cov = B * F * B'
        # 其中 B 是因子载荷矩阵，F 是因子协方差矩阵
        reconstructed_cov = factor_loadings @ factor_cov @ factor_loadings.T
        
        self.pca_factors_['covariance_matrix'] = reconstructed_cov
    
    def _predict_pca_covariance(self, horizon: int) -> pd.DataFrame:
        """预测PCA协方差矩阵"""
        if self.pca_factors_ is None:
            raise ModelNotFittedError("PCA factors not fitted")
        
        base_cov = self.pca_factors_['covariance_matrix']
        return base_cov * horizon
    
    def _predict_hybrid_covariance(self, horizon: int) -> pd.DataFrame:
        """预测混合模型协方差矩阵"""
        # 结合Barra模型和PCA模型的协方差
        if self.underlying_model_ is not None:
            barra_cov = self.underlying_model_.predict_covariance(horizon)
        else:
            barra_cov = 0
        
        if self.pca_factors_ is not None:
            pca_cov = self._predict_pca_covariance(horizon)
        else:
            pca_cov = 0
        
        # 简单相加（实际应用中可能需要更复杂的组合方法）
        if isinstance(barra_cov, pd.DataFrame) and isinstance(pca_cov, pd.DataFrame):
            # 对齐索引
            common_assets = barra_cov.index.intersection(pca_cov.index)
            combined_cov = (barra_cov.reindex(index=common_assets, columns=common_assets) +
                           pca_cov.reindex(index=common_assets, columns=common_assets))
        elif isinstance(barra_cov, pd.DataFrame):
            combined_cov = barra_cov
        elif isinstance(pca_cov, pd.DataFrame):
            combined_cov = pca_cov
        else:
            raise CalculationError("hybrid covariance prediction", "No valid covariance components")
        
        return combined_cov
    
    def get_factor_information(self) -> Dict[str, Any]:
        """获取因子信息"""
        self._check_fitted()
        
        info = {
            'model_type': self.model_type,
            'factors': self.factors
        }
        
        if self.model_type == 'barra' and self.underlying_model_:
            info.update({
                'style_factors': self.underlying_model_.style_factors,
                'industry_factors': self.underlying_model_.industry_factors,
                'avg_r_squared': self.underlying_model_.model_params.get('avg_r_squared')
            })
        
        elif self.model_type == 'pca' and self.pca_factors_:
            info.update({
                'n_pca_factors': len(self.pca_factors_['factor_loadings'].columns),
                'explained_variance_ratio': self.pca_factors_['explained_variance_ratio'],
                'cumulative_variance_explained': np.cumsum(self.pca_factors_['explained_variance_ratio'])
            })
        
        elif self.model_type == 'hybrid':
            info['has_barra_component'] = self.underlying_model_ is not None
            info['has_pca_component'] = self.pca_factors_ is not None
        
        return info
    
    def get_factor_loadings(self) -> Optional[pd.DataFrame]:
        """获取因子载荷"""
        self._check_fitted()
        
        if self.model_type == 'barra' and self.underlying_model_:
            return self.underlying_model_.get_factor_loadings()
        elif self.model_type == 'pca' and self.pca_factors_:
            return self.pca_factors_['factor_loadings']
        elif self.model_type == 'hybrid':
            # 返回主要组件的载荷
            if self.underlying_model_:
                return self.underlying_model_.get_factor_loadings()
            elif self.pca_factors_:
                return self.pca_factors_['factor_loadings']
        
        return None
    
    def get_factor_returns(self) -> Optional[pd.DataFrame]:
        """获取因子收益率"""
        self._check_fitted()
        
        if self.model_type == 'barra' and self.underlying_model_:
            return self.underlying_model_.get_factor_returns()
        elif self.model_type == 'pca' and self.pca_factors_:
            return self.pca_factors_['factor_returns']
        elif self.model_type == 'hybrid':
            # 可以返回组合的因子收益率或主要组件
            if self.underlying_model_:
                return self.underlying_model_.get_factor_returns()
        
        return None