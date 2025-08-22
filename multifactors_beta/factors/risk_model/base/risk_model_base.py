"""
风险模型基类

定义风险模型的统一接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .exceptions import (
    ModelNotFittedError,
    DataFormatError,
    InsufficientDataError,
    InvalidParameterError
)

logger = logging.getLogger(__name__)


class RiskModelBase(ABC):
    """
    风险模型抽象基类
    
    定义了所有风险模型必须实现的接口和通用功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险模型
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            模型配置参数
        """
        self.config = config or {}
        self.is_fitted = False
        self.model_params = {}
        self.fit_timestamp = None
        self.estimation_universe = []
        
        # 默认配置
        self.lookback_window = self.config.get('lookback_window', 252)  # 回看窗口（天）
        self.min_observations = self.config.get('min_observations', 60)  # 最小观测数
        self.handle_missing = self.config.get('handle_missing', 'drop')  # 缺失值处理
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def fit(self, 
            factor_exposures: pd.DataFrame,
            returns: pd.Series,
            **kwargs) -> 'RiskModelBase':
        """
        拟合风险模型
        
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
        RiskModelBase
            拟合后的模型实例
        """
        pass
    
    @abstractmethod
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
            预测的协方差矩阵，index=stocks, columns=stocks
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_risk(self, 
                                weights: pd.Series,
                                horizon: int = 1) -> Dict[str, float]:
        """
        计算组合风险
        
        Parameters
        ----------
        weights : pd.Series
            组合权重，index=stocks
        horizon : int
            风险预测时间范围
            
        Returns
        -------
        Dict[str, float]
            风险指标字典 {'volatility': float, 'var_95': float, ...}
        """
        pass
    
    @abstractmethod  
    def decompose_risk(self, 
                      weights: pd.Series) -> Dict[str, Any]:
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
        pass
    
    def validate_factor_exposures(self, exposures: pd.DataFrame) -> bool:
        """
        验证因子暴露度数据格式
        
        Parameters
        ----------
        exposures : pd.DataFrame
            因子暴露度数据
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        DataFormatError
            数据格式错误
        """
        # 检查MultiIndex格式
        if not isinstance(exposures.index, pd.MultiIndex):
            raise DataFormatError("MultiIndex format", f"{type(exposures.index).__name__}")
        
        # 检查索引名称
        if exposures.index.names != ['date', 'stock']:
            raise DataFormatError("index names ['date', 'stock']", f"{exposures.index.names}")
        
        # 检查数据类型
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in exposures.dtypes):
            raise DataFormatError("numeric dtypes", f"found non-numeric columns")
        
        # 检查空值
        if exposures.isnull().any().any():
            missing_count = exposures.isnull().sum().sum()
            logger.warning(f"Found {missing_count} missing values in factor exposures")
            
            if self.handle_missing == 'error':
                raise DataFormatError("no missing values", f"{missing_count} missing values found")
        
        # 检查数据量
        n_dates = exposures.index.get_level_values(0).nunique()
        if n_dates < self.min_observations:
            raise InsufficientDataError(self.min_observations, n_dates, "time series observations")
        
        return True
    
    def validate_returns(self, returns: pd.Series) -> bool:
        """
        验证收益率数据格式
        
        Parameters
        ----------
        returns : pd.Series
            收益率数据
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        DataFormatError
            数据格式错误
        """
        # 检查MultiIndex格式
        if not isinstance(returns.index, pd.MultiIndex):
            raise DataFormatError("MultiIndex format", f"{type(returns.index).__name__}")
        
        # 检查索引名称
        if returns.index.names != ['date', 'stock']:
            raise DataFormatError("index names ['date', 'stock']", f"{returns.index.names}")
        
        # 检查数据类型
        if not pd.api.types.is_numeric_dtype(returns.dtype):
            raise DataFormatError("numeric dtype", f"{returns.dtype}")
        
        # 检查数值范围（日收益率通常不超过±50%）
        extreme_returns = returns.abs() > 0.5
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            logger.warning(f"Found {extreme_count} extreme returns (>50%)")
        
        return True
    
    def validate_weights(self, weights: pd.Series) -> bool:
        """
        验证权重数据格式
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        DataFormatError
            数据格式错误
        """
        # 检查数据类型
        if not pd.api.types.is_numeric_dtype(weights.dtype):
            raise DataFormatError("numeric dtype", f"{weights.dtype}")
        
        # 检查权重和
        weight_sum = weights.sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise DataFormatError("weights sum to 1.0", f"weights sum to {weight_sum:.6f}")
        
        # 检查极端权重
        if weights.abs().max() > 1.0:
            max_weight = weights.abs().max()
            logger.warning(f"Found extreme weight: {max_weight:.4f}")
        
        # 检查总杠杆
        total_leverage = weights.abs().sum()
        if total_leverage > 3.0:  # 允许适度杠杆
            logger.warning(f"High leverage detected: {total_leverage:.2f}")
        
        return True
    
    def align_data(self, 
                   factor_exposures: pd.DataFrame,
                   returns: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        对齐因子暴露度和收益率数据
        
        Parameters
        ----------
        factor_exposures : pd.DataFrame
            因子暴露度数据
        returns : pd.Series
            收益率数据
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            对齐后的数据
        """
        # 找到公共索引
        common_index = factor_exposures.index.intersection(returns.index)
        
        if len(common_index) == 0:
            raise InsufficientDataError(1, 0, "overlapping observations")
        
        # 对齐数据
        aligned_exposures = factor_exposures.reindex(common_index)
        aligned_returns = returns.reindex(common_index)
        
        # 处理缺失值
        if self.handle_missing == 'drop':
            # 删除任何包含缺失值的行
            mask = aligned_exposures.isnull().any(axis=1) | aligned_returns.isnull()
            aligned_exposures = aligned_exposures[~mask]
            aligned_returns = aligned_returns[~mask]
        elif self.handle_missing == 'forward_fill':
            # 前向填充
            aligned_exposures = aligned_exposures.fillna(method='ffill')
            aligned_returns = aligned_returns.fillna(method='ffill')
        
        # 检查最终数据量
        if len(aligned_exposures) < self.min_observations:
            raise InsufficientDataError(
                self.min_observations, 
                len(aligned_exposures), 
                "aligned observations"
            )
        
        logger.info(f"Aligned data: {len(aligned_exposures)} observations")
        
        return aligned_exposures, aligned_returns
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns
        -------
        Dict[str, Any]
            模型信息字典
        """
        return {
            'model_class': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'fit_timestamp': self.fit_timestamp,
            'config': self.config,
            'estimation_universe_size': len(self.estimation_universe),
            'model_params': {k: str(v) if not isinstance(v, (int, float, bool)) else v 
                           for k, v in self.model_params.items()}
        }
    
    def forecast_volatility(self, horizon: int = 20) -> pd.Series:
        """
        预测波动率
        
        Parameters
        ----------
        horizon : int
            预测期限（天数）
            
        Returns
        -------
        pd.Series
            预测的波动率，index=stocks
        """
        if not self.is_fitted:
            raise ModelNotFittedError(self.__class__.__name__)
        
        # 基础实现：基于协方差矩阵对角线
        cov_matrix = self.predict_covariance(horizon)
        volatility = np.sqrt(np.diag(cov_matrix) * horizon)
        
        return pd.Series(volatility, index=cov_matrix.index, name='volatility')
    
    def calculate_diversification_ratio(self, weights: pd.Series) -> float:
        """
        计算分散化比率
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
            
        Returns
        -------
        float
            分散化比率
        """
        if not self.is_fitted:
            raise ModelNotFittedError(self.__class__.__name__)
        
        self.validate_weights(weights)
        
        # 获取协方差矩阵
        cov_matrix = self.predict_covariance()
        
        # 个股加权平均波动率
        individual_vol = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.sum(weights.abs() * individual_vol)
        
        # 组合波动率
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # 分散化比率
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return diversification_ratio
    
    def stress_test(self, 
                    weights: pd.Series,
                    stress_scenarios: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        压力测试
        
        Parameters
        ----------
        weights : pd.Series
            组合权重
        stress_scenarios : Dict[str, np.ndarray]
            压力测试场景，每个场景是股票收益率向量
            
        Returns
        -------
        Dict[str, float]
            各场景下的组合收益率
        """
        if not self.is_fitted:
            raise ModelNotFittedError(self.__class__.__name__)
        
        self.validate_weights(weights)
        
        stress_results = {}
        
        for scenario_name, returns_vector in stress_scenarios.items():
            # 确保维度匹配
            if len(returns_vector) != len(weights):
                logger.warning(f"Dimension mismatch in scenario {scenario_name}")
                continue
            
            # 计算组合收益率
            portfolio_return = np.sum(weights * returns_vector)
            stress_results[scenario_name] = portfolio_return
        
        return stress_results
    
    def _check_fitted(self):
        """检查模型是否已拟合"""
        if not self.is_fitted:
            raise ModelNotFittedError(self.__class__.__name__)
    
    def _log_fit_completion(self, 
                            n_observations: int,
                            n_assets: int,
                            n_factors: int = None):
        """记录拟合完成信息"""
        self.fit_timestamp = datetime.now()
        self.is_fitted = True
        
        log_msg = (f"Model fitting completed: {n_observations} observations, "
                  f"{n_assets} assets")
        if n_factors:
            log_msg += f", {n_factors} factors"
        
        logger.info(log_msg)