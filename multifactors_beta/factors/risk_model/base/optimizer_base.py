"""
组合优化器基类

定义组合优化器的统一接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .exceptions import (
    OptimizationConvergenceError,
    InvalidParameterError,
    DataFormatError,
    InsufficientDataError
)

logger = logging.getLogger(__name__)


class OptimizerBase(ABC):
    """
    组合优化器抽象基类
    
    定义了所有组合优化器必须实现的接口和通用功能
    """
    
    def __init__(self, risk_model, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化器
        
        Parameters
        ----------
        risk_model : RiskModelBase
            风险模型实例
        config : Dict[str, Any], optional
            优化器配置参数
        """
        self.risk_model = risk_model
        self.config = config or {}
        self.optimization_history = []
        self.last_result = None
        
        # 默认配置
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.tolerance = self.config.get('tolerance', 1e-6)
        self.method = self.config.get('method', 'SLSQP')
        self.random_seed = self.config.get('random_seed', 42)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def optimize(self,
                expected_returns: pd.Series,
                constraints: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行组合优化
        
        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率，index=stocks
        constraints : Dict[str, Any], optional
            约束条件 {
                'max_weight': float,      # 单只股票最大权重
                'min_weight': float,      # 单只股票最小权重  
                'sector_max': Dict,       # 行业权重上限
                'turnover_limit': float,  # 换手率限制
                'target_volatility': float # 目标波动率
            }
        **kwargs : dict
            其他优化参数
            
        Returns
        -------
        Dict[str, Any]
            优化结果 {
                'weights': pd.Series,           # 最优权重
                'expected_return': float,       # 预期收益
                'risk': float,                  # 组合风险
                'sharpe_ratio': float,          # 夏普比率
                'optimization_status': str,     # 优化状态
                'iterations': int               # 迭代次数
            }
        """
        pass
    
    @abstractmethod
    def calculate_efficient_frontier(self,
                                   expected_returns: pd.Series,
                                   risk_range: Tuple[float, float],
                                   n_points: int = 20) -> pd.DataFrame:
        """
        计算有效前沿
        
        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率
        risk_range : Tuple[float, float]
            风险范围 (min_risk, max_risk)
        n_points : int
            前沿点数
            
        Returns
        -------
        pd.DataFrame
            有效前沿，columns=['risk', 'return', 'sharpe']
        """
        pass
    
    def validate_expected_returns(self, expected_returns: pd.Series) -> bool:
        """
        验证预期收益率数据
        
        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率数据
            
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
        if not pd.api.types.is_numeric_dtype(expected_returns.dtype):
            raise DataFormatError("numeric dtype", f"{expected_returns.dtype}")
        
        # 检查空值
        if expected_returns.isnull().any():
            null_count = expected_returns.isnull().sum()
            raise DataFormatError("no missing values", f"{null_count} missing values found")
        
        # 检查极端值（年化收益率通常在-100%到100%之间）
        extreme_returns = expected_returns.abs() > 1.0
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            logger.warning(f"Found {extreme_count} extreme expected returns (>100%)")
        
        # 检查数据量
        if len(expected_returns) < 2:
            raise InsufficientDataError(2, len(expected_returns), "assets for optimization")
        
        return True
    
    def validate_constraints(self, constraints: Dict[str, Any]) -> bool:
        """
        验证约束条件
        
        Parameters
        ----------
        constraints : Dict[str, Any]
            约束条件
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        InvalidParameterError
            参数无效
        """
        if constraints is None:
            return True
        
        # 检查权重约束
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            if not 0 < max_weight <= 1.0:
                raise InvalidParameterError('max_weight', max_weight, "(0, 1]")
        
        if 'min_weight' in constraints:
            min_weight = constraints['min_weight']
            if not -1.0 <= min_weight < 1.0:
                raise InvalidParameterError('min_weight', min_weight, "[-1, 1)")
        
        # 检查权重约束一致性
        if 'max_weight' in constraints and 'min_weight' in constraints:
            if constraints['min_weight'] >= constraints['max_weight']:
                raise InvalidParameterError(
                    'weight constraints', 
                    f"min_weight({constraints['min_weight']}) >= max_weight({constraints['max_weight']})",
                    "min_weight < max_weight"
                )
        
        # 检查目标波动率
        if 'target_volatility' in constraints:
            target_vol = constraints['target_volatility']
            if not 0 < target_vol <= 2.0:  # 年化波动率通常不超过200%
                raise InvalidParameterError('target_volatility', target_vol, "(0, 2]")
        
        # 检查换手率限制
        if 'turnover_limit' in constraints:
            turnover = constraints['turnover_limit']
            if not 0 <= turnover <= 2.0:
                raise InvalidParameterError('turnover_limit', turnover, "[0, 2]")
        
        return True
    
    def setup_constraints(self, 
                         constraints: Dict[str, Any],
                         n_assets: int,
                         current_weights: Optional[pd.Series] = None) -> List[Dict]:
        """
        设置优化约束条件
        
        Parameters
        ----------
        constraints : Dict[str, Any]
            约束条件字典
        n_assets : int
            资产数量
        current_weights : pd.Series, optional
            当前权重（用于换手率约束）
            
        Returns
        -------
        List[Dict]
            scipy.optimize格式的约束条件列表
        """
        constraint_list = []
        
        # 权重和等于1的约束
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        if constraints is None:
            return constraint_list
        
        # 个股权重约束
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: max_weight - w[idx]
                })
        
        if 'min_weight' in constraints:
            min_weight = constraints['min_weight']
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: w[idx] - min_weight
                })
        
        # 目标波动率约束
        if 'target_volatility' in constraints:
            target_vol = constraints['target_volatility']
            
            def vol_constraint(w):
                portfolio_vol = self.calculate_portfolio_risk(w)
                return abs(portfolio_vol - target_vol)
            
            constraint_list.append({
                'type': 'eq',
                'fun': vol_constraint
            })
        
        # 换手率约束
        if 'turnover_limit' in constraints and current_weights is not None:
            turnover_limit = constraints['turnover_limit']
            
            def turnover_constraint(w):
                turnover = np.sum(np.abs(w - current_weights.values))
                return turnover_limit - turnover
            
            constraint_list.append({
                'type': 'ineq',
                'fun': turnover_constraint
            })
        
        return constraint_list
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """
        计算组合风险（波动率）
        
        Parameters
        ----------
        weights : np.ndarray
            组合权重
            
        Returns
        -------
        float
            组合波动率
        """
        # 获取协方差矩阵
        cov_matrix = self.risk_model.predict_covariance()
        
        # 计算组合方差
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # 返回波动率
        return np.sqrt(portfolio_variance)
    
    def calculate_portfolio_return(self, 
                                 weights: np.ndarray,
                                 expected_returns: np.ndarray) -> float:
        """
        计算组合预期收益
        
        Parameters
        ----------
        weights : np.ndarray
            组合权重
        expected_returns : np.ndarray
            预期收益率
            
        Returns
        -------
        float
            组合预期收益
        """
        return np.dot(weights, expected_returns)
    
    def calculate_sharpe_ratio(self,
                              weights: np.ndarray,
                              expected_returns: np.ndarray,
                              risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Parameters
        ----------
        weights : np.ndarray
            组合权重
        expected_returns : np.ndarray
            预期收益率
        risk_free_rate : float
            无风险收益率
            
        Returns
        -------
        float
            夏普比率
        """
        portfolio_return = self.calculate_portfolio_return(weights, expected_returns)
        portfolio_risk = self.calculate_portfolio_risk(weights)
        
        if portfolio_risk == 0:
            return 0.0
        
        return (portfolio_return - risk_free_rate) / portfolio_risk
    
    def generate_initial_weights(self, n_assets: int, method: str = 'equal') -> np.ndarray:
        """
        生成初始权重
        
        Parameters
        ----------
        n_assets : int
            资产数量
        method : str
            初始化方法 {'equal', 'random', 'min_var'}
            
        Returns
        -------
        np.ndarray
            初始权重
        """
        np.random.seed(self.random_seed)
        
        if method == 'equal':
            # 等权重
            return np.ones(n_assets) / n_assets
        
        elif method == 'random':
            # 随机权重
            weights = np.random.rand(n_assets)
            return weights / weights.sum()
        
        elif method == 'min_var':
            # 最小方差权重（近似）
            try:
                cov_matrix = self.risk_model.predict_covariance()
                inv_cov = np.linalg.inv(cov_matrix.values)
                ones = np.ones(n_assets)
                weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
                return weights
            except:
                # 如果矩阵不可逆，返回等权重
                logger.warning("Failed to compute min variance weights, using equal weights")
                return np.ones(n_assets) / n_assets
        
        else:
            raise InvalidParameterError('method', method, "{'equal', 'random', 'min_var'}")
    
    def track_optimization(self, result: Dict[str, Any]):
        """
        跟踪优化结果
        
        Parameters
        ----------
        result : Dict[str, Any]
            优化结果
        """
        self.last_result = result.copy()
        self.last_result['timestamp'] = datetime.now()
        self.optimization_history.append(self.last_result)
        
        # 保留最近100次优化记录
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        获取优化历史
        
        Returns
        -------
        List[Dict[str, Any]]
            优化历史记录
        """
        return self.optimization_history.copy()
    
    def analyze_optimization_performance(self) -> Dict[str, Any]:
        """
        分析优化性能
        
        Returns
        -------
        Dict[str, Any]
            性能分析结果
        """
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        history = pd.DataFrame(self.optimization_history)
        
        performance_metrics = {
            'total_optimizations': len(history),
            'success_rate': (history['optimization_status'] == 'success').mean(),
            'avg_iterations': history['iterations'].mean(),
            'avg_sharpe_ratio': history['sharpe_ratio'].mean(),
            'avg_risk': history['risk'].mean(),
            'avg_return': history['expected_return'].mean(),
            'convergence_issues': (history['optimization_status'] != 'success').sum()
        }
        
        return performance_metrics
    
    def _check_risk_model(self):
        """检查风险模型是否已拟合"""
        if not hasattr(self.risk_model, 'is_fitted') or not self.risk_model.is_fitted:
            raise OptimizationConvergenceError(
                self.__class__.__name__,
                message="Risk model has not been fitted yet"
            )