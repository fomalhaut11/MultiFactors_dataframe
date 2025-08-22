"""
优化权重计算器

基于最优化理论的权重分配方法
"""

from typing import Dict, Optional, Any, Union
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from .base_weight import BaseWeightCalculator

logger = logging.getLogger(__name__)


class OptimalWeightCalculator(BaseWeightCalculator):
    """
    优化权重计算器
    
    使用最优化方法计算因子权重，包括：
    - 最大夏普比率
    - 最小方差
    - 均值-方差优化
    - 风险调整收益最大化
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化权重计算器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        self.objective = self.config.get('objective', 'max_sharpe')  # 优化目标
        self.lookback = self.config.get('lookback', 252)  # 回看期
        self.risk_free_rate = self.config.get('risk_free_rate', 0.03)  # 无风险利率
        self.target_return = self.config.get('target_return', None)  # 目标收益率
        self.risk_aversion = self.config.get('risk_aversion', 1.0)  # 风险厌恶系数
        self.max_iter = self.config.get('max_iter', 1000)  # 最大迭代次数
        self.tol = self.config.get('tol', 1e-8)  # 收敛容差
        self.allow_short = self.config.get('allow_short', False)  # 是否允许做空
    
    def calculate(self,
                 factors: Dict[str, pd.Series],
                 evaluation_results: Optional[Dict] = None,
                 **kwargs) -> Dict[str, float]:
        """
        计算优化权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数，如returns用于计算收益率
            
        Returns
        -------
        Dict[str, float]
            优化权重
        """
        self.validate_inputs(factors, evaluation_results)
        
        # 获取因子收益率
        factor_returns = self._get_factor_returns(
            factors, evaluation_results, **kwargs
        )
        
        if factor_returns is None or factor_returns.empty:
            logger.warning("Cannot calculate returns, using equal weights")
            n = len(factors)
            return {name: 1.0/n for name in factors.keys()}
        
        # 计算统计量
        mean_returns = factor_returns.mean()
        cov_matrix = factor_returns.cov()
        
        # 确保协方差矩阵正定
        cov_matrix = self._ensure_positive_definite(cov_matrix)
        
        # 根据目标函数优化
        if self.objective == 'max_sharpe':
            weights = self._max_sharpe(mean_returns, cov_matrix)
        elif self.objective == 'min_variance':
            weights = self._min_variance(cov_matrix)
        elif self.objective == 'mean_variance':
            weights = self._mean_variance(mean_returns, cov_matrix)
        elif self.objective == 'max_return':
            weights = self._max_return(mean_returns, cov_matrix)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        # 转换为字典格式
        weight_dict = {
            name: weights[i] 
            for i, name in enumerate(factors.keys())
        }
        
        # 应用约束
        weight_dict = self.apply_constraints(weight_dict)
        
        logger.info(f"Calculated optimal weights using {self.objective}")
        return weight_dict
    
    def _get_factor_returns(self,
                          factors: Dict[str, pd.Series],
                          evaluation_results: Optional[Dict] = None,
                          **kwargs) -> pd.DataFrame:
        """
        获取因子收益率
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        pd.DataFrame
            因子收益率矩阵
        """
        # 优先使用提供的收益率数据
        if 'returns' in kwargs and kwargs['returns'] is not None:
            returns = kwargs['returns']
            if isinstance(returns, pd.DataFrame):
                return returns
        
        # 从评估结果中提取收益率
        if evaluation_results:
            returns_dict = {}
            for name, result in evaluation_results.items():
                if isinstance(result, dict):
                    if 'factor_returns' in result:
                        returns_dict[name] = result['factor_returns']
                    elif 'metrics' in result and 'factor_returns' in result['metrics']:
                        returns_dict[name] = result['metrics']['factor_returns']
            
            if returns_dict:
                # 对齐并合并
                return pd.DataFrame(returns_dict)
        
        # 计算因子收益率（简化版本）
        return self._calculate_factor_returns(factors)
    
    def _calculate_factor_returns(self, factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        计算因子收益率（简化版本）
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
            
        Returns
        -------
        pd.DataFrame
            因子收益率矩阵
        """
        try:
            # 将因子转换为DataFrame
            factor_df = pd.DataFrame(factors)
            
            # 按日期分组计算收益率
            dates = factor_df.index.get_level_values(0).unique()
            
            # 计算因子的日收益率
            daily_returns = []
            
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                
                # 获取两个时期的因子值
                prev_values = factor_df.xs(prev_date, level=0)
                curr_values = factor_df.xs(curr_date, level=0)
                
                # 计算横截面平均收益率
                common_stocks = prev_values.index.intersection(curr_values.index)
                if len(common_stocks) > 0:
                    prev_mean = prev_values.loc[common_stocks].mean()
                    curr_mean = curr_values.loc[common_stocks].mean()
                    
                    # 计算收益率
                    returns = {}
                    for col in prev_mean.index:
                        if abs(prev_mean[col]) > 1e-10:
                            returns[col] = (curr_mean[col] - prev_mean[col]) / abs(prev_mean[col])
                        else:
                            returns[col] = 0
                    
                    daily_returns.append(returns)
            
            if daily_returns:
                return_df = pd.DataFrame(daily_returns)
                
                # 只使用最近的数据
                if self.lookback > 0:
                    return_df = return_df.tail(self.lookback)
                
                return return_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Failed to calculate factor returns: {e}")
            return pd.DataFrame()
    
    def _ensure_positive_definite(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        确保协方差矩阵正定
        
        Parameters
        ----------
        cov_matrix : pd.DataFrame
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            正定的协方差矩阵
        """
        cov_array = cov_matrix.values
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_array)
        
        # 修正负特征值
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        # 重构协方差矩阵
        cov_array = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return cov_array
    
    def _max_sharpe(self, mean_returns: pd.Series, cov_matrix: np.ndarray) -> np.ndarray:
        """
        最大化夏普比率
        
        Parameters
        ----------
        mean_returns : pd.Series
            平均收益率
        cov_matrix : np.ndarray
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            权重向量
        """
        n_assets = len(mean_returns)
        
        # 目标函数：负夏普比率（最小化）
        def objective(weights):
            portfolio_return = weights @ mean_returns.values
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            sharpe = (portfolio_return - self.risk_free_rate) / (portfolio_vol + 1e-10)
            return -sharpe  # 负值以进行最小化
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
        ]
        
        # 边界条件
        if self.allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("Sharpe optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _min_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        最小化方差
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            权重向量
        """
        n_assets = len(cov_matrix)
        
        # 目标函数：组合方差
        def objective(weights):
            return weights @ cov_matrix @ weights
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
        ]
        
        # 边界条件
        if self.allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("Min variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _mean_variance(self, mean_returns: pd.Series, cov_matrix: np.ndarray) -> np.ndarray:
        """
        均值-方差优化
        
        Parameters
        ----------
        mean_returns : pd.Series
            平均收益率
        cov_matrix : np.ndarray
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            权重向量
        """
        n_assets = len(mean_returns)
        
        # 目标函数：风险调整收益（均值-方差）
        def objective(weights):
            portfolio_return = weights @ mean_returns.values
            portfolio_risk = weights @ cov_matrix @ weights
            # 均值-方差目标函数
            return -(portfolio_return - self.risk_aversion * portfolio_risk)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
        ]
        
        # 如果设置了目标收益率，添加约束
        if self.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: x @ mean_returns.values - self.target_return
            })
        
        # 边界条件
        if self.allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("Mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _max_return(self, mean_returns: pd.Series, cov_matrix: np.ndarray) -> np.ndarray:
        """
        最大化收益率（带风险约束）
        
        Parameters
        ----------
        mean_returns : pd.Series
            平均收益率
        cov_matrix : np.ndarray
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            权重向量
        """
        n_assets = len(mean_returns)
        
        # 目标函数：负收益率（最小化）
        def objective(weights):
            return -(weights @ mean_returns.values)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
        ]
        
        # 添加风险约束（如果配置了最大风险）
        max_risk = self.config.get('max_risk', None)
        if max_risk is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_risk**2 - x @ cov_matrix @ x
            })
        
        # 边界条件
        if self.allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("Max return optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def calculate_efficient_frontier(self,
                                    mean_returns: pd.Series,
                                    cov_matrix: np.ndarray,
                                    n_portfolios: int = 100) -> Dict[str, np.ndarray]:
        """
        计算有效前沿
        
        Parameters
        ----------
        mean_returns : pd.Series
            平均收益率
        cov_matrix : np.ndarray
            协方差矩阵
        n_portfolios : int
            前沿上的组合数量
            
        Returns
        -------
        Dict[str, np.ndarray]
            包含收益率、风险和权重的字典
        """
        # 获取最小方差组合
        min_var_weights = self._min_variance(cov_matrix)
        min_var_return = min_var_weights @ mean_returns.values
        
        # 获取最大收益组合
        max_return = mean_returns.max()
        
        # 生成目标收益率序列
        target_returns = np.linspace(min_var_return, max_return, n_portfolios)
        
        # 计算每个目标收益率对应的最优组合
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        
        for target in target_returns:
            # 设置目标收益率
            self.target_return = target
            
            # 优化
            weights = self._mean_variance(mean_returns, cov_matrix)
            
            # 计算组合统计量
            portfolio_return = weights @ mean_returns.values
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            
            frontier_returns.append(portfolio_return)
            frontier_risks.append(portfolio_risk)
            frontier_weights.append(weights)
        
        return {
            'returns': np.array(frontier_returns),
            'risks': np.array(frontier_risks),
            'weights': np.array(frontier_weights)
        }