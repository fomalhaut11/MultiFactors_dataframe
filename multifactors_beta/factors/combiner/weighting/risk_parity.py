"""
风险平价权重计算器

基于风险贡献均衡的权重分配方法
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from .base_weight import BaseWeightCalculator

logger = logging.getLogger(__name__)


class RiskParityCalculator(BaseWeightCalculator):
    """
    风险平价权重计算器
    
    使每个因子对组合风险的贡献相等
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险平价计算器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        self.lookback = self.config.get('lookback', 252)  # 协方差计算回看期
        self.method = self.config.get('method', 'naive')  # 'naive' or 'optimized'
        self.risk_budget = self.config.get('risk_budget', None)  # 风险预算
        self.max_iter = self.config.get('max_iter', 1000)  # 最大迭代次数
        self.tol = self.config.get('tol', 1e-8)  # 收敛容差
    
    def calculate(self,
                 factors: Dict[str, pd.Series],
                 evaluation_results: Optional[Dict] = None,
                 **kwargs) -> Dict[str, float]:
        """
        计算风险平价权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果（可选）
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            风险平价权重
        """
        self.validate_inputs(factors, evaluation_results)
        
        # 计算因子收益率
        factor_returns = self._calculate_factor_returns(factors)
        
        if factor_returns is None or factor_returns.empty:
            logger.warning("Cannot calculate returns, using equal weights")
            n = len(factors)
            return {name: 1.0/n for name in factors.keys()}
        
        # 计算协方差矩阵
        cov_matrix = self._calculate_covariance(factor_returns)
        
        # 计算风险平价权重
        if self.method == 'naive':
            weights = self._naive_risk_parity(cov_matrix)
        else:
            weights = self._optimized_risk_parity(cov_matrix)
        
        # 转换为字典格式
        weight_dict = {
            name: weights[i] 
            for i, name in enumerate(factors.keys())
        }
        
        # 应用约束
        weight_dict = self.apply_constraints(weight_dict)
        
        logger.info(f"Calculated risk parity weights for {len(factors)} factors")
        return weight_dict
    
    def _calculate_factor_returns(self, factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        计算因子收益率
        
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
            
            # 计算因子的日收益率（横截面收益）
            daily_returns = []
            
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                
                # 获取两个时期的因子值
                prev_values = factor_df.xs(prev_date, level=0)
                curr_values = factor_df.xs(curr_date, level=0)
                
                # 计算收益率（这里使用因子值的变化率作为代理）
                # 实际应用中可能需要更复杂的收益率计算
                common_stocks = prev_values.index.intersection(curr_values.index)
                if len(common_stocks) > 0:
                    prev_mean = prev_values.loc[common_stocks].mean()
                    curr_mean = curr_values.loc[common_stocks].mean()
                    
                    # 避免除零
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
    
    def _calculate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        计算协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率矩阵
            
        Returns
        -------
        np.ndarray
            协方差矩阵
        """
        # 填充缺失值
        returns = returns.fillna(0)
        
        # 计算协方差
        cov_matrix = returns.cov().values
        
        # 确保正定性（添加小的正则化项）
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvalues) < 1e-8:
            cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        
        return cov_matrix
    
    def _naive_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        简单风险平价（反向波动率加权）
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            权重向量
        """
        # 计算各因子的波动率
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # 反向波动率加权
        inv_vols = 1.0 / (volatilities + 1e-10)
        weights = inv_vols / inv_vols.sum()
        
        return weights
    
    def _optimized_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        优化风险平价
        
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
        
        # 风险预算（默认均等）
        if self.risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets
        else:
            risk_budget = np.array(list(self.risk_budget.values()))
            risk_budget = risk_budget / risk_budget.sum()
        
        # 目标函数：最小化风险贡献与目标的偏差
        def objective(weights):
            # 计算组合风险
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # 计算边际风险贡献
            marginal_contrib = cov_matrix @ weights
            
            # 计算风险贡献
            risk_contrib = weights * marginal_contrib / (portfolio_vol + 1e-10)
            
            # 计算与目标风险预算的偏差
            error = np.sum((risk_contrib - risk_budget * portfolio_vol) ** 2)
            
            return error
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # 初始权重（等权）
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
            weights = result.x
        else:
            logger.warning("Risk parity optimization failed, using naive method")
            weights = self._naive_risk_parity(cov_matrix)
        
        return weights
    
    def calculate_risk_contributions(self,
                                    weights: Dict[str, float],
                                    cov_matrix: np.ndarray) -> Dict[str, float]:
        """
        计算各因子的风险贡献
        
        Parameters
        ----------
        weights : Dict[str, float]
            权重字典
        cov_matrix : np.ndarray
            协方差矩阵
            
        Returns
        -------
        Dict[str, float]
            风险贡献字典
        """
        # 转换为数组
        w = np.array(list(weights.values()))
        
        # 计算组合风险
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)
        
        # 计算边际风险贡献
        marginal_contrib = cov_matrix @ w
        
        # 计算风险贡献
        risk_contrib = w * marginal_contrib / (portfolio_vol + 1e-10)
        
        # 转换为字典
        contrib_dict = {
            name: risk_contrib[i]
            for i, name in enumerate(weights.keys())
        }
        
        return contrib_dict