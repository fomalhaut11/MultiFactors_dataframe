"""
最小方差优化器

实现全局最小方差组合优化
"""

from typing import Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from ..base.optimizer_base import OptimizerBase
from ..base.exceptions import (
    OptimizationConvergenceError,
    InvalidParameterError
)

logger = logging.getLogger(__name__)


class MinVarianceOptimizer(OptimizerBase):
    """
    最小方差优化器

    寻找全局最小方差组合，不考虑预期收益

    Parameters
    ----------
    risk_model : RiskModelBase
        风险模型实例
    config : Dict[str, Any], optional
        优化器配置参数

    Examples
    --------
    >>> from factors.risk_model.models import StatisticalRiskModel
    >>> from factors.risk_model.optimizer import MinVarianceOptimizer
    >>>
    >>> risk_model = StatisticalRiskModel()
    >>> risk_model.fit(returns_data)
    >>>
    >>> optimizer = MinVarianceOptimizer(risk_model)
    >>> result = optimizer.optimize(expected_returns, constraints={'max_weight': 0.1})
    >>> print(f"Portfolio Risk: {result['risk']:.4f}")
    """

    def __init__(self,
                 risk_model,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化最小方差优化器

        Parameters
        ----------
        risk_model : RiskModelBase
            风险模型实例
        config : Dict[str, Any], optional
            优化器配置
        """
        super().__init__(risk_model, config)
        logger.info("Initialized MinVarianceOptimizer")

    def optimize(self,
                expected_returns: pd.Series,
                constraints: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行最小方差优化

        注意：expected_returns仅用于确定资产范围，优化目标是最小化组合方差

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率（用于确定资产范围和计算组合收益）
        constraints : Dict[str, Any], optional
            约束条件:
            - max_weight: float, 单只股票最大权重
            - min_weight: float, 单只股票最小权重
            - sector_max: Dict[str, float], 行业权重上限
        **kwargs : dict
            其他参数

        Returns
        -------
        Dict[str, Any]
            优化结果
        """
        # 验证输入
        self._check_risk_model()
        self.validate_expected_returns(expected_returns)
        self.validate_constraints(constraints or {})

        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        # 获取协方差矩阵
        cov_matrix = self.risk_model.predict_covariance()

        # 资产对齐
        common_assets = list(set(assets) & set(cov_matrix.index))
        if len(common_assets) < n_assets:
            logger.warning(f"Assets mismatch: {n_assets - len(common_assets)} assets not in covariance matrix")
            expected_returns = expected_returns.loc[common_assets]
            n_assets = len(common_assets)
            assets = common_assets

        cov_matrix = cov_matrix.loc[assets, assets]
        returns_array = expected_returns.values
        cov_array = cov_matrix.values

        # 目标函数：最小化方差
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_array, weights))

        # 设置约束
        constraint_list = self.setup_constraints(constraints or {}, n_assets)

        # 设置边界
        lower_bound = constraints.get('min_weight', 0.0) if constraints else 0.0
        upper_bound = constraints.get('max_weight', 1.0) if constraints else 1.0
        bounds = [(lower_bound, upper_bound) for _ in range(n_assets)]

        # 初始权重
        initial_weights = self.generate_initial_weights(n_assets, method='equal')

        # 执行优化
        try:
            result = minimize(
                portfolio_variance,
                initial_weights,
                method=self.method,
                bounds=bounds,
                constraints=constraint_list,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance
                }
            )

            if not result.success:
                # 尝试分析求解（对于无约束情况）
                try:
                    inv_cov = np.linalg.inv(cov_array)
                    ones = np.ones(n_assets)
                    optimal_weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
                    # 应用权重边界
                    optimal_weights = np.clip(optimal_weights, lower_bound, upper_bound)
                    optimal_weights = optimal_weights / optimal_weights.sum()
                    logger.info("Used analytical solution for minimum variance")
                except np.linalg.LinAlgError:
                    raise OptimizationConvergenceError(
                        'MinVarianceOptimizer',
                        result.nit,
                        result.message
                    )
            else:
                optimal_weights = result.x
                optimal_weights = optimal_weights / optimal_weights.sum()

            # 计算组合指标
            portfolio_return = np.dot(optimal_weights, returns_array)
            portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_array, optimal_weights)))

            # 构建结果
            optimization_result = {
                'weights': pd.Series(optimal_weights, index=assets),
                'expected_return': float(portfolio_return),
                'risk': float(portfolio_risk),
                'sharpe_ratio': float(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0,
                'optimization_status': 'success',
                'iterations': result.nit if result.success else 0,
                'convergence_message': result.message if result.success else 'analytical_solution'
            }

            self.track_optimization(optimization_result)

            logger.info(f"MinVariance optimization successful: Risk={portfolio_risk:.4f}")

            return optimization_result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    def calculate_efficient_frontier(self,
                                   expected_returns: pd.Series,
                                   risk_range: Tuple[float, float],
                                   n_points: int = 20) -> pd.DataFrame:
        """
        计算有效前沿

        对于最小方差优化器，仅返回最小方差点

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率
        risk_range : Tuple[float, float]
            风险范围（本优化器忽略此参数）
        n_points : int
            前沿点数（本优化器忽略此参数）

        Returns
        -------
        pd.DataFrame
            仅包含最小方差点
        """
        result = self.optimize(expected_returns)

        frontier_df = pd.DataFrame([{
            'risk': result['risk'],
            'return': result['expected_return'],
            'sharpe': result['sharpe_ratio'],
            'weights': result['weights']
        }])

        return frontier_df
