"""
最大夏普比率优化器

实现最大夏普比率组合优化
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


class MaxSharpeOptimizer(OptimizerBase):
    """
    最大夏普比率优化器

    寻找风险调整后收益最大的组合

    Parameters
    ----------
    risk_model : RiskModelBase
        风险模型实例
    risk_free_rate : float, optional
        无风险收益率，默认0.02（年化2%）
    config : Dict[str, Any], optional
        优化器配置参数

    Examples
    --------
    >>> from factors.risk_model.models import StatisticalRiskModel
    >>> from factors.risk_model.optimizer import MaxSharpeOptimizer
    >>>
    >>> risk_model = StatisticalRiskModel()
    >>> risk_model.fit(returns_data)
    >>>
    >>> optimizer = MaxSharpeOptimizer(risk_model, risk_free_rate=0.02)
    >>> result = optimizer.optimize(expected_returns, constraints={'max_weight': 0.1})
    >>> print(f"Max Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    """

    def __init__(self,
                 risk_model,
                 risk_free_rate: float = 0.02,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化最大夏普比率优化器

        Parameters
        ----------
        risk_model : RiskModelBase
            风险模型实例
        risk_free_rate : float
            无风险收益率
        config : Dict[str, Any], optional
            优化器配置
        """
        super().__init__(risk_model, config)
        self.risk_free_rate = risk_free_rate

        logger.info(f"Initialized MaxSharpeOptimizer with risk_free_rate={risk_free_rate}")

    def optimize(self,
                expected_returns: pd.Series,
                constraints: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行最大夏普比率优化

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率
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

        # 调整后的收益（超额收益）
        excess_returns = returns_array - self.risk_free_rate

        # 目标函数：负夏普比率（最小化）
        def neg_sharpe(weights):
            port_return = np.dot(weights, returns_array)
            port_risk = np.sqrt(np.dot(weights, np.dot(cov_array, weights)))
            if port_risk < 1e-10:
                return 1e10  # 返回大值避免数值问题
            return -(port_return - self.risk_free_rate) / port_risk

        # 设置约束
        constraint_list = self.setup_constraints(constraints or {}, n_assets)

        # 设置边界
        lower_bound = constraints.get('min_weight', 0.0) if constraints else 0.0
        upper_bound = constraints.get('max_weight', 1.0) if constraints else 1.0
        bounds = [(lower_bound, upper_bound) for _ in range(n_assets)]

        # 多次尝试不同初始化
        best_result = None
        best_sharpe = -np.inf

        init_methods = ['equal', 'random', 'min_var']

        for init_method in init_methods:
            try:
                initial_weights = self.generate_initial_weights(n_assets, method=init_method)

                result = minimize(
                    neg_sharpe,
                    initial_weights,
                    method=self.method,
                    bounds=bounds,
                    constraints=constraint_list,
                    options={
                        'maxiter': self.max_iterations,
                        'ftol': self.tolerance
                    }
                )

                if result.success:
                    current_sharpe = -result.fun
                    if current_sharpe > best_sharpe:
                        best_sharpe = current_sharpe
                        best_result = result

            except Exception as e:
                logger.warning(f"Optimization with {init_method} init failed: {str(e)}")
                continue

        if best_result is None:
            raise OptimizationConvergenceError(
                'MaxSharpeOptimizer',
                message="All optimization attempts failed"
            )

        # 提取结果
        optimal_weights = best_result.x
        optimal_weights = optimal_weights / optimal_weights.sum()

        # 计算组合指标
        portfolio_return = np.dot(optimal_weights, returns_array)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_array, optimal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # 构建结果
        optimization_result = {
            'weights': pd.Series(optimal_weights, index=assets),
            'expected_return': float(portfolio_return),
            'risk': float(portfolio_risk),
            'sharpe_ratio': float(sharpe),
            'optimization_status': 'success',
            'iterations': best_result.nit,
            'convergence_message': best_result.message
        }

        self.track_optimization(optimization_result)

        logger.info(f"MaxSharpe optimization successful: Sharpe={sharpe:.4f}")

        return optimization_result

    def calculate_efficient_frontier(self,
                                   expected_returns: pd.Series,
                                   risk_range: Tuple[float, float],
                                   n_points: int = 20) -> pd.DataFrame:
        """
        计算有效前沿

        从最小方差点到最大收益点

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率
        risk_range : Tuple[float, float]
            风险范围
        n_points : int
            前沿点数

        Returns
        -------
        pd.DataFrame
            有效前沿数据
        """
        self._check_risk_model()
        self.validate_expected_returns(expected_returns)

        min_risk, max_risk = risk_range
        if min_risk >= max_risk:
            raise InvalidParameterError('risk_range', risk_range, "min_risk < max_risk")

        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        cov_matrix = self.risk_model.predict_covariance().loc[assets, assets]
        returns_array = expected_returns.values
        cov_array = cov_matrix.values

        target_returns = np.linspace(
            expected_returns.min(),
            expected_returns.max(),
            n_points
        )

        frontier_points = []

        for target_ret in target_returns:
            # 目标函数：最小化方差
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov_array, weights))

            # 约束：权重和为1，目标收益
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, returns_array) - target_ret}
            ]

            bounds = [(0, 1) for _ in range(n_assets)]
            initial_weights = np.ones(n_assets) / n_assets

            try:
                result = minimize(
                    portfolio_variance,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )

                if result.success:
                    weights = result.x / result.x.sum()
                    port_risk = np.sqrt(np.dot(weights, np.dot(cov_array, weights)))
                    port_return = np.dot(weights, returns_array)
                    sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0

                    if min_risk <= port_risk <= max_risk:
                        frontier_points.append({
                            'risk': port_risk,
                            'return': port_return,
                            'sharpe': sharpe,
                            'weights': pd.Series(weights, index=assets)
                        })
            except:
                continue

        if not frontier_points:
            raise OptimizationConvergenceError(
                'MaxSharpeOptimizer',
                message="Failed to compute efficient frontier"
            )

        frontier_df = pd.DataFrame(frontier_points)
        frontier_df = frontier_df.sort_values('risk').drop_duplicates(subset=['risk'])

        logger.info(f"Computed efficient frontier with {len(frontier_df)} points")

        return frontier_df
