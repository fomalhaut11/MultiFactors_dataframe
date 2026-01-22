"""
均值-方差优化器

实现经典的马科维茨均值-方差组合优化
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


class MeanVarianceOptimizer(OptimizerBase):
    """
    均值-方差优化器

    基于马科维茨现代投资组合理论的组合优化实现

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
    >>> from factors.risk_model.optimizer import MeanVarianceOptimizer
    >>>
    >>> # 创建风险模型并拟合
    >>> risk_model = StatisticalRiskModel()
    >>> risk_model.fit(returns_data)
    >>>
    >>> # 创建优化器
    >>> optimizer = MeanVarianceOptimizer(risk_model, risk_free_rate=0.02)
    >>>
    >>> # 执行优化
    >>> result = optimizer.optimize(expected_returns, constraints={'max_weight': 0.1})
    >>> print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    """

    def __init__(self,
                 risk_model,
                 risk_free_rate: float = 0.02,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化均值-方差优化器

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

        logger.info(f"Initialized MeanVarianceOptimizer with risk_free_rate={risk_free_rate}")

    def optimize(self,
                expected_returns: pd.Series,
                constraints: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行均值-方差优化

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率，index为股票代码
        constraints : Dict[str, Any], optional
            约束条件:
            - max_weight: float, 单只股票最大权重
            - min_weight: float, 单只股票最小权重
            - target_return: float, 目标收益率
            - target_volatility: float, 目标波动率
            - sector_max: Dict[str, float], 行业权重上限
            - turnover_limit: float, 换手率限制
        **kwargs : dict
            其他参数:
            - current_weights: pd.Series, 当前权重（用于换手约束）
            - optimization_target: str, 优化目标 {'sharpe', 'return', 'risk'}

        Returns
        -------
        Dict[str, Any]
            优化结果:
            - weights: pd.Series, 最优权重
            - expected_return: float, 组合预期收益
            - risk: float, 组合风险（波动率）
            - sharpe_ratio: float, 夏普比率
            - optimization_status: str, 优化状态
            - iterations: int, 迭代次数
        """
        # 验证输入
        self._check_risk_model()
        self.validate_expected_returns(expected_returns)
        self.validate_constraints(constraints or {})

        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        # 获取协方差矩阵
        cov_matrix = self.risk_model.predict_covariance()

        # 确保协方差矩阵与预期收益的资产一致
        common_assets = list(set(assets) & set(cov_matrix.index))
        if len(common_assets) < n_assets:
            logger.warning(f"Assets mismatch: {n_assets - len(common_assets)} assets not in covariance matrix")
            expected_returns = expected_returns.loc[common_assets]
            n_assets = len(common_assets)
            assets = common_assets

        cov_matrix = cov_matrix.loc[assets, assets]
        returns_array = expected_returns.values
        cov_array = cov_matrix.values

        # 确定优化目标
        optimization_target = kwargs.get('optimization_target', 'sharpe')
        current_weights = kwargs.get('current_weights', None)

        # 定义目标函数
        def neg_sharpe(weights):
            """负夏普比率（用于最大化）"""
            port_return = np.dot(weights, returns_array)
            port_risk = np.sqrt(np.dot(weights, np.dot(cov_array, weights)))
            if port_risk == 0:
                return 0
            return -(port_return - self.risk_free_rate) / port_risk

        def portfolio_variance(weights):
            """组合方差（用于最小化风险）"""
            return np.dot(weights, np.dot(cov_array, weights))

        def neg_return(weights):
            """负收益（用于最大化收益）"""
            return -np.dot(weights, returns_array)

        # 选择目标函数
        if optimization_target == 'sharpe':
            objective = neg_sharpe
        elif optimization_target == 'risk':
            objective = portfolio_variance
        elif optimization_target == 'return':
            objective = neg_return
        else:
            raise InvalidParameterError('optimization_target', optimization_target,
                                        "{'sharpe', 'risk', 'return'}")

        # 设置约束条件
        constraint_list = self.setup_constraints(
            constraints or {},
            n_assets,
            current_weights
        )

        # 设置边界
        if constraints and 'min_weight' in constraints:
            lower_bound = constraints['min_weight']
        else:
            lower_bound = 0.0

        if constraints and 'max_weight' in constraints:
            upper_bound = constraints['max_weight']
        else:
            upper_bound = 1.0

        bounds = [(lower_bound, upper_bound) for _ in range(n_assets)]

        # 目标收益约束
        if constraints and 'target_return' in constraints:
            target_ret = constraints['target_return']
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, returns_array) - target_ret
            })

        # 初始权重
        initial_weights = self.generate_initial_weights(n_assets, method='equal')

        # 执行优化
        try:
            result = minimize(
                objective,
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
                # 尝试使用不同的初始化
                logger.warning(f"First optimization attempt failed: {result.message}. Trying random initialization.")
                initial_weights = self.generate_initial_weights(n_assets, method='random')
                result = minimize(
                    objective,
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
                raise OptimizationConvergenceError(
                    'MeanVarianceOptimizer',
                    result.nit,
                    result.message
                )

            # 提取结果
            optimal_weights = result.x
            # 标准化权重确保和为1
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
                'iterations': result.nit,
                'convergence_message': result.message
            }

            # 记录优化历史
            self.track_optimization(optimization_result)

            logger.info(f"Optimization successful: Sharpe={sharpe:.4f}, Return={portfolio_return:.4f}, Risk={portfolio_risk:.4f}")

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
            有效前沿数据，columns=['risk', 'return', 'sharpe', 'weights']
        """
        self._check_risk_model()
        self.validate_expected_returns(expected_returns)

        min_risk, max_risk = risk_range
        if min_risk >= max_risk:
            raise InvalidParameterError('risk_range', risk_range, "min_risk < max_risk")

        target_risks = np.linspace(min_risk, max_risk, n_points)
        frontier_points = []

        for target_risk in target_risks:
            try:
                result = self.optimize(
                    expected_returns,
                    constraints={'target_volatility': target_risk}
                )
                frontier_points.append({
                    'risk': result['risk'],
                    'return': result['expected_return'],
                    'sharpe': result['sharpe_ratio'],
                    'weights': result['weights']
                })
            except OptimizationConvergenceError:
                logger.warning(f"Failed to find portfolio at risk level {target_risk:.4f}")
                continue

        if not frontier_points:
            raise OptimizationConvergenceError(
                'MeanVarianceOptimizer',
                message="Failed to compute any point on the efficient frontier"
            )

        frontier_df = pd.DataFrame(frontier_points)

        logger.info(f"Computed efficient frontier with {len(frontier_df)} points")

        return frontier_df

    def find_optimal_portfolio(self,
                              expected_returns: pd.Series,
                              constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        寻找最优组合（最大夏普比率）

        这是一个便捷方法，等同于optimize(optimization_target='sharpe')

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率
        constraints : Dict[str, Any], optional
            约束条件

        Returns
        -------
        Dict[str, Any]
            最优组合结果
        """
        return self.optimize(
            expected_returns,
            constraints=constraints,
            optimization_target='sharpe'
        )
