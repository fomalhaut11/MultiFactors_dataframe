"""
风险平价优化器

实现风险平价（Risk Parity）组合优化策略
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


class RiskParityOptimizer(OptimizerBase):
    """
    风险平价优化器

    使每个资产对组合风险的贡献相等

    风险平价策略不依赖于预期收益的预测，
    而是基于风险分散的原则进行资产配置

    Parameters
    ----------
    risk_model : RiskModelBase
        风险模型实例
    budget : np.ndarray, optional
        风险预算，默认为等权风险贡献
    config : Dict[str, Any], optional
        优化器配置参数

    Examples
    --------
    >>> from factors.risk_model.models import StatisticalRiskModel
    >>> from factors.risk_model.optimizer import RiskParityOptimizer
    >>>
    >>> risk_model = StatisticalRiskModel()
    >>> risk_model.fit(returns_data)
    >>>
    >>> optimizer = RiskParityOptimizer(risk_model)
    >>> result = optimizer.optimize(expected_returns)
    >>> print(f"Risk contributions are equal: {result['risk_contributions']}")
    """

    def __init__(self,
                 risk_model,
                 budget: Optional[np.ndarray] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化风险平价优化器

        Parameters
        ----------
        risk_model : RiskModelBase
            风险模型实例
        budget : np.ndarray, optional
            风险预算比例，默认为等权（每个资产贡献相等的风险）
        config : Dict[str, Any], optional
            优化器配置
        """
        super().__init__(risk_model, config)
        self.budget = budget

        logger.info("Initialized RiskParityOptimizer")

    def _calculate_risk_contributions(self,
                                     weights: np.ndarray,
                                     cov_matrix: np.ndarray) -> np.ndarray:
        """
        计算每个资产的风险贡献

        Parameters
        ----------
        weights : np.ndarray
            组合权重
        cov_matrix : np.ndarray
            协方差矩阵

        Returns
        -------
        np.ndarray
            各资产的风险贡献
        """
        # 组合方差
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

        if portfolio_variance <= 0:
            return np.zeros_like(weights)

        # 组合波动率
        portfolio_vol = np.sqrt(portfolio_variance)

        # 边际风险贡献 (Marginal Risk Contribution)
        # MRC_i = (Σw)_i / σ_p
        marginal_contribution = np.dot(cov_matrix, weights) / portfolio_vol

        # 风险贡献 (Risk Contribution)
        # RC_i = w_i * MRC_i
        risk_contribution = weights * marginal_contribution

        return risk_contribution

    def _risk_parity_objective(self,
                              weights: np.ndarray,
                              cov_matrix: np.ndarray,
                              target_budget: np.ndarray) -> float:
        """
        风险平价目标函数

        最小化实际风险贡献与目标风险预算之间的差异

        Parameters
        ----------
        weights : np.ndarray
            组合权重
        cov_matrix : np.ndarray
            协方差矩阵
        target_budget : np.ndarray
            目标风险预算

        Returns
        -------
        float
            目标函数值
        """
        # 计算风险贡献
        risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)

        # 组合总风险
        total_risk = np.sum(risk_contributions)

        if total_risk <= 0:
            return 1e10

        # 风险贡献比例
        risk_contribution_pct = risk_contributions / total_risk

        # 目标函数：风险贡献比例与目标预算的平方差之和
        objective = np.sum((risk_contribution_pct - target_budget) ** 2)

        return objective

    def optimize(self,
                expected_returns: pd.Series,
                constraints: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行风险平价优化

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率（用于确定资产范围和计算组合收益）
        constraints : Dict[str, Any], optional
            约束条件:
            - max_weight: float, 单只股票最大权重
            - min_weight: float, 单只股票最小权重
        **kwargs : dict
            其他参数:
            - risk_budget: np.ndarray, 自定义风险预算

        Returns
        -------
        Dict[str, Any]
            优化结果，包括风险贡献分析
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

        # 确定风险预算
        risk_budget = kwargs.get('risk_budget', self.budget)
        if risk_budget is None:
            # 默认等权风险预算
            risk_budget = np.ones(n_assets) / n_assets
        else:
            risk_budget = np.array(risk_budget)
            risk_budget = risk_budget / risk_budget.sum()  # 标准化

        # 设置边界
        lower_bound = constraints.get('min_weight', 0.001) if constraints else 0.001  # 避免零权重
        upper_bound = constraints.get('max_weight', 1.0) if constraints else 1.0
        bounds = [(lower_bound, upper_bound) for _ in range(n_assets)]

        # 约束：权重和为1
        constraint_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # 初始权重
        initial_weights = np.ones(n_assets) / n_assets

        # 执行优化
        try:
            result = minimize(
                lambda w: self._risk_parity_objective(w, cov_array, risk_budget),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance
                }
            )

            if not result.success:
                # 尝试使用L-BFGS-B方法
                logger.warning("SLSQP failed, trying L-BFGS-B")

                # L-BFGS-B不支持等式约束，使用惩罚项
                def penalized_objective(w):
                    base_obj = self._risk_parity_objective(w, cov_array, risk_budget)
                    penalty = 1000 * (np.sum(w) - 1.0) ** 2
                    return base_obj + penalty

                result = minimize(
                    penalized_objective,
                    initial_weights,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': self.max_iterations,
                        'ftol': self.tolerance
                    }
                )

            if not result.success:
                # 使用简化的Newton-Raphson方法
                logger.warning("Optimization methods failed, using iterative approach")
                optimal_weights = self._iterative_risk_parity(cov_array, risk_budget, max_iter=1000)
            else:
                optimal_weights = result.x

            # 标准化权重
            optimal_weights = optimal_weights / optimal_weights.sum()

            # 计算组合指标
            portfolio_return = np.dot(optimal_weights, returns_array)
            portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_array, optimal_weights)))

            # 计算实际风险贡献
            risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_array)
            total_risk_contribution = np.sum(risk_contributions)
            risk_contribution_pct = risk_contributions / total_risk_contribution if total_risk_contribution > 0 else risk_contributions

            # 构建结果
            optimization_result = {
                'weights': pd.Series(optimal_weights, index=assets),
                'expected_return': float(portfolio_return),
                'risk': float(portfolio_risk),
                'sharpe_ratio': float(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0,
                'optimization_status': 'success',
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'risk_contributions': pd.Series(risk_contributions, index=assets),
                'risk_contribution_pct': pd.Series(risk_contribution_pct, index=assets),
                'target_budget': pd.Series(risk_budget, index=assets),
                'budget_deviation': float(np.sum((risk_contribution_pct - risk_budget) ** 2))
            }

            self.track_optimization(optimization_result)

            logger.info(f"RiskParity optimization successful: Risk={portfolio_risk:.4f}, "
                       f"Budget deviation={optimization_result['budget_deviation']:.6f}")

            return optimization_result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    def _iterative_risk_parity(self,
                              cov_matrix: np.ndarray,
                              target_budget: np.ndarray,
                              max_iter: int = 1000,
                              tol: float = 1e-6) -> np.ndarray:
        """
        使用迭代方法求解风险平价

        基于Spinu (2013)的分析解法

        Parameters
        ----------
        cov_matrix : np.ndarray
            协方差矩阵
        target_budget : np.ndarray
            目标风险预算
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差

        Returns
        -------
        np.ndarray
            最优权重
        """
        n = len(target_budget)
        weights = np.ones(n) / n

        for iteration in range(max_iter):
            # 边际风险贡献
            sigma_w = np.dot(cov_matrix, weights)
            portfolio_vol = np.sqrt(np.dot(weights, sigma_w))

            if portfolio_vol < 1e-10:
                break

            marginal_risk = sigma_w / portfolio_vol

            # 更新权重
            new_weights = target_budget / marginal_risk
            new_weights = new_weights / new_weights.sum()

            # 检查收敛
            if np.max(np.abs(new_weights - weights)) < tol:
                logger.info(f"Iterative risk parity converged in {iteration + 1} iterations")
                break

            weights = new_weights

        return weights

    def calculate_efficient_frontier(self,
                                   expected_returns: pd.Series,
                                   risk_range: Tuple[float, float],
                                   n_points: int = 20) -> pd.DataFrame:
        """
        计算风险预算有效前沿

        通过调整风险预算，生成不同的风险平价组合

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率
        risk_range : Tuple[float, float]
            风险范围（本优化器可能忽略）
        n_points : int
            前沿点数

        Returns
        -------
        pd.DataFrame
            不同风险预算下的组合
        """
        self._check_risk_model()

        n_assets = len(expected_returns)

        # 生成不同的风险预算
        # 从集中到分散的预算分配
        frontier_points = []

        for concentration in np.linspace(0.1, 1.0, n_points):
            # 调整预算的集中度
            base_budget = np.ones(n_assets) / n_assets

            # 创建偏向高收益资产的预算
            returns_rank = expected_returns.rank(ascending=False)
            adjustment = (returns_rank / returns_rank.max()) ** (1 / concentration)
            adjusted_budget = base_budget * adjustment.values
            adjusted_budget = adjusted_budget / adjusted_budget.sum()

            try:
                result = self.optimize(
                    expected_returns,
                    risk_budget=adjusted_budget
                )

                frontier_points.append({
                    'risk': result['risk'],
                    'return': result['expected_return'],
                    'sharpe': result['sharpe_ratio'],
                    'weights': result['weights'],
                    'concentration': concentration
                })

            except Exception as e:
                logger.warning(f"Failed at concentration {concentration}: {str(e)}")
                continue

        if not frontier_points:
            raise OptimizationConvergenceError(
                'RiskParityOptimizer',
                message="Failed to compute risk parity frontier"
            )

        frontier_df = pd.DataFrame(frontier_points)

        logger.info(f"Computed risk parity frontier with {len(frontier_df)} points")

        return frontier_df
