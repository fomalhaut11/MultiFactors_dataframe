"""
Portfolio Optimizer 单元测试

测试MeanVarianceOptimizer, MinVarianceOptimizer, MaxSharpeOptimizer, RiskParityOptimizer
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock


class TestMeanVarianceOptimizer:
    """MeanVarianceOptimizer测试类"""

    @pytest.fixture
    def mock_risk_model(self):
        """创建模拟风险模型"""
        risk_model = Mock()
        risk_model.is_fitted = True

        # 创建3x3协方差矩阵
        cov_data = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16]
        ])
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        cov_matrix = pd.DataFrame(cov_data, index=stocks, columns=stocks)
        risk_model.predict_covariance = Mock(return_value=cov_matrix)

        return risk_model

    @pytest.fixture
    def expected_returns(self):
        """创建预期收益率"""
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        returns = pd.Series([0.10, 0.15, 0.20], index=stocks)
        return returns

    def test_optimizer_initialization(self, mock_risk_model):
        """测试优化器初始化"""
        from factors.risk_model.optimizer import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(mock_risk_model, risk_free_rate=0.02)

        assert optimizer.risk_model is mock_risk_model
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.method == 'SLSQP'

    def test_optimize_basic(self, mock_risk_model, expected_returns):
        """测试基本优化功能"""
        from factors.risk_model.optimizer import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(mock_risk_model, risk_free_rate=0.02)
        result = optimizer.optimize(expected_returns)

        # 验证返回结构
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'risk' in result
        assert 'sharpe_ratio' in result
        assert 'optimization_status' in result

        # 验证权重和为1
        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # 验证权重非负
        assert (result['weights'] >= -1e-6).all()

        # 验证状态
        assert result['optimization_status'] == 'success'

    def test_optimize_with_constraints(self, mock_risk_model, expected_returns):
        """测试带约束的优化"""
        from factors.risk_model.optimizer import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(mock_risk_model, risk_free_rate=0.02)
        constraints = {
            'max_weight': 0.5,
            'min_weight': 0.1
        }
        result = optimizer.optimize(expected_returns, constraints=constraints)

        # 验证约束满足
        assert (result['weights'] <= 0.5 + 1e-6).all()
        assert (result['weights'] >= 0.1 - 1e-6).all()

    def test_efficient_frontier(self, mock_risk_model, expected_returns):
        """测试有效前沿计算"""
        from factors.risk_model.optimizer import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(mock_risk_model, risk_free_rate=0.02)
        frontier = optimizer.calculate_efficient_frontier(
            expected_returns,
            risk_range=(0.15, 0.35),
            n_points=5
        )

        # 验证返回DataFrame
        assert isinstance(frontier, pd.DataFrame)
        assert 'risk' in frontier.columns
        assert 'return' in frontier.columns
        assert 'sharpe' in frontier.columns


class TestMinVarianceOptimizer:
    """MinVarianceOptimizer测试类"""

    @pytest.fixture
    def mock_risk_model(self):
        """创建模拟风险模型"""
        risk_model = Mock()
        risk_model.is_fitted = True

        cov_data = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16]
        ])
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        cov_matrix = pd.DataFrame(cov_data, index=stocks, columns=stocks)
        risk_model.predict_covariance = Mock(return_value=cov_matrix)

        return risk_model

    @pytest.fixture
    def expected_returns(self):
        """创建预期收益率"""
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        return pd.Series([0.10, 0.15, 0.20], index=stocks)

    def test_min_variance_optimization(self, mock_risk_model, expected_returns):
        """测试最小方差优化"""
        from factors.risk_model.optimizer import MinVarianceOptimizer

        optimizer = MinVarianceOptimizer(mock_risk_model)
        result = optimizer.optimize(expected_returns)

        # 验证返回
        assert result['optimization_status'] == 'success'
        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # 最小方差组合应该偏向低波动资产（Stock_A波动最低）
        assert result['weights']['Stock_A'] > result['weights']['Stock_C']


class TestMaxSharpeOptimizer:
    """MaxSharpeOptimizer测试类"""

    @pytest.fixture
    def mock_risk_model(self):
        """创建模拟风险模型"""
        risk_model = Mock()
        risk_model.is_fitted = True

        cov_data = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16]
        ])
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        cov_matrix = pd.DataFrame(cov_data, index=stocks, columns=stocks)
        risk_model.predict_covariance = Mock(return_value=cov_matrix)

        return risk_model

    @pytest.fixture
    def expected_returns(self):
        """创建预期收益率"""
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        return pd.Series([0.10, 0.15, 0.20], index=stocks)

    def test_max_sharpe_optimization(self, mock_risk_model, expected_returns):
        """测试最大夏普比率优化"""
        from factors.risk_model.optimizer import MaxSharpeOptimizer

        optimizer = MaxSharpeOptimizer(mock_risk_model, risk_free_rate=0.02)
        result = optimizer.optimize(expected_returns)

        # 验证返回
        assert result['optimization_status'] == 'success'
        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # 验证夏普比率为正（收益高于无风险利率）
        assert result['sharpe_ratio'] > 0


class TestRiskParityOptimizer:
    """RiskParityOptimizer测试类"""

    @pytest.fixture
    def mock_risk_model(self):
        """创建模拟风险模型"""
        risk_model = Mock()
        risk_model.is_fitted = True

        # 使用对角协方差矩阵便于验证风险贡献
        cov_data = np.array([
            [0.04, 0.0, 0.0],
            [0.0, 0.09, 0.0],
            [0.0, 0.0, 0.16]
        ])
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        cov_matrix = pd.DataFrame(cov_data, index=stocks, columns=stocks)
        risk_model.predict_covariance = Mock(return_value=cov_matrix)

        return risk_model

    @pytest.fixture
    def expected_returns(self):
        """创建预期收益率"""
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        return pd.Series([0.10, 0.15, 0.20], index=stocks)

    def test_risk_parity_optimization(self, mock_risk_model, expected_returns):
        """测试风险平价优化"""
        from factors.risk_model.optimizer import RiskParityOptimizer

        optimizer = RiskParityOptimizer(mock_risk_model)
        result = optimizer.optimize(expected_returns)

        # 验证返回
        assert result['optimization_status'] == 'success'
        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # 验证风险贡献
        assert 'risk_contributions' in result
        assert 'risk_contribution_pct' in result

        # 风险贡献比例应该接近等权（1/3）
        risk_pct = result['risk_contribution_pct']
        target = 1.0 / 3.0
        # 允许一定误差
        assert all(abs(risk_pct - target) < 0.1)


class TestOptimizerValidation:
    """优化器验证功能测试"""

    @pytest.fixture
    def mock_risk_model(self):
        """创建模拟风险模型"""
        risk_model = Mock()
        risk_model.is_fitted = True

        cov_data = np.eye(3) * 0.04
        stocks = ['Stock_A', 'Stock_B', 'Stock_C']
        cov_matrix = pd.DataFrame(cov_data, index=stocks, columns=stocks)
        risk_model.predict_covariance = Mock(return_value=cov_matrix)

        return risk_model

    def test_validate_expected_returns_with_nan(self, mock_risk_model):
        """测试包含NaN的预期收益率验证"""
        from factors.risk_model.optimizer import MeanVarianceOptimizer
        from factors.risk_model.base.exceptions import DataFormatError

        optimizer = MeanVarianceOptimizer(mock_risk_model)

        # 包含NaN的收益率
        returns_with_nan = pd.Series([0.10, np.nan, 0.20], index=['A', 'B', 'C'])

        with pytest.raises(DataFormatError):
            optimizer.validate_expected_returns(returns_with_nan)

    def test_validate_constraints_invalid_max_weight(self, mock_risk_model):
        """测试无效权重约束验证"""
        from factors.risk_model.optimizer import MeanVarianceOptimizer
        from factors.risk_model.base.exceptions import InvalidParameterError

        optimizer = MeanVarianceOptimizer(mock_risk_model)

        # 无效的最大权重
        constraints = {'max_weight': 1.5}

        with pytest.raises(InvalidParameterError):
            optimizer.validate_constraints(constraints)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
