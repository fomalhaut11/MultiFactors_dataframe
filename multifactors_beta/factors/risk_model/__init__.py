"""
风险模型模块

提供完整的风险建模、度量、分解和组合优化功能
"""

# 核心风险模型
from .models.barra_model import BarraModel
from .models.covariance_model import CovarianceModel
from .models.factor_model import FactorModel

# 估计器
from .estimators.sample_covariance import SampleCovarianceEstimator
from .estimators.ledoit_wolf import LedoitWolfEstimator
from .estimators.exponential_weighted import ExponentialWeightedEstimator
from .estimators.robust_estimators import RobustCovarianceEstimator

# 组合优化器、度量工具、预测工具等将在后续阶段实现
# from .optimizer.mean_variance import MeanVarianceOptimizer
# from .optimizer.risk_parity import RiskParityOptimizer  
# from .optimizer.black_litterman import BlackLittermanOptimizer
# from .metrics.risk_metrics import RiskMetrics
# from .metrics.var_cvar import VaRCalculator
# from .decomposition.risk_decomposer import RiskDecomposer
# from .prediction.volatility_forecast import VolatilityForecast
# from .prediction.correlation_forecast import CorrelationForecast

# 基础接口和异常
from .base.risk_model_base import RiskModelBase
from .base.optimizer_base import OptimizerBase
from .base.exceptions import (
    RiskModelError,
    ModelNotFittedError, 
    SingularCovarianceError,
    OptimizationConvergenceError,
    InsufficientDataError
)

__all__ = [
    # 核心风险模型
    'BarraModel',
    'CovarianceModel', 
    'FactorModel',
    
    # 估计器
    'SampleCovarianceEstimator',
    'LedoitWolfEstimator',
    'ExponentialWeightedEstimator',
    'RobustCovarianceEstimator',
    
    # 基础类
    'RiskModelBase',
    'OptimizerBase',
    
    # 异常类
    'RiskModelError',
    'ModelNotFittedError',
    'SingularCovarianceError', 
    'OptimizationConvergenceError',
    'InsufficientDataError'
    
    # 未来将添加的组件
    # 'MeanVarianceOptimizer',
    # 'RiskParityOptimizer', 
    # 'BlackLittermanOptimizer',
    # 'RiskMetrics',
    # 'VaRCalculator',
    # 'RiskDecomposer',
    # 'VolatilityForecast',
    # 'CorrelationForecast'
]

__version__ = '1.0.0'