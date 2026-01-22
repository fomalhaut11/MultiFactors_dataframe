"""
组合优化器模块

提供多种组合优化策略的实现

包含的优化器:
- MeanVarianceOptimizer: 均值-方差优化
- MinVarianceOptimizer: 最小方差优化
- MaxSharpeOptimizer: 最大夏普比率优化
- RiskParityOptimizer: 风险平价优化

Author: AI Assistant
Date: 2025-01
"""

from .mean_variance import MeanVarianceOptimizer
from .min_variance import MinVarianceOptimizer
from .max_sharpe import MaxSharpeOptimizer
from .risk_parity import RiskParityOptimizer

__all__ = [
    'MeanVarianceOptimizer',
    'MinVarianceOptimizer',
    'MaxSharpeOptimizer',
    'RiskParityOptimizer'
]

__version__ = '1.0.0'
