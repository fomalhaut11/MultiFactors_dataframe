"""
因子选择模块

提供因子筛选和选择功能，支持多种选择策略和筛选器
"""

from .factor_selector import FactorSelector
from .base.selector_base import SelectorBase
from .factor_pool import FactorPool
from .filters import (
    PerformanceFilter,
    CorrelationFilter,
    StabilityFilter,
    CompositeFilter
)
from .strategies import (
    TopNSelector,
    ThresholdSelector,
    ClusteringSelector
)

__all__ = [
    'FactorSelector',
    'SelectorBase', 
    'FactorPool',
    'PerformanceFilter',
    'CorrelationFilter',
    'StabilityFilter',
    'CompositeFilter',
    'TopNSelector',
    'ThresholdSelector',
    'ClusteringSelector'
]

__version__ = '1.0.0'