"""
因子组合模块

提供多因子线性组合、正交化处理和权重优化功能
"""

from .base.combiner_base import CombinerBase
from .factor_combiner import FactorCombiner

__all__ = [
    'CombinerBase',
    'FactorCombiner',
]