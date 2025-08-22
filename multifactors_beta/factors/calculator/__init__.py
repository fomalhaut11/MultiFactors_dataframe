"""
因子计算器模块

提供统一的因子计算和管理接口
位于generator层之上，作为业务组合器使用
"""

from .factor_calculator import FactorCalculator, FactorDataLoader

__all__ = [
    'FactorCalculator',
    'FactorDataLoader'
]

# 版本信息
__version__ = '1.0.0'