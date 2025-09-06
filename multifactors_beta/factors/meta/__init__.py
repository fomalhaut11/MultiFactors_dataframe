"""
因子元数据管理模块
"""

from .factor_registry import (
    FactorRegistry,
    FactorMetadata,
    FactorType,
    NeutralizationCategory,
    get_factor_registry
)

__all__ = [
    'FactorRegistry',
    'FactorMetadata', 
    'FactorType',
    'NeutralizationCategory',
    'get_factor_registry'
]