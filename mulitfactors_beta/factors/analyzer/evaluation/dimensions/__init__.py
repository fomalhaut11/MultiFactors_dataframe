"""
评估维度模块
实现各个维度的具体评估逻辑
"""

from .base_dimension import BaseDimension, DimensionScore, CompositeDimension

__all__ = [
    'BaseDimension',
    'DimensionScore',
    'CompositeDimension',
]