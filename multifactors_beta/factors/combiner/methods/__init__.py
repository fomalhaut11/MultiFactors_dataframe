"""
组合方法模块
"""

from .linear import LinearCombiner
from .orthogonal import OrthogonalCombiner
from .pca import PCACombiner
from .neutralization import NeutralizationCombiner

__all__ = [
    'LinearCombiner',
    'OrthogonalCombiner',
    'PCACombiner',
    'NeutralizationCombiner',
]