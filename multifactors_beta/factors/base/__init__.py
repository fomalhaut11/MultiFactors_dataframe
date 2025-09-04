"""
因子基础模块

提供所有因子计算的基础类和通用功能
"""

from .factor_base import FactorBase, MultiFactorBase
from .data_processing_mixin import DataProcessingMixin
from .flexible_data_adapter import ColumnMapperMixin, FlexibleDataAdapter
from .validation import DataValidator
from .testable_mixin import TestableMixin

__all__ = [
    # 基础类
    'FactorBase',              # 因子基类
    'MultiFactorBase',         # 多因子基类
    
    # Mixins
    'DataProcessingMixin',     # 数据处理混入
    'TestableMixin',          # 可测试混入
    
    # 适配器
    'ColumnMapperMixin',      # 列映射混入
    'FlexibleDataAdapter',    # 灵活数据适配器
    
    # 验证器
    'DataValidator',          # 数据验证器
]

# 版本信息
__version__ = '1.0.0'