"""
因子分析模块

提供因子筛选、评估、相关性分析和稳定性分析等功能
"""

from .config import AnalyzerConfig, get_analyzer_config
from .screening import FactorScreener

__all__ = [
    'AnalyzerConfig',
    'get_analyzer_config',
    'FactorScreener'
]

# 版本信息
__version__ = '1.0.0'