"""
因子综合评估模块
提供多维度的因子评估和诊断功能
"""

from .factor_evaluator import FactorEvaluator, EvaluationResult
from .scoring.score_calculator import ScoreCalculator, ScoreResult
from .dimensions.base_dimension import BaseDimension, DimensionScore
from .dimensions import (
    ProfitabilityDimension,
    StabilityDimension,
    TradabilityDimension,
    UniquenessDimension,
    TimelinesDimension
)

__all__ = [
    # 主评估器
    'FactorEvaluator',
    'EvaluationResult',
    
    # 评分系统
    'ScoreCalculator',
    'ScoreResult',
    
    # 维度基类
    'BaseDimension',
    'DimensionScore',
    
    # 具体维度
    'ProfitabilityDimension',
    'StabilityDimension',
    'TradabilityDimension',
    'UniquenessDimension',
    'TimelinesDimension',
]