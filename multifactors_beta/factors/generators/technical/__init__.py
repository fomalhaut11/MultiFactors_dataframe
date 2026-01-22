"""
技术指标工具包

提供各种技术分析指标的计算功能，专门服务于技术因子生成。

主要模块：
- moving_average: 移动平均类指标
- oscillator: 振荡器指标（RSI、MACD、KDJ、CCI等）
- volatility: 波动率指标
- momentum: 动量因子
- reversal: 反转因子
- volume: 成交量因子

Author: AI Assistant
Date: 2025-09-03
Updated: 2025-01
"""

from .moving_average import MovingAverageCalculator
from .oscillator import TechnicalIndicators
from .volatility import VolatilityCalculator
from .momentum import MomentumCalculator
from .reversal import ReversalCalculator
from .volume import VolumeCalculator

__all__ = [
    'MovingAverageCalculator',
    'TechnicalIndicators',
    'VolatilityCalculator',
    'MomentumCalculator',
    'ReversalCalculator',
    'VolumeCalculator'
]

# 版本信息
__version__ = '2.0.0'