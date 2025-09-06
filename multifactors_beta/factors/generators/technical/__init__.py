"""
技术指标工具包

提供各种技术分析指标的计算功能，专门服务于技术因子生成。

主要模块：
- moving_average: 移动平均类指标
- oscillator: 振荡器指标（RSI、MACD等）  
- volatility: 波动率指标
- trend: 趋势指标

Author: AI Assistant
Date: 2025-09-03
"""

from .moving_average import MovingAverageCalculator
from .oscillator import TechnicalIndicators
from .volatility import VolatilityCalculator

__all__ = [
    'MovingAverageCalculator',
    'TechnicalIndicators', 
    'VolatilityCalculator'
]

# 版本信息
__version__ = '1.0.0'