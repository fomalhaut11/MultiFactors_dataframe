"""
混合因子模块

需要多种数据源的复合因子计算模块

主要因子子模块：
- 估值因子（financial + market_cap）：BP、EP、SP、CFP等
- 市值因子（直接基于market_cap）：Size、LogSize等  
- 流动性因子（price + volume）：Liquidity、Turnover等
- 质量因子（financial + technical）：Quality、Stability等
- 风格因子（multiple data）：Growth、Value、Momentum等

架构特点：
- 每个子类别有独立的计算器
- 统一的混合因子接口
- 灵活的数据需求定义
"""

from .valuation_factors import (
    ValuationFactorCalculator,
    calculate_valuation_factor,
    calculate_multiple_valuation_factors,
    get_all_valuation_factors
)

from .mixed_factor_manager import (
    MixedFactorManager,
    get_mixed_factor_manager,
    calculate_mixed_factor,
    calculate_multiple_mixed_factors,
    get_all_mixed_factors,
    get_mixed_factor_data_requirements
)

# 未来扩展的混合因子计算器
# from .size_factors import SizeFactorCalculator
# from .liquidity_factors import LiquidityFactorCalculator  
# from .quality_factors import QualityFactorCalculator

__all__ = [
    # 统一管理器（推荐使用）
    'MixedFactorManager',
    'get_mixed_factor_manager', 
    'calculate_mixed_factor',
    'calculate_multiple_mixed_factors',
    'get_all_mixed_factors',
    'get_mixed_factor_data_requirements',
    
    # 估值因子（直接使用）
    'ValuationFactorCalculator',
    'calculate_valuation_factor',
    'calculate_multiple_valuation_factors', 
    'get_all_valuation_factors',
    
    # 未来扩展
    # 'SizeFactorCalculator',
    # 'LiquidityFactorCalculator', 
    # 'QualityFactorCalculator'
]

# 版本信息
__version__ = '1.0.0'