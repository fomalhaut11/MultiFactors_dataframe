"""
数据处理工具生成器模块

纯工具函数库，专门用于数据处理和计算
不包含Factor类定义，仅提供可复用的计算函数

主要模块：
- financial: 财务数据处理工具
- technical: 技术指标计算工具  
- mixed: 混合数据处理工具
- alpha191: Alpha191因子计算工具
"""

__version__ = '1.0.0'

# 工具模块导入
from . import financial
from . import technical
from . import mixed
from . import alpha191

# 便捷导入 - 财务工具
from .financial import (
    FinancialReportProcessor,
    calculate_ttm,
    calculate_single_quarter,
    calculate_yoy,
    calculate_qoq,
    calculate_zscore,
    expand_to_daily_vectorized
)

# 便捷导入 - 技术指标工具
from .technical import (
    MovingAverageCalculator,
    TechnicalIndicators,
    VolatilityCalculator
)

# 便捷导入 - 混合数据工具
from .mixed import (
    MixedDataProcessor,
    align_financial_with_market,
    calculate_relative_ratio,
    calculate_market_cap_weighted_ratio
)

# 便捷导入 - Alpha191工具
from .alpha191 import (
    convert_to_alpha191_format,
    prepare_alpha191_data,
    ts_rank,
    ts_mean,
    delta,
    rank,
    scale
)

__all__ = [
    # 模块
    'financial',
    'technical', 
    'mixed',
    'alpha191',
    
    # 财务工具
    'FinancialReportProcessor',
    'calculate_ttm',
    'calculate_single_quarter', 
    'calculate_yoy',
    'calculate_qoq',
    'calculate_zscore',
    'expand_to_daily_vectorized',
    
    # 技术指标工具
    'MovingAverageCalculator',
    'TechnicalIndicators',
    'VolatilityCalculator',
    
    # 混合数据工具
    'MixedDataProcessor',
    'align_financial_with_market',
    'calculate_relative_ratio',
    'calculate_market_cap_weighted_ratio',
    
    # Alpha191工具
    'convert_to_alpha191_format',
    'prepare_alpha191_data',
    'ts_rank',
    'ts_mean', 
    'delta',
    'rank',
    'scale',
]