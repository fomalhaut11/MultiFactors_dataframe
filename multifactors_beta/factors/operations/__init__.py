#!/usr/bin/env python3
"""
因子操作模块

提供常用的因子操作功能，包括：
- 截面操作：排序、标准化、分位数等
- 时序操作：滚动统计、移动平均等
- 因子组合：线性组合、正交化等

所有操作都遵循标准的MultiIndex Series格式：
- 第一级索引：TradingDates（交易日期）
- 第二级索引：StockCodes（股票代码）

⚠️ 重要：本模块仅处理已展开到交易日维度的因子数据
- 输入必须是[TradingDates, StockCodes]格式
- 不处理[ReportDates, StockCodes]格式的财报数据

📋 模块边界说明：
- 本模块：对生成的因子进行通用数学/统计变换（因子处理阶段）
- factors/base/time_series_processor：财报数据的业务逻辑变换（因子生成阶段）
  * TTM、YoY等财务概念操作
  * 基于财报发布日期的截面标准化  
  * 财报数据展开到交易日的处理

Author: AI Assistant
Date: 2025-08-26
"""

from .cross_sectional import (
    cross_rank,
    cross_zscore,
    cross_percentile,
    cross_winsorize,
    cross_neutralize
)

from .time_series import (
    rolling_mean,
    rolling_std,
    rolling_corr,
    ewm,
    lag,
    diff,
    returns
)

from .combination import (
    linear_combine,
    orthogonalize,
    residualize
)

from .composite import (
    momentum,
    volatility,
    mean_reversion,
    quality_score,
    size_neutral
)

from .pipeline import (
    FactorPipeline,
    pipeline
)

# 便捷的别名
rank = cross_rank
zscore = cross_zscore
percentile = cross_percentile
winsorize = cross_winsorize
neutralize = cross_neutralize

ma = rolling_mean
std = rolling_std
corr = rolling_corr

combine = linear_combine
orthogonalize_factor = orthogonalize
residualize_factor = residualize

# 复合因子别名
momentum_factor = momentum
volatility_factor = volatility
mean_reversion_factor = mean_reversion

__all__ = [
    # 截面操作
    'cross_rank', 'cross_zscore', 'cross_percentile', 'cross_winsorize', 'cross_neutralize',
    
    # 时序操作
    'rolling_mean', 'rolling_std', 'rolling_corr', 'ewm', 'lag', 'diff', 'returns',
    
    # 因子组合
    'linear_combine', 'orthogonalize', 'residualize',
    
    # 复合因子
    'momentum', 'volatility', 'mean_reversion', 'quality_score', 'size_neutral',
    
    # 管道操作
    'FactorPipeline', 'pipeline',
    
    # 便捷别名
    'rank', 'zscore', 'percentile', 'winsorize', 'neutralize',
    'ma', 'std', 'corr',
    'combine', 'orthogonalize_factor', 'residualize_factor',
    'momentum_factor', 'volatility_factor', 'mean_reversion_factor'
]

# 版本信息
__version__ = '1.0.0'