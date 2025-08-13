#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块
"""

# 编码修复工具 - 安全版本自动执行
from .encoding_fix import fix_encoding, safe_print, printer, EncodingSafePrinter
fix_encoding()  # 安全版本的自动修复编码

from .exit_handler import (
    install_exit_handler, 
    register_cleanup_function, 
    force_cleanup,
    cleanup_temp_files,
    cleanup_pickle_temp_files
)

# 数据清洗工具
from .data_cleaning import OutlierHandler, Normalizer, DataCleaner

# 技术指标工具
from .technical_indicators import (
    MovingAverageCalculator,
    VolatilityCalculator,
    TechnicalIndicators
)

# 因子处理工具
from .factor_processing import FactorOrthogonalizer, FactorProcessor

# 市场微观结构工具
from .market_microstructure import MarketCapFilter, LiquidityMetrics

__all__ = [
    # 编码修复工具
    'fix_encoding',
    'safe_print',
    'printer',
    'EncodingSafePrinter',
    # 退出处理
    'install_exit_handler',
    'register_cleanup_function', 
    'force_cleanup',
    'cleanup_temp_files',
    'cleanup_pickle_temp_files',
    # 数据处理工具
    'OutlierHandler',
    'Normalizer',
    'DataCleaner',
    'MovingAverageCalculator',
    'VolatilityCalculator',
    'TechnicalIndicators',
    'FactorOrthogonalizer',
    'FactorProcessor',
    'MarketCapFilter',
    'LiquidityMetrics'
]