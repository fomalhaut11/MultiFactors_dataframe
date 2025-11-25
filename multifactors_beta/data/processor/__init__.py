"""
数据处理模块

提供股票数据的处理、转换和特征生成功能
"""
from .base_processor import BaseDataProcessor
from .price_processor import PriceDataProcessor
from .return_calculator import ReturnCalculator
from .financial_processor import FinancialDataProcessor
from .data_processing_pipeline import DataProcessingPipeline
from .enhanced_pipeline import EnhancedDataProcessingPipeline
from .sector_metrics_calculator import SectorMetricsCalculator, SectorValuationFromStockPE
from .integrated_pipeline import IntegratedDataPipeline, DataUpdateScheduler

__all__ = [
    'BaseDataProcessor',
    'PriceDataProcessor',
    'ReturnCalculator',
    'FinancialDataProcessor',
    'DataProcessingPipeline',
    'EnhancedDataProcessingPipeline',
    'SectorMetricsCalculator',          # 新名称（推荐使用）
    'SectorValuationFromStockPE',       # 向后兼容别名
    'IntegratedDataPipeline',
    'DataUpdateScheduler'
]