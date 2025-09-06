"""
财务数据处理工具模块

提供财务报表数据处理的核心工具函数:
- TTM (Trailing Twelve Months) 计算
- 单季度值转换
- YoY/QoQ 增长率计算  
- 日频数据扩展
- Z-Score标准化

所有函数都是纯工具函数，不包含Factor类定义
"""

from .financial_report_processor import FinancialReportProcessor, TimeSeriesProcessor

# 导出核心处理器
__all__ = [
    'FinancialReportProcessor', 
    'TimeSeriesProcessor',  # 兼容性别名
]

# 便捷函数导出 - 直接从处理器暴露常用方法
def calculate_ttm(data):
    """计算TTM (Trailing Twelve Months) 值"""
    return FinancialReportProcessor.calculate_ttm(data)

def calculate_single_quarter(data):
    """将累积值转换为单季度值"""
    return FinancialReportProcessor.calculate_single_quarter(data)

def calculate_yoy(data):
    """计算YoY (Year-over-Year) 同比增长率"""
    return FinancialReportProcessor.calculate_yoy(data)

def calculate_qoq(data):
    """计算QoQ (Quarter-over-Quarter) 环比增长率"""
    return FinancialReportProcessor.calculate_qoq(data)

def calculate_zscore(data, window=12, min_periods=4):
    """计算财务指标的时序Z-Score"""
    return FinancialReportProcessor.calculate_zscore(data, window, min_periods)

def expand_to_daily_vectorized(factor_data, release_dates, trading_dates):
    """向量化的日频扩展方法"""
    return FinancialReportProcessor.expand_to_daily_vectorized(
        factor_data, release_dates, trading_dates
    )

# 将便捷函数添加到__all__
__all__.extend([
    'calculate_ttm',
    'calculate_single_quarter', 
    'calculate_yoy',
    'calculate_qoq',
    'calculate_zscore',
    'expand_to_daily_vectorized',
])