"""
数据获取模块

提供从数据库获取股票数据的功能
"""

# 导入数据获取器相关类
from .data_fetcher import (
    BaseDataFetcher,
    StockDataFetcher, 
    MarketDataFetcher,
    DataFetcherManager,
    data_fetcher_manager
)
from .chunked_price_fetcher import ChunkedPriceFetcher
from .incremental_price_updater import IncrementalPriceUpdater
from .incremental_stop_price_updater import IncrementalStopPriceUpdater
from .incremental_financial_updater import IncrementalFinancialUpdater

# 导入基础数据本地化的主要函数
from .BasicDataLocalization import (
    GetStockDayDataDFFromSql,
    Get3SheetsFromSql,
    GetMacroIndexFromSql_save,
    GetAllDayPriceDataFromSql_save
)

__all__ = [
    # 数据获取器类
    'BaseDataFetcher',         # 数据获取器基类
    'StockDataFetcher',        # 股票数据获取器
    'MarketDataFetcher',       # 市场数据获取器
    'DataFetcherManager',      # 数据获取器管理器
    'data_fetcher_manager',    # 全局获取器管理器实例
    'ChunkedPriceFetcher',     # 分块价格获取器
    'IncrementalPriceUpdater', # 增量价格更新器
    'IncrementalStopPriceUpdater', # 增量涨跌停价格更新器
    'IncrementalFinancialUpdater', # 增量财务数据更新器
    # 基础数据本地化函数
    'GetStockDayDataDFFromSql',
    'Get3SheetsFromSql', 
    'GetMacroIndexFromSql_save',
    'GetAllDayPriceDataFromSql_save'
]

# 版本信息
__version__ = '1.0.0'

# 便捷函数
def fetch_price_data(start_date=None, end_date=None, **kwargs):
    """快速获取价格数据"""
    stock_fetcher = StockDataFetcher()
    return stock_fetcher.fetch_data('price', begin_date=start_date, end_date=end_date, **kwargs)

def update_price_incremental():
    """快速进行增量更新"""
    updater = IncrementalPriceUpdater()
    return updater.update_price_file()