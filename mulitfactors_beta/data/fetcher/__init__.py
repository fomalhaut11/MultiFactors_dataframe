"""
数据获取模块

提供从数据库获取股票数据的功能
"""

from .BasicDataLocalization import BasicDataLocalization
from .data_fetcher import DataFetcher
from .chunked_price_fetcher import ChunkedPriceFetcher
from .incremental_price_updater import IncrementalPriceUpdater

__all__ = [
    'BasicDataLocalization',    # 基础数据本地化
    'DataFetcher',             # 通用数据获取器
    'ChunkedPriceFetcher',     # 分块价格获取器
    'IncrementalPriceUpdater', # 增量价格更新器
]

# 版本信息
__version__ = '1.0.0'

# 便捷函数
def fetch_price_data(start_date=None, end_date=None, stocks=None):
    """快速获取价格数据"""
    fetcher = DataFetcher()
    return fetcher.fetch_price(start_date, end_date, stocks)

def update_price_incremental():
    """快速进行增量更新"""
    updater = IncrementalPriceUpdater()
    return updater.update()