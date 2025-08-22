from .data_adapter import DataAdapter, FactorDataAdapter

# 延迟导入避免循环依赖
def get_factor_updater():
    """延迟导入FactorUpdater"""
    from .factor_updater import FactorUpdater, UpdateTracker
    return FactorUpdater, UpdateTracker

__all__ = [
    'DataAdapter', 
    'FactorDataAdapter',
    'get_factor_updater'
]