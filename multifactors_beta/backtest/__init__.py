"""
回测系统模块

提供完整的事件驱动回测功能，支持权重驱动和策略驱动两种模式。
主要用于验证因子策略和组合优化的实际效果。
"""

from .engine.backtest_engine import BacktestEngine, run_weights_backtest
from .performance.result import BacktestResult  
from .performance.metrics import PerformanceMetrics
from .utils.validation import WeightsValidator
from .utils.trading_constraints import TradingConstraints, create_trading_constraints
from .portfolio.portfolio_manager import PortfolioManager
from .cost import CommissionModel, SlippageModel, MarketImpactModel

__all__ = [
    'BacktestEngine',
    'run_weights_backtest',
    'BacktestResult', 
    'PerformanceMetrics',
    'WeightsValidator',
    'TradingConstraints',
    'create_trading_constraints',
    'PortfolioManager',
    'CommissionModel',
    'SlippageModel', 
    'MarketImpactModel'
]

__version__ = '1.0.0'