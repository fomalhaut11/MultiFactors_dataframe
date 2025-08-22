"""
回测引擎核心模块

提供权重驱动的投资组合回测功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
import logging
from datetime import datetime, timedelta
import warnings

from ..utils.validation import WeightsValidator
from ..performance.result import BacktestResult
from ..performance.metrics import PerformanceMetrics
from ..cost.commission import CommissionModel
from ..cost.slippage import SlippageModel
from ..cost.market_impact import MarketImpactModel
from ..portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    回测引擎核心类
    
    支持权重驱动的投资组合回测，模拟真实交易环境
    """
    
    def __init__(self,
                 initial_capital: float = 1000000.0,
                 commission_model: Optional[CommissionModel] = None,
                 slippage_model: Optional[SlippageModel] = None,
                 market_impact_model: Optional[MarketImpactModel] = None,
                 trading_constraints=None,
                 benchmark_data: Optional[pd.Series] = None):
        """
        初始化回测引擎
        
        Parameters
        ----------
        initial_capital : float
            初始资金
        commission_model : CommissionModel, optional
            手续费模型
        slippage_model : SlippageModel, optional
            滑点模型
        market_impact_model : MarketImpactModel, optional
            市场冲击模型
        trading_constraints : TradingConstraints, optional
            交易约束检查器
        benchmark_data : pd.Series, optional
            基准数据（日期索引，价格或收益率）
        """
        self.initial_capital = initial_capital
        
        # 成本模型
        self.commission_model = commission_model or CommissionModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        
        # 交易约束
        self.trading_constraints = trading_constraints
        
        # 基准数据
        self.benchmark_data = benchmark_data
        
        # 数据验证器
        self.validator = WeightsValidator()
        
        # 绩效计算器
        self.performance_calculator = PerformanceMetrics()
        
        # 组合管理器
        self.portfolio_manager = None
        
        # 回测结果存储
        self.result = None
        
        logger.info(f"BacktestEngine初始化完成，初始资金: {initial_capital:,.0f}")
    
    def run_with_weights(self,
                        weights_data: pd.DataFrame,
                        price_data: pd.DataFrame,
                        market_data: Optional[Dict[str, pd.DataFrame]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        rebalance_freq: str = 'daily',
                        **kwargs) -> BacktestResult:
        """
        使用权重数据运行回测
        
        Parameters
        ----------
        weights_data : pd.DataFrame
            权重数据，索引为日期，列为股票代码
            每行代表一天的权重分配，应该满足权重和为1
        price_data : pd.DataFrame
            价格数据，索引为日期，列为股票代码
        market_data : Dict[str, pd.DataFrame], optional
            市场数据字典，用于交易约束检查，包含：
            - 'open': 开盘价, 'high': 最高价, 'low': 最低价
            - 'close': 收盘价, 'volume': 成交量, 'prev_close': 昨收盘价
        start_date : str, optional
            回测开始日期
        end_date : str, optional
            回测结束日期
        rebalance_freq : str
            再平衡频率 ('daily', 'weekly', 'monthly')
        **kwargs
            其他参数
            
        Returns
        -------
        BacktestResult
            回测结果对象
        """
        try:
            logger.info("开始权重驱动回测")
            
            # 1. 数据验证和预处理
            validated_weights, aligned_prices = self._prepare_data(
                weights_data, price_data, start_date, end_date
            )
            
            # 2. 初始化组合管理器
            self.portfolio_manager = PortfolioManager(
                initial_capital=self.initial_capital,
                commission_model=self.commission_model,
                slippage_model=self.slippage_model,
                market_impact_model=self.market_impact_model,
                trading_constraints=self.trading_constraints
            )
            
            # 3. 初始化结果记录器
            self.result = BacktestResult(
                initial_capital=self.initial_capital,
                benchmark_data=self.benchmark_data
            )
            
            # 4. 执行回测主循环
            self._run_backtest_loop(
                validated_weights, 
                aligned_prices, 
                market_data,
                rebalance_freq,
                **kwargs
            )
            
            # 5. 计算最终绩效指标
            self.result.finalize_results()
            
            logger.info("回测完成")
            return self.result
            
        except Exception as e:
            logger.error(f"回测过程中发生错误: {str(e)}")
            raise
    
    def _prepare_data(self,
                     weights_data: pd.DataFrame,
                     price_data: pd.DataFrame,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备和验证回测数据"""
        logger.info("开始数据预处理")
        
        # 1. 验证权重数据格式
        validated_weights = self.validator.validate_weights_format(weights_data)
        
        # 2. 日期范围筛选
        if start_date:
            start_date = pd.to_datetime(start_date)
            validated_weights = validated_weights[validated_weights.index >= start_date]
            price_data = price_data[price_data.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            validated_weights = validated_weights[validated_weights.index <= end_date]
            price_data = price_data[price_data.index <= end_date]
        
        # 3. 数据对齐
        common_dates = validated_weights.index.intersection(price_data.index)
        if len(common_dates) == 0:
            raise ValueError("权重数据和价格数据没有共同的日期")
        
        aligned_weights = validated_weights.loc[common_dates]
        aligned_prices = price_data.loc[common_dates]
        
        # 4. 股票代码对齐
        common_stocks = aligned_weights.columns.intersection(aligned_prices.columns)
        if len(common_stocks) == 0:
            raise ValueError("权重数据和价格数据没有共同的股票")
        
        aligned_weights = aligned_weights[common_stocks]
        aligned_prices = aligned_prices[common_stocks]
        
        # 5. 处理缺失数据
        aligned_weights = aligned_weights.fillna(0)
        aligned_prices = aligned_prices.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"数据预处理完成: {len(aligned_weights)} 个交易日, {len(common_stocks)} 只股票")
        
        return aligned_weights, aligned_prices
    
    def _run_backtest_loop(self,
                          weights_data: pd.DataFrame,
                          price_data: pd.DataFrame,
                          market_data: Optional[Dict[str, pd.DataFrame]],
                          rebalance_freq: str,
                          **kwargs):
        """执行回测主循环"""
        dates = weights_data.index
        rebalance_dates = self._get_rebalance_dates(dates, rebalance_freq)
        
        logger.info(f"回测时间范围: {dates[0]} 到 {dates[-1]}")
        logger.info(f"再平衡频率: {rebalance_freq}, 共 {len(rebalance_dates)} 次再平衡")
        
        for i, current_date in enumerate(dates):
            # 获取当前价格
            current_prices = price_data.loc[current_date]
            
            # 检查是否需要再平衡
            need_rebalance = current_date in rebalance_dates
            target_weights = weights_data.loc[current_date] if need_rebalance else None
            
            # 更新组合状态
            daily_result = self.portfolio_manager.update_portfolio(
                date=current_date,
                prices=current_prices,
                target_weights=target_weights,
                force_rebalance=need_rebalance,
                market_data=market_data
            )
            
            # 记录每日结果
            self.result.record_daily_data(
                date=current_date,
                portfolio_value=daily_result['portfolio_value'],
                positions=daily_result['positions'],
                trades=daily_result['trades'],
                costs=daily_result['costs'],
                cash=daily_result['cash']
            )
            
            # 应用市场冲击时间衰减
            self.market_impact_model.apply_time_decay()
            
            # 进度报告
            if (i + 1) % 100 == 0:
                progress = (i + 1) / len(dates) * 100
                logger.info(f"回测进度: {progress:.1f}% ({i + 1}/{len(dates)})")
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
        """获取再平衡日期"""
        if freq == 'daily':
            return list(dates)
        elif freq == 'weekly':
            # 每周一再平衡
            return [date for date in dates if date.weekday() == 0]
        elif freq == 'monthly':
            # 每月第一个交易日再平衡
            monthly_dates = []
            current_month = None
            for date in dates:
                if current_month != date.month:
                    monthly_dates.append(date)
                    current_month = date.month
            return monthly_dates
        else:
            raise ValueError(f"不支持的再平衡频率: {freq}")
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合汇总信息"""
        if self.result is None:
            raise ValueError("请先运行回测")
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.result.get_final_portfolio_value(),
            'total_return': self.result.get_total_return(),
            'annual_return': self.result.get_annual_return(),
            'max_drawdown': self.result.get_max_drawdown(),
            'sharpe_ratio': self.result.get_sharpe_ratio(),
            'total_trades': self.result.get_total_trades(),
            'total_costs': self.result.get_total_costs()
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取详细绩效指标"""
        if self.result is None:
            raise ValueError("请先运行回测")
        
        return self.result.calculate_performance_metrics()
    
    def get_trade_statistics(self) -> Dict:
        """获取交易统计信息"""
        if self.result is None:
            raise ValueError("请先运行回测")
        
        return self.result.calculate_trade_statistics()
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8), **kwargs):
        """绘制回测结果图表"""
        if self.result is None:
            raise ValueError("请先运行回测")
        
        return self.result.plot_performance(figsize=figsize, **kwargs)
    
    def save_results(self, filepath: str, format: str = 'excel'):
        """保存回测结果到文件"""
        if self.result is None:
            raise ValueError("请先运行回测")
        
        self.result.save_to_file(filepath, format)
        logger.info(f"回测结果已保存到: {filepath}")

# 便捷函数
def run_weights_backtest(weights_data: pd.DataFrame,
                        price_data: pd.DataFrame,
                        initial_capital: float = 1000000.0,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        commission_rate: float = 0.001,
                        slippage_rate: float = 0.0005,
                        **kwargs) -> BacktestResult:
    """
    权重回测便捷函数
    
    Parameters
    ----------
    weights_data : pd.DataFrame
        权重数据
    price_data : pd.DataFrame
        价格数据
    initial_capital : float
        初始资金
    start_date : str, optional
        开始日期
    end_date : str, optional
        结束日期
    commission_rate : float
        手续费率
    slippage_rate : float
        滑点率
    **kwargs
        其他参数
        
    Returns
    -------
    BacktestResult
        回测结果
    """
    from ..cost.commission import create_commission_model
    from ..cost.slippage import create_slippage_model
    
    # 创建成本模型
    commission_model = create_commission_model('default', commission_rate=commission_rate)
    slippage_model = create_slippage_model('moderate_impact', fixed_slippage=slippage_rate)
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_model=commission_model,
        slippage_model=slippage_model
    )
    
    # 运行回测
    return engine.run_with_weights(
        weights_data=weights_data,
        price_data=price_data,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )