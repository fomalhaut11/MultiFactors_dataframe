"""
回测系统API设计示例
详细展示如何调用回测模块以及不同的使用模式
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime, date

# ============================================
# 核心API接口设计
# ============================================

class BacktestEngine:
    """
    回测引擎主类 - 核心API接口
    
    支持两种主要使用模式：
    1. 策略驱动：提供策略对象，引擎自动生成信号和权重
    2. 权重驱动：直接提供每日权重，引擎执行交易模拟
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001,
                 market_impact_model: str = 'linear',
                 **kwargs):
        """
        初始化回测引擎
        
        Parameters
        ----------
        initial_capital : float
            初始资金
        commission_rate : float  
            手续费率
        slippage_rate : float
            滑点率
        market_impact_model : str
            市场冲击模型 ('linear', 'sqrt', 'none')
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact_model = market_impact_model
        
        # 内部组件（实际实现时注入）
        self._data_handler = None
        self._portfolio_manager = None
        self._cost_calculator = None
        self._performance_analyzer = None
    
    # ========================================
    # 方式一：策略驱动模式
    # ========================================
    
    def run_strategy(self, 
                    strategy,
                    start_date: Union[str, datetime],
                    end_date: Union[str, datetime],
                    rebalance_freq: str = 'daily',
                    **kwargs) -> 'BacktestResult':
        """
        运行策略驱动的回测
        
        Parameters
        ----------
        strategy : StrategyBase
            策略对象，包含信号生成逻辑
        start_date : str/datetime
            开始日期
        end_date : str/datetime  
            结束日期
        rebalance_freq : str
            调仓频率 ('daily', 'weekly', 'monthly')
            
        Returns
        -------
        BacktestResult
            回测结果对象
        """
        # 这里是核心执行逻辑的框架
        result = BacktestResult()
        
        # 1. 初始化数据和时间序列
        trading_dates = self._get_trading_dates(start_date, end_date)
        
        # 2. 按时间循环执行
        for date in trading_dates:
            # 获取策略信号
            signals = strategy.generate_signals(date)
            
            # 转换为权重
            weights = strategy.signals_to_weights(signals, date)
            
            # 执行交易
            trades = self._execute_trades(weights, date)
            
            # 更新组合
            self._portfolio_manager.update(trades, date)
            
            # 记录业绩
            result.record_daily_performance(date, self._portfolio_manager.get_portfolio_value())
        
        # 3. 计算最终结果
        result.calculate_final_metrics()
        return result
    
    # ========================================  
    # 方式二：权重驱动模式（你关心的）
    # ========================================
    
    def run_with_weights(self,
                        portfolio_weights: pd.DataFrame,
                        price_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> 'BacktestResult':
        """
        使用预定义权重运行回测
        
        Parameters
        ----------
        portfolio_weights : pd.DataFrame
            每日权重数据，格式：
            - index: 日期 (DatetimeIndex) - 每一行代表一天
            - columns: 股票代码 - 每一列代表一只股票
            - values: 权重 (每行权重和为1.0) - 每个值是该股票在该日的权重
            
            示例：
                           000001.SZ  000002.SZ  000300.SZ
            2020-01-01     0.333      0.333      0.334    ← 第一天的权重分配
            2020-01-02     0.300      0.400      0.300    ← 第二天的权重分配
            2020-01-03     0.250      0.450      0.300    ← 第三天的权重分配
            ...
            
            含义：
            - 2020-01-01这天：33.3%买入000001.SZ，33.3%买入000002.SZ，33.4%买入000300.SZ
            - 2020-01-02这天：调整为30%持有000001.SZ，40%持有000002.SZ，30%持有000300.SZ
            
        price_data : pd.DataFrame, optional
            价格数据，如果不提供则自动加载
            
        Returns
        -------
        BacktestResult
            回测结果对象
        """
        result = BacktestResult()
        
        # 1. 数据验证和预处理
        weights = self._validate_weights_data(portfolio_weights)
        if price_data is None:
            price_data = self._load_price_data(weights.columns, weights.index)
        
        # 2. 逐日执行交易
        previous_weights = None
        
        for date in weights.index:
            current_weights = weights.loc[date]
            
            if previous_weights is not None:
                # 计算权重变化，确定需要的交易
                trades = self._calculate_required_trades(
                    previous_weights, current_weights, date
                )
                
                # 执行交易（考虑成本）
                execution_result = self._execute_trades_with_costs(
                    trades, price_data.loc[date], date
                )
                
                # 更新组合
                self._portfolio_manager.update(execution_result, date)
            else:
                # 首日建仓
                initial_trades = self._create_initial_position(current_weights, date)
                self._portfolio_manager.initialize(initial_trades, date)
            
            # 记录当日业绩
            portfolio_value = self._portfolio_manager.get_portfolio_value(date)
            result.record_daily_performance(date, portfolio_value)
            
            previous_weights = current_weights
        
        # 3. 计算绩效指标
        result.calculate_final_metrics()
        return result
    
    # ========================================
    # 方式三：实时权重更新模式
    # ========================================
    
    def run_streaming(self,
                     weight_generator: Callable,
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime]) -> 'BacktestResult':
        """
        流式权重更新模式
        
        适用于需要动态计算权重的场景（如强化学习、在线优化等）
        
        Parameters
        ----------
        weight_generator : Callable
            权重生成函数，签名：weight_generator(date, current_portfolio) -> weights
        start_date : str/datetime
            开始日期
        end_date : str/datetime
            结束日期
            
        Returns
        -------
        BacktestResult
            回测结果对象
        """
        result = BacktestResult()
        trading_dates = self._get_trading_dates(start_date, end_date)
        
        for date in trading_dates:
            # 动态生成权重
            current_portfolio = self._portfolio_manager.get_current_holdings()
            new_weights = weight_generator(date, current_portfolio)
            
            # 执行调仓
            trades = self._calculate_required_trades(
                current_portfolio.weights, new_weights, date
            )
            
            execution_result = self._execute_trades_with_costs(trades, date)
            self._portfolio_manager.update(execution_result, date)
            
            # 记录业绩
            result.record_daily_performance(date, self._portfolio_manager.get_portfolio_value())
        
        result.calculate_final_metrics()
        return result

# ============================================
# 数据结构定义
# ============================================

class BacktestResult:
    """回测结果类"""
    
    def __init__(self):
        self.daily_returns = pd.Series(dtype=float)
        self.daily_positions = pd.DataFrame()
        self.daily_trades = pd.DataFrame() 
        self.daily_costs = pd.Series(dtype=float)
        
        # 绩效指标
        self.total_return = None
        self.annual_return = None
        self.sharpe_ratio = None
        self.max_drawdown = None
        self.calmar_ratio = None
        
        # 交易统计
        self.total_trades = None
        self.turnover_rate = None
        self.total_costs = None
        
    def record_daily_performance(self, date, portfolio_value):
        """记录每日业绩"""
        # 实现记录逻辑
        pass
        
    def calculate_final_metrics(self):
        """计算最终绩效指标"""
        # 实现指标计算
        pass
        
    def generate_report(self, save_path: str = None):
        """生成回测报告"""
        # 实现报告生成
        pass

# ============================================
# 使用示例
# ============================================

def example_usage():
    """演示不同的使用方式"""
    
    # =================================
    # 示例1：权重驱动模式（你关心的）
    # =================================
    
    # 1. 准备每日权重数据 - 正确格式
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')  # 10天示例
    stocks = ['000001.SZ', '000002.SZ', '000300.SZ', '000858.SZ', '002415.SZ']
    
    # 模拟每日权重（实际中这些权重来自你的因子模型/优化器）
    np.random.seed(42)
    weights_data = []
    
    for i, date in enumerate(dates):
        # 每天生成一组权重分配（实际中是你的模型输出）
        if i == 0:
            # 第一天：等权重
            daily_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        elif i == 1:  
            # 第二天：调整权重（比如基于因子信号）
            daily_weights = [0.3, 0.25, 0.15, 0.2, 0.1]
        else:
            # 其他天：随机调整（实际中是策略决定）
            raw_weights = np.random.exponential(1, len(stocks))
            daily_weights = raw_weights / raw_weights.sum()
        
        weights_data.append(daily_weights)
    
    # 正确的权重DataFrame格式：
    # - 每行是一天的权重分配
    # - 每列是一只股票  
    # - 每行权重和为1.0
    portfolio_weights = pd.DataFrame(
        weights_data,      # 每行是一天的权重分配
        index=dates,       # 行索引是日期
        columns=stocks     # 列索引是股票代码
    )
    
    print("权重数据示例:")
    print(portfolio_weights.head())
    print(f"每行权重和: {portfolio_weights.sum(axis=1).head()}")  # 应该都是1.0
    
    # 2. 创建回测引擎
    engine = BacktestEngine(
        initial_capital=1000000,
        commission_rate=0.001,  # 0.1%手续费
        slippage_rate=0.0005    # 0.05%滑点
    )
    
    # 3. 执行回测
    result = engine.run_with_weights(portfolio_weights)
    
    # 4. 查看结果
    print(f"总收益率: {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    
    # =================================
    # 示例2：与因子模块集成
    # =================================
    
    def factor_driven_backtest():
        """展示如何与现有factors模块集成"""
        
        # 1. 使用factors模块生成因子
        from factors.generator.financial import PureFinancialFactorCalculator
        from factors.combiner import FactorCombiner
        from factors.risk_model import BarraModel, MeanVarianceOptimizer
        
        # 2. 生成因子信号
        calculator = PureFinancialFactorCalculator()
        factors = calculator.calculate_multiple(['ROE_ttm', 'CurrentRatio'])
        
        # 3. 因子组合
        combiner = FactorCombiner(method='ic_weight')
        composite_factor = combiner.combine(factors)
        
        # 4. 组合优化（转换因子信号为权重）
        risk_model = BarraModel()
        optimizer = MeanVarianceOptimizer(risk_model)
        
        # 按日期生成权重
        daily_weights = []
        for date in composite_factor.index.get_level_values(0).unique():
            daily_factor = composite_factor.xs(date, level=0)
            expected_returns = daily_factor  # 简化：直接用因子值作为预期收益
            
            opt_result = optimizer.optimize(
                expected_returns=expected_returns,
                constraints={'max_weight': 0.1, 'min_weight': 0.0}
            )
            daily_weights.append(opt_result['weights'])
        
        portfolio_weights = pd.DataFrame(daily_weights)
        
        # 5. 回测
        result = engine.run_with_weights(portfolio_weights)
        return result
    
    # =================================
    # 示例3：实时权重生成
    # =================================
    
    def dynamic_weight_generator(date, current_portfolio):
        """动态权重生成函数"""
        # 这里可以调用实时的因子计算、风险模型等
        # 返回当日的目标权重
        
        # 示例：简单的动量策略
        recent_returns = get_recent_returns(date, lookback=20)  # 假设函数
        momentum_scores = recent_returns.mean()
        
        # 转换为权重
        positive_momentum = momentum_scores[momentum_scores > 0]
        if len(positive_momentum) > 0:
            raw_weights = positive_momentum
            weights = raw_weights / raw_weights.sum()
        else:
            # 现金
            weights = pd.Series()
        
        return weights
    
    # 使用流式模式
    streaming_result = engine.run_streaming(
        weight_generator=dynamic_weight_generator,
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

def get_recent_returns(date, lookback=20):
    """获取最近收益率（示例函数）"""
    # 实际实现中从数据源获取
    pass

if __name__ == "__main__":
    example_usage()