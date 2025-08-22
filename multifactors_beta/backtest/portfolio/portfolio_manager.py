"""
投资组合管理模块

管理投资组合的持仓、交易、现金等状态
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    投资组合管理器
    
    负责管理投资组合的持仓状态、执行交易、计算成本等
    """
    
    def __init__(self,
                 initial_capital: float,
                 commission_model,
                 slippage_model,
                 market_impact_model,
                 trading_constraints=None,
                 min_trade_amount: float = 100.0):
        """
        初始化投资组合管理器
        
        Parameters
        ----------
        initial_capital : float
            初始资金
        commission_model : CommissionModel
            手续费模型
        slippage_model : SlippageModel
            滑点模型
        market_impact_model : MarketImpactModel
            市场冲击模型
        trading_constraints : TradingConstraints, optional
            交易约束检查器
        min_trade_amount : float
            最小交易金额（避免零碎交易）
        """
        self.initial_capital = initial_capital
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model
        self.trading_constraints = trading_constraints
        self.min_trade_amount = min_trade_amount
        
        # 投资组合状态
        self.cash = initial_capital  # 现金
        self.positions = {}  # 持仓：{股票代码: 股数}
        self.position_values = {}  # 持仓市值：{股票代码: 市值}
        self.current_prices = {}  # 当前价格
        
        # 交易记录
        self.trade_history = []
        self.cost_history = []
        
        # 组合价值历史
        self.portfolio_value_history = []
        
        logger.info(f"PortfolioManager初始化: 初始资金={initial_capital:,.0f}")
    
    def update_portfolio(self,
                        date: pd.Timestamp,
                        prices: pd.Series,
                        target_weights: Optional[pd.Series] = None,
                        force_rebalance: bool = False,
                        market_data: Optional[Dict] = None) -> Dict:
        """
        更新投资组合状态
        
        Parameters
        ----------
        date : pd.Timestamp
            当前日期
        prices : pd.Series
            当前价格
        target_weights : pd.Series, optional
            目标权重
        force_rebalance : bool
            是否强制再平衡
        market_data : Dict, optional
            市场数据（用于交易约束检查）
            
        Returns
        -------
        Dict
            当日投资组合状态和交易信息
        """
        # 更新当前价格
        self.current_prices = prices.to_dict()
        
        # 更新持仓市值
        self._update_position_values()
        
        # 计算当前组合价值
        current_portfolio_value = self._calculate_portfolio_value()
        
        # 执行交易（如果有目标权重）
        daily_trades = []
        daily_costs = {'total': 0.0, 'commission': 0.0, 'slippage': 0.0, 'impact': 0.0}
        
        if target_weights is not None and force_rebalance:
            # 应用交易约束检查
            filtered_weights = self._apply_trading_constraints(
                date, target_weights, market_data
            )
            daily_trades, daily_costs = self._rebalance_portfolio(
                filtered_weights, current_portfolio_value
            )
        
        # 记录历史
        self.portfolio_value_history.append({
            'date': date,
            'value': current_portfolio_value,
            'cash': self.cash,
            'positions_value': sum(self.position_values.values())
        })
        
        return {
            'portfolio_value': current_portfolio_value,
            'positions': self.positions.copy(),
            'position_values': self.position_values.copy(),
            'cash': self.cash,
            'trades': daily_trades,
            'costs': daily_costs
        }
    
    def _update_position_values(self):
        """更新持仓市值"""
        for stock, shares in self.positions.items():
            if stock in self.current_prices:
                self.position_values[stock] = shares * self.current_prices[stock]
            else:
                # 如果价格不可用，保持之前的市值
                if stock not in self.position_values:
                    self.position_values[stock] = 0.0
    
    def _calculate_portfolio_value(self) -> float:
        """计算总组合价值"""
        positions_value = sum(self.position_values.values())
        return self.cash + positions_value
    
    def _rebalance_portfolio(self,
                           target_weights: pd.Series,
                           current_portfolio_value: float) -> Tuple[List[Dict], Dict]:
        """
        执行组合再平衡
        
        Parameters
        ----------
        target_weights : pd.Series
            目标权重
        current_portfolio_value : float
            当前组合价值
            
        Returns
        -------
        Tuple[List[Dict], Dict]
            交易列表和成本明细
        """
        trades = []
        total_costs = {'total': 0.0, 'commission': 0.0, 'slippage': 0.0, 'impact': 0.0}
        
        # 计算目标持仓金额
        target_amounts = target_weights * current_portfolio_value
        
        # 计算当前持仓金额
        current_amounts = pd.Series(self.position_values)
        
        # 对齐股票列表
        all_stocks = set(target_amounts.index) | set(current_amounts.index)
        
        for stock in all_stocks:
            target_amount = target_amounts.get(stock, 0.0)
            current_amount = current_amounts.get(stock, 0.0)
            
            # 计算需要交易的金额
            trade_amount = target_amount - current_amount
            
            if abs(trade_amount) > self.min_trade_amount:
                # 执行交易
                trade_result = self._execute_trade(
                    stock=stock,
                    trade_amount=trade_amount,
                    current_price=self.current_prices.get(stock, 0)
                )
                
                if trade_result:
                    trades.append(trade_result)
                    
                    # 累计成本
                    for cost_type in total_costs:
                        total_costs[cost_type] += trade_result['costs'].get(cost_type, 0)
        
        return trades, total_costs
    
    def _execute_trade(self,
                      stock: str,
                      trade_amount: float,
                      current_price: float) -> Optional[Dict]:
        """
        执行单笔交易
        
        Parameters
        ----------
        stock : str
            股票代码
        trade_amount : float
            交易金额（正数买入，负数卖出）
        current_price : float
            当前价格
            
        Returns
        -------
        Optional[Dict]
            交易结果
        """
        if current_price <= 0:
            logger.warning(f"股票 {stock} 价格无效: {current_price}")
            return None
        
        # 计算交易股数
        shares_to_trade = int(trade_amount / current_price)
        if shares_to_trade == 0:
            return None
        
        actual_trade_amount = shares_to_trade * current_price
        trade_direction = 1 if shares_to_trade > 0 else -1
        
        # 计算交易成本
        costs = self._calculate_trade_costs(
            stock=stock,
            trade_amount=abs(actual_trade_amount),
            trade_direction=trade_direction,
            current_price=current_price
        )
        
        # 检查现金是否足够（买入时）
        if shares_to_trade > 0:
            total_cost = actual_trade_amount + costs['total']
            if total_cost > self.cash:
                # 调整交易量以适应可用现金
                available_amount = self.cash - costs['commission'] - costs['slippage']
                if available_amount > 0:
                    shares_to_trade = int(available_amount / (current_price * (1 + costs['impact']/abs(actual_trade_amount))))
                    actual_trade_amount = shares_to_trade * current_price
                    if shares_to_trade <= 0:
                        logger.warning(f"现金不足，无法买入 {stock}")
                        return None
                else:
                    logger.warning(f"现金不足，无法买入 {stock}")
                    return None
        
        # 执行交易
        execution_price = self._get_execution_price(stock, current_price, trade_direction)
        
        # 更新持仓
        current_shares = self.positions.get(stock, 0)
        new_shares = current_shares + shares_to_trade
        
        if new_shares == 0:
            # 完全卖出
            self.positions.pop(stock, None)
            self.position_values.pop(stock, None)
        else:
            self.positions[stock] = new_shares
            self.position_values[stock] = new_shares * current_price
        
        # 更新现金
        cash_flow = -actual_trade_amount - costs['total']  # 买入为负，卖出为正
        if shares_to_trade < 0:  # 卖出
            cash_flow = -cash_flow - costs['total']  # 卖出获得现金，减去成本
        
        self.cash += cash_flow
        
        # 记录交易
        trade_record = {
            'stock': stock,
            'shares': shares_to_trade,
            'price': execution_price,
            'amount': actual_trade_amount,
            'direction': 'buy' if shares_to_trade > 0 else 'sell',
            'costs': costs,
            'cash_flow': cash_flow
        }
        
        self.trade_history.append(trade_record)
        self.cost_history.append(costs)
        
        return trade_record
    
    def _calculate_trade_costs(self,
                             stock: str,
                             trade_amount: float,
                             trade_direction: int,
                             current_price: float) -> Dict[str, float]:
        """计算交易成本"""
        costs = {'commission': 0.0, 'slippage': 0.0, 'impact': 0.0, 'total': 0.0}
        
        # 手续费
        trade_type = 'buy' if trade_direction > 0 else 'sell'
        costs['commission'] = self.commission_model.calculate_commission(trade_amount, trade_type)
        
        # 滑点成本
        slippage_rate = self.slippage_model.calculate_slippage(trade_amount)
        costs['slippage'] = trade_amount * slippage_rate
        
        # 市场冲击成本
        # 假设平均日成交量为交易金额的10倍（简化）
        avg_daily_volume = trade_amount * 10
        impact_result = self.market_impact_model.calculate_market_impact(
            stock=stock,
            trade_amount=trade_amount,
            trade_direction=trade_direction,
            avg_daily_volume=avg_daily_volume,
            current_price=current_price
        )
        costs['impact'] = abs(impact_result['total_impact']) * trade_amount
        
        # 总成本
        costs['total'] = costs['commission'] + costs['slippage'] + costs['impact']
        
        return costs
    
    def _get_execution_price(self, stock: str, current_price: float, direction: int) -> float:
        """获取实际成交价格（考虑市场冲击）"""
        impact = self.market_impact_model.get_current_impact(stock)
        adjusted_price = self.market_impact_model.calculate_adjusted_price(stock, current_price)
        return adjusted_price
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """获取当前持仓详情"""
        positions_detail = {}
        
        for stock, shares in self.positions.items():
            current_price = self.current_prices.get(stock, 0)
            market_value = shares * current_price
            
            positions_detail[stock] = {
                'shares': shares,
                'price': current_price,
                'market_value': market_value,
                'weight': market_value / self._calculate_portfolio_value() if self._calculate_portfolio_value() > 0 else 0
            }
        
        return positions_detail
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合汇总"""
        total_value = self._calculate_portfolio_value()
        positions_value = sum(self.position_values.values())
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'cash_ratio': self.cash / total_value if total_value > 0 else 0,
            'positions_count': len(self.positions),
            'total_trades': len(self.trade_history),
            'total_costs': sum(cost['total'] for cost in self.cost_history)
        }
    
    def get_trade_summary(self) -> Dict:
        """获取交易汇总统计"""
        if not self.trade_history:
            return {'total_trades': 0}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        buy_trades = trades_df[trades_df['direction'] == 'buy']
        sell_trades = trades_df[trades_df['direction'] == 'sell']
        
        return {
            'total_trades': len(trades_df),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': trades_df['amount'].sum(),
            'avg_trade_size': trades_df['amount'].mean(),
            'total_costs': sum(cost['total'] for cost in self.cost_history),
            'avg_cost_rate': sum(cost['total'] for cost in self.cost_history) / trades_df['amount'].sum() if trades_df['amount'].sum() > 0 else 0
        }
    
    def _apply_trading_constraints(self,
                                 date: pd.Timestamp,
                                 target_weights: pd.Series,
                                 market_data: Optional[Dict] = None) -> pd.Series:
        """
        应用交易约束检查，过滤不可交易股票
        
        Parameters
        ----------
        date : pd.Timestamp
            当前日期
        target_weights : pd.Series
            目标权重
        market_data : Dict, optional
            市场数据
            
        Returns
        -------
        pd.Series
            过滤后的权重
        """
        if self.trading_constraints is None or market_data is None:
            # 没有约束检查器或市场数据，直接返回原权重
            return target_weights
        
        try:
            # 使用交易约束检查器过滤权重
            filtered_weights = self.trading_constraints.filter_tradable_weights(
                date=date,
                target_weights=target_weights,
                market_data=market_data,
                rebalance_method='proportional'  # 按比例重新分配
            )
            
            return filtered_weights
            
        except Exception as e:
            logger.warning(f"应用交易约束时出错: {str(e)}")
            # 出错时返回原权重
            return target_weights