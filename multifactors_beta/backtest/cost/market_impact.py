"""
市场冲击模型

模拟大额交易对市场价格的冲击影响
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class MarketImpactModel:
    """
    市场冲击模型
    
    模拟交易对市场价格的临时和永久冲击
    """
    
    def __init__(self,
                 temporary_impact_coeff: float = 0.0001,
                 permanent_impact_coeff: float = 0.00005,
                 impact_decay_factor: float = 0.9,
                 nonlinear_factor: float = 0.5):
        """
        初始化市场冲击模型
        
        Parameters
        ----------
        temporary_impact_coeff : float
            临时冲击系数
        permanent_impact_coeff : float
            永久冲击系数
        impact_decay_factor : float
            冲击衰减因子（用于临时冲击的恢复）
        nonlinear_factor : float
            非线性因子（用于大额交易的超线性冲击）
        """
        self.temporary_impact_coeff = temporary_impact_coeff
        self.permanent_impact_coeff = permanent_impact_coeff
        self.impact_decay_factor = impact_decay_factor
        self.nonlinear_factor = nonlinear_factor
        
        # 维护每只股票的价格冲击状态
        self.price_impacts = {}  # {stock: {'temp_impact': float, 'perm_impact': float}}
        
        logger.debug(f"MarketImpactModel初始化: 临时冲击={temporary_impact_coeff}, 永久冲击={permanent_impact_coeff}")
    
    def calculate_market_impact(self,
                              stock: str,
                              trade_amount: float,
                              trade_direction: int,  # 1 for buy, -1 for sell
                              avg_daily_volume: float,
                              current_price: float) -> Dict[str, float]:
        """
        计算市场冲击
        
        Parameters
        ----------
        stock : str
            股票代码
        trade_amount : float
            交易金额（绝对值）
        trade_direction : int
            交易方向（1买入，-1卖出）
        avg_daily_volume : float
            平均日成交量
        current_price : float
            当前价格
            
        Returns
        -------
        Dict[str, float]
            市场冲击详情
        """
        if avg_daily_volume <= 0:
            logger.warning(f"股票{stock}的平均成交量为0，无法计算市场冲击")
            return {'temporary_impact': 0.0, 'permanent_impact': 0.0, 'total_impact': 0.0}
        
        # 计算交易量占比
        volume_ratio = trade_amount / avg_daily_volume
        
        # 临时冲击（交易完成后会逐渐恢复）
        temporary_impact = self._calculate_temporary_impact(volume_ratio, trade_direction)
        
        # 永久冲击（对价格的持久影响）
        permanent_impact = self._calculate_permanent_impact(volume_ratio, trade_direction)
        
        # 更新股票的冲击状态
        self._update_impact_state(stock, temporary_impact, permanent_impact)
        
        total_impact = temporary_impact + permanent_impact
        
        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': total_impact,
            'volume_ratio': volume_ratio
        }
    
    def _calculate_temporary_impact(self, volume_ratio: float, direction: int) -> float:
        """计算临时市场冲击"""
        # 基础临时冲击：与交易量的平方根成正比
        base_impact = self.temporary_impact_coeff * np.sqrt(volume_ratio)
        
        # 非线性调整：大额交易的超线性冲击
        if volume_ratio > 0.05:  # 如果交易量超过日均成交量的5%
            nonlinear_adjustment = (volume_ratio - 0.05) * self.nonlinear_factor
            base_impact *= (1 + nonlinear_adjustment)
        
        # 应用方向
        return base_impact * direction
    
    def _calculate_permanent_impact(self, volume_ratio: float, direction: int) -> float:
        """计算永久市场冲击"""
        # 永久冲击通常小于临时冲击，与交易量的对数成正比
        base_impact = self.permanent_impact_coeff * np.log(1 + volume_ratio)
        
        # 应用方向
        return base_impact * direction
    
    def _update_impact_state(self, stock: str, temp_impact: float, perm_impact: float):
        """更新股票的冲击状态"""
        if stock not in self.price_impacts:
            self.price_impacts[stock] = {'temp_impact': 0.0, 'perm_impact': 0.0}
        
        # 累积永久冲击
        self.price_impacts[stock]['perm_impact'] += perm_impact
        
        # 临时冲击会衰减，然后加上新的冲击
        old_temp_impact = self.price_impacts[stock]['temp_impact']
        decayed_temp_impact = old_temp_impact * self.impact_decay_factor
        self.price_impacts[stock]['temp_impact'] = decayed_temp_impact + temp_impact
    
    def get_current_impact(self, stock: str) -> Dict[str, float]:
        """
        获取股票当前的价格冲击
        
        Parameters
        ----------
        stock : str
            股票代码
            
        Returns
        -------
        Dict[str, float]
            当前冲击状态
        """
        if stock not in self.price_impacts:
            return {'temp_impact': 0.0, 'perm_impact': 0.0, 'total_impact': 0.0}
        
        temp_impact = self.price_impacts[stock]['temp_impact']
        perm_impact = self.price_impacts[stock]['perm_impact']
        
        return {
            'temp_impact': temp_impact,
            'perm_impact': perm_impact,
            'total_impact': temp_impact + perm_impact
        }
    
    def apply_time_decay(self):
        """应用时间衰减（每个时间步调用一次）"""
        for stock in self.price_impacts:
            # 临时冲击衰减
            self.price_impacts[stock]['temp_impact'] *= self.impact_decay_factor
            
            # 永久冲击也可以有微小的衰减（可选）
            # self.price_impacts[stock]['perm_impact'] *= 0.999
    
    def calculate_adjusted_price(self, stock: str, base_price: float) -> float:
        """
        计算受市场冲击影响后的价格
        
        Parameters
        ----------
        stock : str
            股票代码
        base_price : float
            基础价格
            
        Returns
        -------
        float
            调整后的价格
        """
        impact = self.get_current_impact(stock)
        total_impact_ratio = impact['total_impact']
        
        # 价格调整：正向冲击推高价格，负向冲击压低价格
        adjusted_price = base_price * (1 + total_impact_ratio)
        
        return adjusted_price
    
    def calculate_daily_impact_cost(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算当日总的市场冲击成本
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            交易记录，包含列：
            - stock: 股票代码
            - quantity: 交易数量
            - price: 交易价格
            - amount: 交易金额
            - avg_volume: 平均成交量
            
        Returns
        -------
        Dict[str, float]
            市场冲击成本明细
        """
        if trades_df.empty:
            return {'total_impact_cost': 0.0, 'temporary_cost': 0.0, 'permanent_cost': 0.0}
        
        total_temp_cost = 0.0
        total_perm_cost = 0.0
        
        for _, trade in trades_df.iterrows():
            stock = trade['stock']
            amount = abs(trade['amount'])
            direction = 1 if trade['quantity'] > 0 else -1
            avg_volume = trade.get('avg_volume', amount * 10)  # 默认假设
            current_price = trade['price']
            
            # 计算市场冲击
            impact = self.calculate_market_impact(
                stock=stock,
                trade_amount=amount,
                trade_direction=direction,
                avg_daily_volume=avg_volume,
                current_price=current_price
            )
            
            # 计算成本（冲击导致的价格不利变动）
            temp_cost = abs(impact['temporary_impact']) * amount
            perm_cost = abs(impact['permanent_impact']) * amount
            
            total_temp_cost += temp_cost
            total_perm_cost += perm_cost
        
        return {
            'total_impact_cost': total_temp_cost + total_perm_cost,
            'temporary_cost': total_temp_cost,
            'permanent_cost': total_perm_cost,
            'avg_impact_rate': (total_temp_cost + total_perm_cost) / trades_df['amount'].abs().sum() if not trades_df.empty else 0
        }
    
    def reset_impacts(self, stocks: Optional[list] = None):
        """
        重置市场冲击状态
        
        Parameters
        ----------
        stocks : list, optional
            要重置的股票列表，如果为None则重置所有
        """
        if stocks is None:
            self.price_impacts.clear()
        else:
            for stock in stocks:
                if stock in self.price_impacts:
                    del self.price_impacts[stock]
        
        logger.debug(f"市场冲击状态已重置: {stocks or '全部'}")

class AdvancedMarketImpactModel(MarketImpactModel):
    """
    高级市场冲击模型
    
    考虑更多因素：流动性、波动率、市场微观结构等
    """
    
    def __init__(self,
                 base_temp_coeff: float = 0.0001,
                 base_perm_coeff: float = 0.00005,
                 liquidity_adjustment: bool = True,
                 volatility_adjustment: bool = True,
                 time_of_day_adjustment: bool = True,
                 **kwargs):
        """
        初始化高级市场冲击模型
        
        Parameters
        ----------
        base_temp_coeff : float
            基础临时冲击系数
        base_perm_coeff : float
            基础永久冲击系数
        liquidity_adjustment : bool
            是否根据流动性调整
        volatility_adjustment : bool
            是否根据波动率调整
        time_of_day_adjustment : bool
            是否根据交易时间调整
        """
        super().__init__(
            temporary_impact_coeff=base_temp_coeff,
            permanent_impact_coeff=base_perm_coeff,
            **kwargs
        )
        
        self.liquidity_adjustment = liquidity_adjustment
        self.volatility_adjustment = volatility_adjustment
        self.time_of_day_adjustment = time_of_day_adjustment
        
        # 时间调整因子
        self.time_adjustments = {
            'market_open': 1.5,     # 开盘时冲击较大
            'market_close': 1.3,    # 收盘时冲击较大
            'lunch_break': 1.2,     # 午休时流动性较低
            'normal': 1.0
        }
        
        self.current_time_period = 'normal'
    
    def set_time_period(self, period: str):
        """设置当前时间段"""
        self.current_time_period = period
    
    def calculate_market_impact(self,
                              stock: str,
                              trade_amount: float,
                              trade_direction: int,
                              avg_daily_volume: float,
                              current_price: float,
                              bid_ask_spread: Optional[float] = None,
                              volatility: Optional[float] = None) -> Dict[str, float]:
        """
        计算高级市场冲击（重写）
        
        新增参数：
        - bid_ask_spread: 买卖价差（流动性指标）
        - volatility: 历史波动率
        """
        # 基础冲击计算
        base_impact = super().calculate_market_impact(
            stock, trade_amount, trade_direction, avg_daily_volume, current_price
        )
        
        # 调整因子
        adjustment_factor = 1.0
        
        # 流动性调整
        if self.liquidity_adjustment and bid_ask_spread is not None:
            # 买卖价差越大，流动性越差，冲击越大
            liquidity_factor = 1 + (bid_ask_spread / current_price) * 10
            adjustment_factor *= liquidity_factor
        
        # 波动率调整
        if self.volatility_adjustment and volatility is not None:
            # 高波动率增加市场冲击
            volatility_factor = 1 + volatility * 2
            adjustment_factor *= volatility_factor
        
        # 时间调整
        if self.time_of_day_adjustment:
            time_factor = self.time_adjustments.get(self.current_time_period, 1.0)
            adjustment_factor *= time_factor
        
        # 应用调整
        adjusted_impact = {
            'temporary_impact': base_impact['temporary_impact'] * adjustment_factor,
            'permanent_impact': base_impact['permanent_impact'] * adjustment_factor,
            'volume_ratio': base_impact['volume_ratio'],
            'adjustment_factor': adjustment_factor
        }
        
        adjusted_impact['total_impact'] = (
            adjusted_impact['temporary_impact'] + adjusted_impact['permanent_impact']
        )
        
        # 更新状态（使用调整后的冲击）
        self._update_impact_state(
            stock, 
            adjusted_impact['temporary_impact'], 
            adjusted_impact['permanent_impact']
        )
        
        return adjusted_impact

# 预定义的市场冲击配置
MARKET_IMPACT_CONFIGS = {
    'low_impact': {
        'temporary_impact_coeff': 0.00005,
        'permanent_impact_coeff': 0.00002,
        'impact_decay_factor': 0.95
    },
    'moderate_impact': {
        'temporary_impact_coeff': 0.0001,
        'permanent_impact_coeff': 0.00005,
        'impact_decay_factor': 0.9
    },
    'high_impact': {
        'temporary_impact_coeff': 0.0002,
        'permanent_impact_coeff': 0.0001,
        'impact_decay_factor': 0.85,
        'nonlinear_factor': 1.0
    },
    'institutional': {
        'temporary_impact_coeff': 0.00003,
        'permanent_impact_coeff': 0.00001,
        'impact_decay_factor': 0.98
    }
}

def create_market_impact_model(config_name: str = 'moderate_impact', **overrides) -> MarketImpactModel:
    """
    创建预定义的市场冲击模型
    
    Parameters
    ----------
    config_name : str
        配置名称
    **overrides
        覆盖参数
        
    Returns
    -------
    MarketImpactModel
        市场冲击模型实例
    """
    if config_name not in MARKET_IMPACT_CONFIGS:
        raise ValueError(f"未知的市场冲击配置: {config_name}")
    
    config = MARKET_IMPACT_CONFIGS[config_name].copy()
    config.update(overrides)
    
    return MarketImpactModel(**config)