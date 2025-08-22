"""
滑点模型

模拟交易滑点成本
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SlippageModel:
    """
    滑点模型
    
    模拟由于市场流动性、交易时机等因素导致的价格滑点
    """
    
    def __init__(self,
                 fixed_slippage: float = 0.0005,
                 proportional_slippage: float = 0.0001,
                 sqrt_slippage: float = 0.0,
                 model_type: str = 'fixed'):
        """
        初始化滑点模型
        
        Parameters
        ----------
        fixed_slippage : float
            固定滑点率（默认0.05%）
        proportional_slippage : float
            比例滑点系数
        sqrt_slippage : float
            平方根滑点系数
        model_type : str
            滑点模型类型：
            - 'fixed': 固定滑点
            - 'proportional': 比例滑点
            - 'sqrt': 平方根滑点
            - 'linear': 线性滑点
            - 'combined': 组合模型
        """
        self.fixed_slippage = fixed_slippage
        self.proportional_slippage = proportional_slippage
        self.sqrt_slippage = sqrt_slippage
        self.model_type = model_type
        
        logger.debug(f"SlippageModel初始化: 类型={model_type}, 固定滑点={fixed_slippage:.4f}")
    
    def calculate_slippage(self,
                          trade_amount: Union[float, pd.Series],
                          avg_daily_volume: Optional[Union[float, pd.Series]] = None,
                          volatility: Optional[Union[float, pd.Series]] = None) -> Union[float, pd.Series]:
        """
        计算滑点成本
        
        Parameters
        ----------
        trade_amount : float or pd.Series
            交易金额（绝对值）
        avg_daily_volume : float or pd.Series, optional
            平均日成交量（用于比例滑点计算）
        volatility : float or pd.Series, optional
            波动率（用于动态滑点调整）
            
        Returns
        -------
        float or pd.Series
            滑点成本（相对于交易金额的比例）
        """
        if isinstance(trade_amount, pd.Series):
            return trade_amount.apply(
                lambda x: self._calculate_single_slippage(x, avg_daily_volume, volatility)
            )
        else:
            return self._calculate_single_slippage(trade_amount, avg_daily_volume, volatility)
    
    def _calculate_single_slippage(self,
                                  amount: float,
                                  avg_volume: Optional[float] = None,
                                  volatility: Optional[float] = None) -> float:
        """计算单笔交易的滑点"""
        amount = abs(amount)
        
        if amount <= 0:
            return 0.0
        
        if self.model_type == 'fixed':
            return self.fixed_slippage
        
        elif self.model_type == 'proportional':
            if avg_volume is None or avg_volume <= 0:
                logger.warning("比例滑点模型需要平均成交量数据，使用固定滑点")
                return self.fixed_slippage
            
            # 交易量占比
            volume_ratio = amount / avg_volume
            slippage = self.proportional_slippage * volume_ratio
            
            return slippage
        
        elif self.model_type == 'sqrt':
            if avg_volume is None or avg_volume <= 0:
                logger.warning("平方根滑点模型需要平均成交量数据，使用固定滑点")
                return self.fixed_slippage
            
            # 平方根模型：滑点与交易量平方根成正比
            volume_ratio = amount / avg_volume
            slippage = self.sqrt_slippage * np.sqrt(volume_ratio)
            
            return slippage
        
        elif self.model_type == 'linear':
            if avg_volume is None or avg_volume <= 0:
                return self.fixed_slippage
            
            # 线性模型：滑点与交易量占比成正比
            volume_ratio = amount / avg_volume
            slippage = self.fixed_slippage + self.proportional_slippage * volume_ratio
            
            return slippage
        
        elif self.model_type == 'combined':
            # 组合模型：固定 + 比例 + 平方根
            base_slippage = self.fixed_slippage
            
            if avg_volume is not None and avg_volume > 0:
                volume_ratio = amount / avg_volume
                proportional_component = self.proportional_slippage * volume_ratio
                sqrt_component = self.sqrt_slippage * np.sqrt(volume_ratio)
                
                total_slippage = base_slippage + proportional_component + sqrt_component
            else:
                total_slippage = base_slippage
            
            # 波动率调整
            if volatility is not None and volatility > 0:
                volatility_multiplier = 1 + volatility  # 高波动率增加滑点
                total_slippage *= volatility_multiplier
            
            return total_slippage
        
        else:
            raise ValueError(f"未知的滑点模型类型: {self.model_type}")
    
    def calculate_daily_slippage(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算当日总滑点成本
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            交易记录，包含列：
            - stock: 股票代码
            - quantity: 交易数量
            - price: 交易价格
            - amount: 交易金额
            - avg_volume: 平均成交量（可选）
            - volatility: 波动率（可选）
            
        Returns
        -------
        Dict[str, float]
            滑点成本明细
        """
        if trades_df.empty:
            return {'total_slippage': 0.0, 'buy_slippage': 0.0, 'sell_slippage': 0.0}
        
        total_slippage = 0.0
        buy_slippage = 0.0
        sell_slippage = 0.0
        
        for _, trade in trades_df.iterrows():
            amount = abs(trade['amount'])
            avg_volume = trade.get('avg_volume', None)
            volatility = trade.get('volatility', None)
            
            slippage_rate = self._calculate_single_slippage(amount, avg_volume, volatility)
            slippage_cost = amount * slippage_rate
            
            total_slippage += slippage_cost
            
            if trade['quantity'] > 0:  # 买入
                buy_slippage += slippage_cost
            else:  # 卖出
                sell_slippage += slippage_cost
        
        return {
            'total_slippage': total_slippage,
            'buy_slippage': buy_slippage,
            'sell_slippage': sell_slippage,
            'avg_slippage_rate': total_slippage / trades_df['amount'].abs().sum() if not trades_df.empty else 0
        }

class AdaptiveSlippageModel(SlippageModel):
    """
    自适应滑点模型
    
    根据市场条件、时间、交易量等因素动态调整滑点
    """
    
    def __init__(self,
                 base_slippage: float = 0.0005,
                 time_factors: Optional[Dict[str, float]] = None,
                 market_factors: Optional[Dict[str, float]] = None,
                 **kwargs):
        """
        初始化自适应滑点模型
        
        Parameters
        ----------
        base_slippage : float
            基础滑点率
        time_factors : Dict[str, float], optional
            时间因素调整，格式：
            {
                'market_open': 1.5,    # 开盘时段滑点倍数
                'market_close': 1.3,   # 收盘时段滑点倍数
                'lunch_break': 1.2     # 午休时段滑点倍数
            }
        market_factors : Dict[str, float], optional
            市场因素调整，格式：
            {
                'high_volatility': 1.5,  # 高波动率时期
                'low_liquidity': 2.0,    # 低流动性时期
                'trend_day': 1.2         # 趋势明显的日子
            }
        """
        super().__init__(fixed_slippage=base_slippage, **kwargs)
        
        self.time_factors = time_factors or {
            'market_open': 1.5,
            'market_close': 1.3,
            'lunch_break': 1.2,
            'normal': 1.0
        }
        
        self.market_factors = market_factors or {
            'high_volatility': 1.5,
            'low_liquidity': 2.0,
            'trend_day': 1.2,
            'normal': 1.0
        }
        
        # 当前市场状态
        self.current_time_factor = 'normal'
        self.current_market_factors = ['normal']
    
    def set_market_condition(self,
                           time_factor: str = 'normal',
                           market_factors: list = None):
        """
        设置当前市场条件
        
        Parameters
        ----------
        time_factor : str
            时间因素
        market_factors : list
            市场因素列表
        """
        self.current_time_factor = time_factor
        self.current_market_factors = market_factors or ['normal']
        
        logger.debug(f"市场条件更新: 时间={time_factor}, 市场={market_factors}")
    
    def _calculate_single_slippage(self,
                                  amount: float,
                                  avg_volume: Optional[float] = None,
                                  volatility: Optional[float] = None) -> float:
        """计算自适应滑点（重写）"""
        # 基础滑点
        base_slippage = super()._calculate_single_slippage(amount, avg_volume, volatility)
        
        # 时间因素调整
        time_multiplier = self.time_factors.get(self.current_time_factor, 1.0)
        
        # 市场因素调整
        market_multiplier = 1.0
        for factor in self.current_market_factors:
            factor_multiplier = self.market_factors.get(factor, 1.0)
            market_multiplier *= factor_multiplier
        
        # 总调整倍数
        total_multiplier = time_multiplier * market_multiplier
        
        adaptive_slippage = base_slippage * total_multiplier
        
        return adaptive_slippage

# 预定义的滑点配置
SLIPPAGE_CONFIGS = {
    'low_impact': {
        'fixed_slippage': 0.0003,
        'model_type': 'fixed'
    },
    'moderate_impact': {
        'fixed_slippage': 0.0005,
        'proportional_slippage': 0.0001,
        'model_type': 'linear'
    },
    'high_impact': {
        'fixed_slippage': 0.001,
        'proportional_slippage': 0.0002,
        'sqrt_slippage': 0.0001,
        'model_type': 'combined'
    },
    'market_making': {
        'fixed_slippage': 0.0001,
        'model_type': 'fixed'
    }
}

def create_slippage_model(config_name: str = 'moderate_impact', **overrides) -> SlippageModel:
    """
    创建预定义的滑点模型
    
    Parameters
    ----------
    config_name : str
        配置名称
    **overrides
        覆盖参数
        
    Returns
    -------
    SlippageModel
        滑点模型实例
    """
    if config_name not in SLIPPAGE_CONFIGS:
        raise ValueError(f"未知的滑点配置: {config_name}")
    
    config = SLIPPAGE_CONFIGS[config_name].copy()
    config.update(overrides)
    
    return SlippageModel(**config)