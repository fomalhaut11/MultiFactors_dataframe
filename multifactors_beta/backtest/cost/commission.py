"""
手续费模型

模拟各种手续费计算方式
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class CommissionModel:
    """
    手续费模型
    
    支持多种手续费计算方式，包括固定费率、分层费率、最小手续费等
    """
    
    def __init__(self, 
                 commission_rate: float = 0.001,
                 min_commission: float = 5.0,
                 max_commission: Optional[float] = None,
                 commission_tiers: Optional[Dict[float, float]] = None):
        """
        初始化手续费模型
        
        Parameters
        ----------
        commission_rate : float
            基础手续费率（默认0.1%）
        min_commission : float
            最小手续费金额
        max_commission : float, optional
            最大手续费金额
        commission_tiers : Dict[float, float], optional
            分层手续费，格式：{交易金额阈值: 费率}
            例如：{0: 0.001, 100000: 0.0008, 1000000: 0.0005}
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.commission_tiers = commission_tiers or {}
        
        logger.debug(f"CommissionModel初始化: 费率={commission_rate:.4f}, 最小手续费={min_commission}")
    
    def calculate_commission(self, 
                           trade_amount: Union[float, pd.Series], 
                           trade_type: str = 'both') -> Union[float, pd.Series]:
        """
        计算手续费
        
        Parameters
        ----------
        trade_amount : float or pd.Series
            交易金额（绝对值）
        trade_type : str
            交易类型 ('buy', 'sell', 'both')
            - 'buy': 只收买入手续费
            - 'sell': 只收卖出手续费（可能包含印花税）
            - 'both': 买卖双边都收费
            
        Returns
        -------
        float or pd.Series
            手续费金额
        """
        if isinstance(trade_amount, pd.Series):
            return trade_amount.apply(lambda x: self._calculate_single_commission(x, trade_type))
        else:
            return self._calculate_single_commission(trade_amount, trade_type)
    
    def _calculate_single_commission(self, amount: float, trade_type: str) -> float:
        """计算单笔交易手续费"""
        amount = abs(amount)  # 确保是正数
        
        if amount <= 0:
            return 0.0
        
        # 根据分层费率计算
        if self.commission_tiers:
            rate = self._get_tiered_rate(amount)
        else:
            rate = self.commission_rate
        
        # 基础手续费
        commission = amount * rate
        
        # 应用最小和最大限制
        if commission < self.min_commission:
            commission = self.min_commission
        
        if self.max_commission is not None and commission > self.max_commission:
            commission = self.max_commission
        
        # 根据交易类型调整
        if trade_type == 'sell':
            # 卖出可能包含印花税（中国A股0.1%）
            stamp_duty = amount * 0.001  # 0.1%印花税
            commission += stamp_duty
        elif trade_type == 'both':
            # 双边收费
            commission *= 2
            # 卖出部分加印花税
            stamp_duty = amount * 0.001
            commission += stamp_duty
        
        return commission
    
    def _get_tiered_rate(self, amount: float) -> float:
        """根据交易金额获取对应的费率"""
        applicable_rate = self.commission_rate
        
        # 找到适用的费率层级
        for threshold, rate in sorted(self.commission_tiers.items()):
            if amount >= threshold:
                applicable_rate = rate
            else:
                break
        
        return applicable_rate
    
    def calculate_daily_commission(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算当日总手续费
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            交易记录，包含列：
            - stock: 股票代码
            - quantity: 交易数量（正数买入，负数卖出）
            - price: 交易价格
            - amount: 交易金额
            
        Returns
        -------
        Dict[str, float]
            手续费明细
        """
        if trades_df.empty:
            return {'total_commission': 0.0, 'buy_commission': 0.0, 'sell_commission': 0.0, 'stamp_duty': 0.0}
        
        # 分离买卖交易
        buy_trades = trades_df[trades_df['quantity'] > 0]
        sell_trades = trades_df[trades_df['quantity'] < 0]
        
        # 计算买入手续费
        buy_commission = 0.0
        if not buy_trades.empty:
            buy_amounts = buy_trades['amount'].abs()
            buy_commission = buy_amounts.apply(
                lambda x: self._calculate_single_commission(x, 'buy')
            ).sum()
        
        # 计算卖出手续费
        sell_commission = 0.0
        stamp_duty = 0.0
        if not sell_trades.empty:
            sell_amounts = sell_trades['amount'].abs()
            # 分别计算手续费和印花税
            for amount in sell_amounts:
                commission = amount * self.commission_rate
                if commission < self.min_commission:
                    commission = self.min_commission
                if self.max_commission is not None and commission > self.max_commission:
                    commission = self.max_commission
                
                sell_commission += commission
                stamp_duty += amount * 0.001  # 印花税
        
        total_commission = buy_commission + sell_commission + stamp_duty
        
        return {
            'total_commission': total_commission,
            'buy_commission': buy_commission,
            'sell_commission': sell_commission,
            'stamp_duty': stamp_duty
        }

class AdvancedCommissionModel(CommissionModel):
    """
    高级手续费模型
    
    支持更复杂的费用结构，如VIP等级、交易频次优惠等
    """
    
    def __init__(self, 
                 base_rate: float = 0.001,
                 min_commission: float = 5.0,
                 vip_levels: Optional[Dict[str, Dict]] = None,
                 volume_discounts: Optional[Dict[float, float]] = None,
                 **kwargs):
        """
        初始化高级手续费模型
        
        Parameters
        ----------
        base_rate : float
            基础费率
        min_commission : float
            最小手续费
        vip_levels : Dict[str, Dict], optional
            VIP等级配置，格式：
            {
                'level1': {'min_assets': 100000, 'rate': 0.0008},
                'level2': {'min_assets': 1000000, 'rate': 0.0005}
            }
        volume_discounts : Dict[float, float], optional
            交易量折扣，格式：{月交易量阈值: 折扣比例}
        """
        super().__init__(commission_rate=base_rate, min_commission=min_commission, **kwargs)
        
        self.vip_levels = vip_levels or {}
        self.volume_discounts = volume_discounts or {}
        
        # 交易量统计（用于动态调整费率）
        self.monthly_volume = 0.0
        self.current_vip_level = None
    
    def set_client_status(self, 
                         total_assets: float, 
                         monthly_volume: float = 0.0) -> None:
        """
        设置客户状态，用于确定适用的费率
        
        Parameters
        ----------
        total_assets : float
            客户总资产
        monthly_volume : float
            月交易量
        """
        self.monthly_volume = monthly_volume
        
        # 确定VIP等级
        self.current_vip_level = None
        max_assets_requirement = 0
        
        for level, config in self.vip_levels.items():
            min_assets = config.get('min_assets', 0)
            if total_assets >= min_assets and min_assets > max_assets_requirement:
                self.current_vip_level = level
                max_assets_requirement = min_assets
        
        logger.debug(f"客户状态: 资产={total_assets:,.0f}, VIP等级={self.current_vip_level}")
    
    def get_effective_rate(self, trade_amount: float) -> float:
        """
        获取有效费率（考虑VIP和交易量折扣）
        
        Parameters
        ----------
        trade_amount : float
            交易金额
            
        Returns
        -------
        float
            有效费率
        """
        # 基础费率
        if self.commission_tiers:
            base_rate = self._get_tiered_rate(trade_amount)
        else:
            base_rate = self.commission_rate
        
        # VIP折扣
        if self.current_vip_level and self.current_vip_level in self.vip_levels:
            vip_rate = self.vip_levels[self.current_vip_level].get('rate', base_rate)
            base_rate = min(base_rate, vip_rate)
        
        # 交易量折扣
        volume_discount = 1.0
        for threshold, discount in sorted(self.volume_discounts.items()):
            if self.monthly_volume >= threshold:
                volume_discount = discount
        
        effective_rate = base_rate * volume_discount
        
        return effective_rate
    
    def _calculate_single_commission(self, amount: float, trade_type: str) -> float:
        """计算单笔交易手续费（重写以支持动态费率）"""
        amount = abs(amount)
        
        if amount <= 0:
            return 0.0
        
        # 获取有效费率
        rate = self.get_effective_rate(amount)
        
        # 基础手续费
        commission = amount * rate
        
        # 应用最小和最大限制
        if commission < self.min_commission:
            commission = self.min_commission
        
        if self.max_commission is not None and commission > self.max_commission:
            commission = self.max_commission
        
        # 根据交易类型调整
        if trade_type == 'sell':
            # 卖出加印花税
            stamp_duty = amount * 0.001
            commission += stamp_duty
        elif trade_type == 'both':
            # 双边收费
            commission *= 2
            # 卖出部分加印花税
            stamp_duty = amount * 0.001
            commission += stamp_duty
        
        return commission

# 预定义的手续费配置
COMMISSION_CONFIGS = {
    'default': {
        'commission_rate': 0.001,
        'min_commission': 5.0
    },
    'low_cost': {
        'commission_rate': 0.0005,
        'min_commission': 1.0
    },
    'institutional': {
        'commission_rate': 0.0003,
        'min_commission': 0.0,
        'commission_tiers': {
            0: 0.0003,
            1000000: 0.0002,
            10000000: 0.0001
        }
    },
    'china_a_share': {
        'commission_rate': 0.0008,
        'min_commission': 5.0,
        'max_commission': None
    }
}

def create_commission_model(config_name: str = 'default', **overrides) -> CommissionModel:
    """
    创建预定义的手续费模型
    
    Parameters
    ----------
    config_name : str
        配置名称
    **overrides
        覆盖参数
        
    Returns
    -------
    CommissionModel
        手续费模型实例
    """
    if config_name not in COMMISSION_CONFIGS:
        raise ValueError(f"未知的手续费配置: {config_name}")
    
    config = COMMISSION_CONFIGS[config_name].copy()
    config.update(overrides)
    
    return CommissionModel(**config)