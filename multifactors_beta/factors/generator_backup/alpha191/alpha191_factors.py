"""
Alpha191 标准化因子类实现

将每个 Alpha 因子封装为标准的因子类
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from .alpha191_base import Alpha191Base
from .alpha191_calculator import Alpha191Calculator

logger = logging.getLogger(__name__)


class Alpha001Factor(Alpha191Base):
    """Alpha001: 基于成交量和价格变动的相关性因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=1,
            description="基于成交量变化和价格变动的相关性因子"
        )
        self.min_periods = 10  # 至少需要10个交易日
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame], 
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_001()


class Alpha002Factor(Alpha191Base):
    """Alpha002: 基于价格位置的变化因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=2,
            description="基于收盘价在高低价区间位置变化的因子"
        )
        self.min_periods = 5
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_002()


class Alpha003Factor(Alpha191Base):
    """Alpha003: 基于价格突破的累积因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=3,
            description="基于价格相对前期收盘价突破情况的累积因子"
        )
        self.min_periods = 10
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_003()


class Alpha004Factor(Alpha191Base):
    """Alpha004: 基于价格均值回归和成交量的条件因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=4,
            description="基于价格均值回归趋势和成交量条件的复合因子"
        )
        self.min_periods = 25
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_004()


class Alpha005Factor(Alpha191Base):
    """Alpha005: 基于成交量和价格时序排名相关性的因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=5,
            description="基于成交量和最高价时序排名相关性的因子"
        )
        self.min_periods = 15
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_005()


class Alpha006Factor(Alpha191Base):
    """Alpha006: 基于开盘价和最高价组合变化的趋势因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=6,
            description="基于开盘价和最高价加权组合变化趋势的因子"
        )
        self.min_periods = 8
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_006()


class Alpha007Factor(Alpha191Base):
    """Alpha007: 基于VWAP与收盘价差异和成交量变化的复合因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=7,
            description="基于VWAP与收盘价差异和成交量变化的复合因子"
        )
        self.min_periods = 8
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_007()


class Alpha008Factor(Alpha191Base):
    """Alpha008: 基于价格中枢和VWAP组合变化的因子"""
    
    def __init__(self):
        super().__init__(
            alpha_num=8,
            description="基于价格中枢与VWAP加权组合变化的反转因子"
        )
        self.min_periods = 8
    
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        calculator = Alpha191Calculator()
        calculator.prepare_data(data, benchmark_data)
        return calculator.alpha_008()


# ==================== 因子注册表 ====================

# Alpha 因子注册表：编号 -> 因子类
ALPHA_FACTOR_REGISTRY = {
    1: Alpha001Factor,
    2: Alpha002Factor,
    3: Alpha003Factor,
    4: Alpha004Factor,
    5: Alpha005Factor,
    6: Alpha006Factor,
    7: Alpha007Factor,
    8: Alpha008Factor,
    # 更多因子将逐步添加...
}


# ==================== 便捷函数 ====================

def create_alpha_factor(alpha_num: int) -> Alpha191Base:
    """
    创建指定编号的 Alpha 因子实例
    
    Parameters
    ----------
    alpha_num : int
        因子编号
        
    Returns
    -------
    Alpha191Base
        因子实例
    """
    if alpha_num not in ALPHA_FACTOR_REGISTRY:
        raise NotImplementedError(f"Alpha{alpha_num:03d} 尚未实现")
    
    factor_class = ALPHA_FACTOR_REGISTRY[alpha_num]
    return factor_class()


def get_implemented_alphas():
    """获取已实现的 Alpha 因子编号列表"""
    return sorted(ALPHA_FACTOR_REGISTRY.keys())


def get_alpha_factor_info(alpha_num: int) -> Dict[str, Any]:
    """
    获取指定 Alpha 因子的信息
    
    Parameters
    ----------
    alpha_num : int
        因子编号
        
    Returns
    -------
    dict
        因子信息
    """
    if alpha_num not in ALPHA_FACTOR_REGISTRY:
        return {
            'alpha_num': alpha_num,
            'implemented': False,
            'error': f"Alpha{alpha_num:03d} 尚未实现"
        }
    
    factor = create_alpha_factor(alpha_num)
    info = factor.get_factor_info()
    info['implemented'] = True
    
    return info


def list_all_alpha_factors():
    """列出所有 Alpha 因子的信息"""
    factor_info = {}
    
    for alpha_num in range(1, 192):  # Alpha001 到 Alpha191
        factor_info[alpha_num] = get_alpha_factor_info(alpha_num)
    
    return factor_info


# ==================== 因子分组 ====================

# 按特征分组
ALPHA_GROUPS = {
    'momentum': [1, 6, 8],  # 动量类因子
    'mean_reversion': [2, 4, 7],  # 均值回归类因子
    'volume_price': [1, 3, 5, 7],  # 量价关系因子
    'pattern': [3, 4],  # 模式识别因子
    'composite': [4, 7, 8],  # 复合因子
}

def get_alpha_group(group_name: str):
    """
    获取指定分组的 Alpha 因子
    
    Parameters
    ----------
    group_name : str
        分组名称
        
    Returns
    -------
    list
        因子编号列表
    """
    if group_name not in ALPHA_GROUPS:
        available_groups = list(ALPHA_GROUPS.keys())
        raise ValueError(f"未知分组: {group_name}. 可用分组: {available_groups}")
    
    return ALPHA_GROUPS[group_name]


# ==================== 批量创建函数 ====================

def create_alpha_factors(alpha_nums: list = None):
    """
    批量创建 Alpha 因子实例
    
    Parameters
    ----------
    alpha_nums : list, optional
        因子编号列表，默认创建所有已实现因子
        
    Returns
    -------
    dict
        编号到因子实例的映射
    """
    if alpha_nums is None:
        alpha_nums = get_implemented_alphas()
    
    factors = {}
    failed = []
    
    for alpha_num in alpha_nums:
        try:
            factors[alpha_num] = create_alpha_factor(alpha_num)
        except NotImplementedError:
            failed.append(alpha_num)
            logger.warning(f"Alpha{alpha_num:03d} 尚未实现，跳过")
    
    if failed:
        logger.info(f"成功创建 {len(factors)} 个因子，{len(failed)} 个尚未实现")
    else:
        logger.info(f"成功创建所有 {len(factors)} 个 Alpha 因子")
    
    return factors