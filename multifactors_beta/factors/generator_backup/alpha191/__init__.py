"""
Alpha191 因子模块

国泰君安 Alpha191 因子的标准化实现
基于经典的 WorldQuant Alpha191 因子集合

主要特性：
- 191 个经典技术因子
- 完全集成项目标准框架
- 支持批量和单个因子计算
- 数据格式自动适配
- 标准化预处理流程
"""

from .alpha191_base import Alpha191Base, Alpha191DataAdapter
from .alpha191_calculator import Alpha191Calculator
from .alpha191_batch import Alpha191BatchCalculator

# 导入常用因子类（延迟导入避免循环依赖）
def get_alpha_factor(alpha_num: int):
    """
    获取指定编号的 Alpha 因子类
    
    Parameters
    ----------
    alpha_num : int
        因子编号 (1-191)
        
    Returns
    -------
    FactorBase
        对应的因子类实例
        
    Examples
    --------
    >>> alpha001 = get_alpha_factor(1)
    >>> result = alpha001.calculate(price_data)
    """
    from .alpha191_factors import ALPHA_FACTOR_REGISTRY
    
    if alpha_num not in ALPHA_FACTOR_REGISTRY:
        raise ValueError(f"Alpha{alpha_num:03d} 因子尚未实现")
    
    factor_class = ALPHA_FACTOR_REGISTRY[alpha_num]
    return factor_class()

# 批量获取因子
def get_alpha_factors(alpha_nums: list = None):
    """
    批量获取 Alpha 因子类
    
    Parameters
    ----------
    alpha_nums : list, optional
        因子编号列表，默认获取所有已实现因子
        
    Returns
    -------
    dict
        因子编号到因子实例的映射
    """
    from .alpha191_factors import ALPHA_FACTOR_REGISTRY
    
    if alpha_nums is None:
        alpha_nums = list(ALPHA_FACTOR_REGISTRY.keys())
    
    factors = {}
    for alpha_num in alpha_nums:
        if alpha_num in ALPHA_FACTOR_REGISTRY:
            factors[alpha_num] = ALPHA_FACTOR_REGISTRY[alpha_num]()
        else:
            print(f"警告: Alpha{alpha_num:03d} 因子尚未实现，跳过")
    
    return factors

__all__ = [
    # 核心类
    'Alpha191Base',
    'Alpha191DataAdapter',
    'Alpha191Calculator',
    'Alpha191BatchCalculator',
    
    # 便捷函数
    'get_alpha_factor',
    'get_alpha_factors',
]

# 版本信息
__version__ = '1.0.0'

# 模块元数据
TOTAL_FACTORS = 191
IMPLEMENTED_FACTORS = None  # 动态计算

def get_implementation_status():
    """获取因子实现状态"""
    from .alpha191_factors import ALPHA_FACTOR_REGISTRY
    
    implemented = len(ALPHA_FACTOR_REGISTRY)
    total = TOTAL_FACTORS
    
    return {
        'implemented': implemented,
        'total': total,
        'completion_rate': implemented / total,
        'missing': [i for i in range(1, total + 1) if i not in ALPHA_FACTOR_REGISTRY]
    }

# 设置实现数量
def _update_implemented_count():
    global IMPLEMENTED_FACTORS
    try:
        from .alpha191_factors import ALPHA_FACTOR_REGISTRY
        IMPLEMENTED_FACTORS = len(ALPHA_FACTOR_REGISTRY)
    except ImportError:
        IMPLEMENTED_FACTORS = 0

_update_implemented_count()