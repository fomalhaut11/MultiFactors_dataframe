"""
财务因子计算工具集成模块

整合所有财务相关的计算工具函数
"""

# 导入所有工具模块
from .profitability_tools import PROFITABILITY_TOOLS
from .value_tools import VALUE_TOOLS  
from .quality_tools import QUALITY_TOOLS

# 导入具体函数以便直接使用
from .profitability_tools import (
    calculate_roe_ttm,
    calculate_roa_ttm,
    calculate_gross_profit_margin_ttm,
    calculate_profit_cost_ttm,
    calculate_operating_profit_margin_ttm,
)

from .value_tools import (
    calculate_ep_ttm,
    calculate_bp_ttm,
    calculate_sp_ttm,
    calculate_cfp_ttm,
    calculate_dp_ttm,
)

from .quality_tools import (
    calculate_accrual_ratio_ttm,
    calculate_cash_flow_coverage_ttm,
    calculate_earning_persistence_ttm,
    calculate_asset_turnover_ttm,
    calculate_receivables_turnover_ttm,
)

# 整合所有工具函数注册表
FINANCIAL_TOOLS_REGISTRY = {}
FINANCIAL_TOOLS_REGISTRY.update(PROFITABILITY_TOOLS)
FINANCIAL_TOOLS_REGISTRY.update(VALUE_TOOLS)
FINANCIAL_TOOLS_REGISTRY.update(QUALITY_TOOLS)

# 便捷函数：根据名称获取计算函数
def get_financial_calculator(factor_name: str):
    """
    根据因子名称获取对应的计算函数
    
    Parameters
    ----------
    factor_name : str
        因子名称
        
    Returns
    -------
    callable
        对应的计算函数
        
    Examples
    --------
    >>> calc_func = get_financial_calculator('ROE_ttm')
    >>> result = calc_func(financial_data)
    """
    if factor_name in FINANCIAL_TOOLS_REGISTRY:
        return FINANCIAL_TOOLS_REGISTRY[factor_name]
    else:
        available_factors = list(FINANCIAL_TOOLS_REGISTRY.keys())
        raise ValueError(f"未找到因子 {factor_name}，可用因子: {available_factors}")

# 列出所有可用的财务因子
def list_financial_factors():
    """
    列出所有可用的财务因子
    
    Returns
    -------
    dict
        按类别组织的因子列表
    """
    return {
        'profitability': list(PROFITABILITY_TOOLS.keys()),
        'value': list(VALUE_TOOLS.keys()),
        'quality': list(QUALITY_TOOLS.keys()),
    }

# 导出清单
__all__ = [
    # 注册表
    'FINANCIAL_TOOLS_REGISTRY',
    'PROFITABILITY_TOOLS',
    'VALUE_TOOLS', 
    'QUALITY_TOOLS',
    
    # 便捷函数
    'get_financial_calculator',
    'list_financial_factors',
    
    # 盈利能力工具
    'calculate_roe_ttm',
    'calculate_roa_ttm',
    'calculate_gross_profit_margin_ttm', 
    'calculate_profit_cost_ttm',
    'calculate_operating_profit_margin_ttm',
    
    # 估值工具
    'calculate_ep_ttm',
    'calculate_bp_ttm',
    'calculate_sp_ttm',
    'calculate_cfp_ttm',
    'calculate_dp_ttm',
    
    # 质量工具
    'calculate_accrual_ratio_ttm',
    'calculate_cash_flow_coverage_ttm',
    'calculate_earning_persistence_ttm',
    'calculate_asset_turnover_ttm',
    'calculate_receivables_turnover_ttm',
]