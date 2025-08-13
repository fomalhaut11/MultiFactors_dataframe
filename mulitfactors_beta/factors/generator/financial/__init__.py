"""
财务因子模块

包含所有基于财务报表数据的因子计算功能

主要因子类别：
- 盈利能力因子（ROE、ROA、利润率等）
- 偿债能力因子（流动比率、负债率等）
- 营运效率因子（周转率、周转天数等）
- 成长能力因子（增长率、复合增长率等）
- 现金流因子（现金流比率、自由现金流等）
- 盈余惊喜因子（SUE、盈余修正等）
"""

from . import pure_financial_factors
from .pure_financial_factors import (
    PureFinancialFactorCalculator,
    calculate_pure_financial_factor,
    calculate_multiple_pure_financial_factors,
    calculate_financial_factors_by_category,
    get_all_pure_financial_factors,
    get_pure_financial_factor_categories
)

from .earnings_surprise_factors import (
    SUE,
    EarningsRevision,
    EarningsMomentum
)

# 导入格式适配器（支持MultiIndex Series）
try:
    from .financial_factors_adapter import (
        PureFinancialFactorCalculatorV2,
        create_financial_calculator
    )
except ImportError:
    PureFinancialFactorCalculatorV2 = None
    create_financial_calculator = None

__all__ = [
    # 主要类
    'PureFinancialFactorCalculator',
    'PureFinancialFactorCalculatorV2',  # MultiIndex版本
    'SUE',
    'EarningsRevision',
    'EarningsMomentum',
    
    # 便捷函数
    'calculate_pure_financial_factor', 
    'calculate_multiple_pure_financial_factors',
    'calculate_financial_factors_by_category',
    'get_all_pure_financial_factors',
    'get_pure_financial_factor_categories',
    'create_financial_calculator',  # 创建计算器的工厂函数
    
    # 模块引用
    'pure_financial_factors',
]

# 版本信息
__version__ = '1.0.0'


# ========== 便捷函数 ==========

def calculate_financial_factor(factor_name: str, data, **kwargs):
    """
    计算单个财务因子
    
    Parameters
    ----------
    factor_name : str
        因子名称
    data : pd.Series
        财务数据，MultiIndex格式[TradingDates, StockCodes]
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series
        计算结果，MultiIndex格式
        
    Examples
    --------
    >>> from factors.generator.financial import calculate_financial_factor
    >>> roe = calculate_financial_factor('ROE_ttm', financial_data)
    """
    # SUE类因子
    if factor_name == 'SUE':
        calculator = SUE()
        return calculator.calculate(data, **kwargs)
    elif factor_name == 'EarningsRevision':
        calculator = EarningsRevision()
        return calculator.calculate(data, **kwargs)
    elif factor_name == 'EarningsMomentum':
        calculator = EarningsMomentum()
        return calculator.calculate(data, **kwargs)
    else:
        # 纯财务因子
        return calculate_pure_financial_factor(factor_name, data, **kwargs)


def list_financial_factors():
    """
    列出所有可用的财务因子
    
    Returns
    -------
    dict
        按类别组织的因子列表
    """
    result = get_pure_financial_factor_categories()
    result['earnings_surprise'] = ['SUE', 'EarningsRevision', 'EarningsMomentum']
    return result