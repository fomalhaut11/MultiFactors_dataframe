"""
因子生成器模块

包含各类因子的生成器和计算逻辑

主要模块：
- financial: 纯财务因子（仅依赖财务报表数据）
- mixed: 混合因子（需要多种数据源）
  - valuation: 估值因子（财务+市值）
  - 未来扩展: size, liquidity, quality 等
- technical: 技术因子（基于价格、成交量）
- risk: 风险因子（Beta、波动率等）
- alpha191: Alpha191 因子集（191个技术因子）
"""

from .factor_generator import (
    FactorGenerator,
    FinancialFactorGenerator,
    TechnicalFactorGenerator,
    RiskFactorGenerator,
    create_generator
)

# 导入各类因子
from .financial import (
    SUE,
    EarningsRevision,
    EarningsMomentum
)

from .mixed import (
    MixedFactorManager,
    ValuationFactorCalculator,
    get_mixed_factor_manager,
    calculate_mixed_factor,
    get_all_mixed_factors
)

__all__ = [
    # 生成器基类
    'FactorGenerator',
    'FinancialFactorGenerator',
    'TechnicalFactorGenerator',
    'RiskFactorGenerator',
    'create_generator',
    
    # 财务因子
    'SUE',
    'EarningsRevision',
    'EarningsMomentum',
    
    # 混合因子
    'MixedFactorManager',
    'ValuationFactorCalculator',
    'get_mixed_factor_manager',
    'calculate_mixed_factor',
    'get_all_mixed_factors',
]

# 版本信息
__version__ = '1.0.0'


# ========== 便捷函数 ==========

def generate_factor(factor_name: str, 
                   data,
                   factor_type: str = None,
                   **kwargs):
    """
    快速生成单个因子
    
    Parameters
    ----------
    factor_name : str
        因子名称
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    factor_type : str, optional
        因子类型，如果不指定会自动推断
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series
        生成的因子数据，MultiIndex格式
        
    Examples
    --------
    >>> from factors.generator import generate_factor
    >>> roe = generate_factor('ROE_ttm', financial_data)
    >>> beta = generate_factor('Beta', price_data, factor_type='risk')
    """
    # 自动推断因子类型
    if factor_type is None:
        # 根据因子名称推断类型
        financial_keywords = ['ROE', 'ROA', 'PE', 'PB', 'SUE', 'Earnings']
        technical_keywords = ['MA', 'RSI', 'MACD', 'Momentum', 'Volatility']
        risk_keywords = ['Beta', 'VaR', 'CVaR', 'Residual']
        
        factor_name_upper = factor_name.upper()
        if any(kw in factor_name_upper for kw in financial_keywords):
            factor_type = 'financial'
        elif any(kw in factor_name_upper for kw in technical_keywords):
            factor_type = 'technical'
        elif any(kw in factor_name_upper for kw in risk_keywords):
            factor_type = 'risk'
        else:
            raise ValueError(f"无法推断因子 {factor_name} 的类型，请指定factor_type参数")
    
    # 创建生成器并生成因子
    generator = create_generator(factor_type)
    return generator.generate(factor_name, data, **kwargs)


def batch_generate_factors(factor_names: list,
                          data,
                          factor_type: str = None,
                          **kwargs):
    """
    批量生成多个因子
    
    Parameters
    ----------
    factor_names : list
        因子名称列表
    data : pd.Series
        输入数据，MultiIndex格式[TradingDates, StockCodes]
    factor_type : str, optional
        因子类型
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        因子名称到因子数据的映射
        
    Examples
    --------
    >>> from factors.generator import batch_generate_factors
    >>> factors = batch_generate_factors(
    ...     ['ROE_ttm', 'ROA_ttm', 'CurrentRatio'],
    ...     financial_data,
    ...     factor_type='financial'
    ... )
    """
    results = {}
    for factor_name in factor_names:
        try:
            results[factor_name] = generate_factor(
                factor_name, data, factor_type, **kwargs
            )
        except Exception as e:
            print(f"生成因子 {factor_name} 失败: {e}")
            results[factor_name] = None
    return results


def list_available_factors(factor_type: str = None):
    """
    列出可用的因子
    
    Parameters
    ----------
    factor_type : str, optional
        因子类型，如果不指定则列出所有类型
        
    Returns
    -------
    dict or list
        如果指定类型返回列表，否则返回字典
        
    Examples
    --------
    >>> from factors.generator import list_available_factors
    >>> # 列出所有因子
    >>> all_factors = list_available_factors()
    >>> # 列出财务因子
    >>> financial_factors = list_available_factors('financial')
    """
    if factor_type:
        generator = create_generator(factor_type)
        return generator.get_available_factors()
    else:
        result = {}
        for ft in ['financial', 'technical', 'risk']:
            generator = create_generator(ft)
            result[ft] = generator.get_available_factors()
        return result