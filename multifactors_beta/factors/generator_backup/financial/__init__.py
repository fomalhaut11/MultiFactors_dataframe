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

# 导入新的因子实现
from .profitability_factors import (
    ROE_ttm_Factor, ROA_ttm_Factor, GrossProfitMargin_ttm_Factor, ProfitCost_ttm_Factor
)
from .solvency_factors import (
    CurrentRatio_Factor, DebtToAssets_Factor
)
from .legacy_financial_factors import (
    SUE_ttm_120d_Factor
)
from .value_factors import (
    EPRatioFactor, BPRatioFactor, SPRatioFactor, CFPRatioFactor, EarningsYieldFactor
)
from .quality_factors import (
    ROEQualityFactor, EarningsQualityFactor, DebtQualityFactor,
    ProfitabilityStabilityFactor, AssetQualityFactor
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
    # 盈利能力因子
    'ROE_ttm_Factor', 'ROA_ttm_Factor', 'GrossProfitMargin_ttm_Factor', 'ProfitCost_ttm_Factor',
    
    # 偿债能力因子
    'CurrentRatio_Factor', 'DebtToAssets_Factor',
    
    # 遗留财务因子
    'SUE_ttm_120d_Factor',
    
    # 价值因子
    'EPRatioFactor', 'BPRatioFactor', 'SPRatioFactor', 'CFPRatioFactor', 'EarningsYieldFactor',
    
    # 质量因子
    'ROEQualityFactor', 'EarningsQualityFactor', 'DebtQualityFactor',
    'ProfitabilityStabilityFactor', 'AssetQualityFactor',
    
    # 盈余惊喜因子
    'SUE', 'EarningsRevision', 'EarningsMomentum',
    
    # 便捷函数
    'calculate_financial_factor',
    'list_financial_factors',
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
    data : pd.DataFrame
        财务数据，包含所需的财务字段
    **kwargs
        其他参数，如market_cap等
        
    Returns
    -------
    pd.Series
        计算结果，MultiIndex格式[TradingDates, StockCodes]
        
    Examples
    --------
    >>> from factors.generator.financial import calculate_financial_factor
    >>> roe = calculate_financial_factor('ROE_ttm', financial_data)
    """
    # 创建因子映射
    factor_map = {
        'ROE_ttm': ROE_ttm_Factor,
        'ROA_ttm': ROA_ttm_Factor, 
        'GrossProfitMargin_ttm': GrossProfitMargin_ttm_Factor,
        'ProfitCost_ttm': ProfitCost_ttm_Factor,
        'CurrentRatio': CurrentRatio_Factor,
        'DebtToAssets': DebtToAssets_Factor,
        'SUE_ttm_120d': SUE_ttm_120d_Factor,
        'EP_Ratio': EPRatioFactor,
        'BP_Ratio': BPRatioFactor,
        'SP_Ratio': SPRatioFactor,
        'CFP_Ratio': CFPRatioFactor,
        'EarningsYield': EarningsYieldFactor,
        'ROEQuality': ROEQualityFactor,
        'EarningsQuality': EarningsQualityFactor,
        'DebtQuality': DebtQualityFactor,
        'ProfitabilityStability': ProfitabilityStabilityFactor,
        'AssetQuality': AssetQualityFactor,
    }
    
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
    elif factor_name in factor_map:
        calculator = factor_map[factor_name]()
        return calculator.calculate(data, **kwargs)
    else:
        raise ValueError(f"未知的财务因子: {factor_name}")


def list_financial_factors():
    """
    列出所有可用的财务因子
    
    Returns
    -------
    dict
        按类别组织的因子列表
    """
    return {
        'profitability': ['ROE_ttm', 'ROA_ttm', 'GrossProfitMargin_ttm', 'ProfitCost_ttm'],
        'solvency': ['CurrentRatio', 'DebtToAssets'],
        'value': ['EP_Ratio', 'BP_Ratio', 'SP_Ratio', 'CFP_Ratio', 'EarningsYield'],
        'quality': ['ROEQuality', 'EarningsQuality', 'DebtQuality', 'ProfitabilityStability', 'AssetQuality'],
        'earnings_surprise': ['SUE_ttm_120d', 'SUE', 'EarningsRevision', 'EarningsMomentum']
    }