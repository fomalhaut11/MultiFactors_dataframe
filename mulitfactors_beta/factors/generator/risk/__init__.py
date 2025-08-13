"""
风险因子模块

计算各类风险相关的因子

主要因子类别：
- 市场风险因子（Beta、相关性等）
- 波动率风险因子（残差波动率、特质波动率等）
- 尾部风险因子（VaR、CVaR、偏度、峰度等）
- 系统性风险因子（市场敏感度、行业暴露等）
"""

from . import beta_factors

__all__ = [
    # 模块引用
    'beta_factors',
]

# 版本信息
__version__ = '1.0.0'


# ========== 便捷函数 ==========

def calculate_risk_factor(factor_name: str, data, market_data=None, **kwargs):
    """
    计算单个风险因子
    
    Parameters
    ----------
    factor_name : str
        因子名称
    data : pd.DataFrame
        股票收益率数据
    market_data : pd.DataFrame, optional
        市场收益率数据（计算Beta等需要）
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series or pd.DataFrame
        计算结果
        
    Examples
    --------
    >>> from factors.generator.risk import calculate_risk_factor
    >>> beta = calculate_risk_factor('Beta', stock_returns, market_returns)
    """
    if factor_name == 'Beta':
        # 使用BetaFactor类计算
        from .beta_factors import BetaFactor
        beta_calculator = BetaFactor()
        return beta_calculator.calculate(data, market_data, **kwargs)
    else:
        raise NotImplementedError(f"风险因子 {factor_name} 尚未实现")


def list_risk_factors():
    """
    列出所有可用的风险因子
    
    Returns
    -------
    dict
        按类别组织的因子列表
    """
    return {
        'market_risk': ['Beta', 'Correlation', 'MarketSensitivity'],
        'volatility_risk': ['ResidualVolatility', 'IdiosyncraticVolatility', 'DownsideVolatility'],
        'tail_risk': ['VaR', 'CVaR', 'Skewness', 'Kurtosis'],
        'systematic_risk': ['IndustryExposure', 'FactorExposure']
    }