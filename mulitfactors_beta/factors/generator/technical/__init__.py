"""
技术因子模块

基于价格、成交量等市场数据计算的技术因子

主要因子类别：
- 价格因子（动量、反转、趋势等）
- 波动率因子（历史波动率、GARCH等）
- 技术指标（MA、RSI、MACD等）
- 成交量因子（换手率、量价关系等）
"""

from . import price_factors, volatility_factors

__all__ = [
    # 模块引用
    'price_factors',
    'volatility_factors',
]

# 版本信息
__version__ = '1.0.0'


# ========== 便捷函数 ==========

def calculate_technical_factor(factor_name: str, data, **kwargs):
    """
    计算单个技术因子
    
    Parameters
    ----------
    factor_name : str
        因子名称
    data : pd.DataFrame
        市场数据（价格、成交量等）
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series or pd.DataFrame
        计算结果
        
    Examples
    --------
    >>> from factors.generator.technical import calculate_technical_factor
    >>> momentum = calculate_technical_factor('Momentum_20d', price_data)
    """
    # TODO: 实现具体的技术因子计算逻辑
    raise NotImplementedError(f"技术因子 {factor_name} 尚未实现")


def list_technical_factors():
    """
    列出所有可用的技术因子
    
    Returns
    -------
    dict
        按类别组织的因子列表
    """
    return {
        'price': ['Momentum', 'Reversal', 'MA', 'EMA'],
        'volatility': ['HistoricalVolatility', 'GARCH', 'RealizedVolatility'],
        'technical_indicators': ['RSI', 'MACD', 'Bollinger'],
        'volume': ['Turnover', 'VWAP', 'VolumeRatio']
    }