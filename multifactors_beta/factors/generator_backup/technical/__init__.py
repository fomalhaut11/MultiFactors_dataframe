"""
技术因子模块

基于价格、成交量等市场数据计算的技术因子

主要因子类别：
- 价格因子（动量、反转、趋势等）
- 波动率因子（历史波动率、GARCH等）
- 技术指标（MACD、RSI、振荡器等）
- 成交量因子（换手率、量价关系等）
"""

from . import price_factors, volatility_factors, oscillator_factors

__all__ = [
    # 模块引用
    'price_factors',
    'volatility_factors', 
    'oscillator_factors',
]

# 版本信息
__version__ = '2.0.0'


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
    # 动量因子
    if 'Momentum' in factor_name:
        from .price_factors import MomentumFactor
        if 'Momentum_20d' in factor_name:
            factor = MomentumFactor(window=20)
        else:
            # 解析窗口参数
            import re
            match = re.search(r'Momentum_(\d+)d', factor_name)
            window = int(match.group(1)) if match else 20
            factor = MomentumFactor(window=window)
        return factor.calculate(data, **kwargs)
    
    # RSI因子
    elif 'RSI' in factor_name:
        from .oscillator_factors import RSIFactor
        import re
        match = re.search(r'RSI_(\d+)', factor_name)
        window = int(match.group(1)) if match else 14
        factor = RSIFactor(window=window)
        return factor.calculate(data, **kwargs)
    
    # MACD因子
    elif 'MACD' in factor_name:
        from .oscillator_factors import MACDFactor
        factor = MACDFactor()  # 使用默认参数
        return factor.calculate(data, **kwargs)
    
    # 波动率因子
    elif 'Vol' in factor_name or 'Volatility' in factor_name:
        from .volatility_factors import HistoricalVolatilityFactor
        factor = HistoricalVolatilityFactor(window=20)
        return factor.calculate(data, **kwargs)
    
    else:
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
        'price': [
            'Momentum_5d', 'Momentum_10d', 'Momentum_20d', 'Momentum_60d', 'Momentum_120d',
            'Reversal_5d', 'MovingAverage', 'GapReturn', 'PricePosition'
        ],
        'volatility': [
            'HistVol_20d_simple', 'HistVol_20d_parkinson', 'HistVol_20d_garman_klass', 
            'RealizedVol_20d', 'DownsideVol_20d', 'GARCH_252d', 'VolRatio_5_20',
            'VolSkew_60d', 'VolKurt_60d'
        ],
        'oscillators': [
            'MACD_12_26_9_macd', 'MACD_12_26_9_signal', 'MACD_12_26_9_histogram',
            'RSI_6', 'RSI_14', 'RSI_21',
            'WilliamsR_14', 'WilliamsR_21',
            'CCI_14', 'CCI_20',
            'StochK_14_3', 'StochD_14_3',
            'ROC_12', 'ROC_20',
            'UltimateOsc_7_14_28'
        ],
        'volume': ['Turnover', 'VWAP', 'VolumeRatio']  # 待实现
    }


def get_factor_batch_generators():
    """
    获取批量因子生成器
    
    Returns
    -------
    dict
        生成器类映射
    """
    from .price_factors import MultiPeriodMomentumFactory
    from .volatility_factors import MultiVolatilityFactory
    from .oscillator_factors import MultiOscillatorFactory
    
    return {
        'momentum': MultiPeriodMomentumFactory,
        'volatility': MultiVolatilityFactory,
        'oscillator': MultiOscillatorFactory
    }


def create_technical_factor_suite(price_data, factor_categories=None):
    """
    创建完整的技术因子套件
    
    Parameters
    ----------
    price_data : pd.DataFrame
        价格数据
    factor_categories : list, optional
        要生成的因子类别列表
        
    Returns
    -------
    dict
        因子名称到因子值的映射
    """
    if factor_categories is None:
        factor_categories = ['momentum', 'volatility', 'oscillator']
    
    generators = get_factor_batch_generators()
    all_factors = {}
    
    # 动量因子
    if 'momentum' in factor_categories:
        momentum_factory = generators['momentum']([5, 10, 20, 60])
        momentum_factors = momentum_factory.generate_momentum_factors(
            price_data, factor_type='standard'
        )
        all_factors.update(momentum_factors)
    
    # 波动率因子
    if 'volatility' in factor_categories:
        volatility_factory = generators['volatility']([5, 10, 20, 60])
        volatility_factors = volatility_factory.generate_volatility_factors(
            price_data, factor_types=['historical', 'realized']
        )
        all_factors.update(volatility_factors)
    
    # 振荡器因子
    if 'oscillator' in factor_categories:
        oscillator_factory = generators['oscillator']()
        oscillator_factors = oscillator_factory.generate_oscillator_factors(
            price_data, factor_types=['MACD', 'RSI', 'WilliamsR']
        )
        all_factors.update(oscillator_factors)
    
    return all_factors