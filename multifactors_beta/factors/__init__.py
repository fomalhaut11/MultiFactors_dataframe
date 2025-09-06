"""
因子模块 - 统一的因子生成、测试和分析框架

⚠️  重要使用约束 ⚠️
====================
此模块必须在项目根目录下导入和使用，不能在factors子目录下运行！

正确使用方式：
    # 在 multifactors_beta/ 目录下
    import factors
    result = factors.generate('ROE_ttm', data)

错误使用方式：
    # 在 multifactors_beta/factors/ 目录下
    import factors  # 会导致配置导入错误！

提供完整的因子研究工作流：
1. 因子生成（generator）- 生成各类因子
2. 因子测试（tester）- 测试因子有效性
3. 因子分析（analyzer）- 分析和筛选因子

模块结构：
=========
- generator/   因子生成模块
  - financial/ 财务因子
  - technical/ 技术因子
  - risk/      风险因子
- tester/      因子测试模块
- analyzer/    因子分析模块
- operations/  因子操作工具箱（新增）
  - cross_sectional.py  截面操作
  - time_series.py       时序操作
  - combination.py       因子组合
- base/        基础类和工具
- utils/       通用工具函数
"""

# 导入子模块（移除已删除的generator）
from . import tester, analyzer, base, utils, operations, combiner
from . import generators, library

# 从generators导入基础数据处理工具
from .generators.financial import (
    calculate_ttm,
    calculate_single_quarter,
    calculate_yoy,
    calculate_qoq,
    calculate_zscore,
    expand_to_daily_vectorized
)

from .generators.technical import (
    MovingAverageCalculator,
    TechnicalIndicators,
    VolatilityCalculator
)

from .generators.alpha191 import (
    ts_rank,
    ts_mean,
    delta,
    rank,
    scale
)

# 从library导入因子计算接口
from .library import (
    get_factor,
    calculate_factor,
    batch_calculate_factors,
    list_factors,
    search_factors,
    get_factor_info,
    get_factor_summary
)

from .tester import (
    SingleFactorTestPipeline,
    TestResult,
    test_factor,
    batch_test,
    load_test_result
)

from .analyzer import (
    FactorScreener,
    get_analyzer_config
)

from .combiner import (
    CombinerBase,
    FactorCombiner
)

from .base import (
    FactorBase,
    MultiFactorBase,
    DataProcessingMixin
)


__all__ = [
    # 子模块
    'generators',
    'library',
    'tester',
    'analyzer',
    'combiner',
    'base',
    'utils',
    'operations',
    
    # 基础数据处理工具
    'calculate_ttm',
    'calculate_single_quarter',
    'calculate_yoy',
    'calculate_qoq',
    'calculate_zscore',
    'expand_to_daily_vectorized',
    'MovingAverageCalculator',
    'TechnicalIndicators',
    'VolatilityCalculator',
    'ts_rank',
    'ts_mean',
    'delta',
    'rank',
    'scale',
    
    # 因子计算接口
    'get_factor',
    'calculate_factor',
    'batch_calculate_factors',
    'list_factors',
    'search_factors',
    'get_factor_info',
    'get_factor_summary',
    'list_available_factors',
    
    # 测试接口
    'SingleFactorTestPipeline',
    'TestResult',
    'test_factor',
    'batch_test',
    'load_test_result',
    
    # 分析接口
    'FactorScreener',
    'get_analyzer_config',
    
    # 因子组合接口
    'CombinerBase',
    'FactorCombiner',
    
    # 基础类
    'FactorBase',
    'MultiFactorBase',
    'FinancialReportProcessor',  # 财务报表数据处理器
    'TimeSeriesProcessor',       # 向后兼容别名
    'DataProcessingMixin',
    
]

# 版本信息
__version__ = '2.0.0'  # 重构后的新版本


# ========== 统一便捷函数 ==========

def generate(factor_name: str, data, **kwargs):
    """
    生成因子的统一入口
    
    自动识别因子类型并调用相应的生成器
    
    Parameters
    ----------
    factor_name : str
        因子名称
    data : pd.DataFrame
        输入数据
    **kwargs
        其他参数
        
    Returns
    -------
    pd.DataFrame
        生成的因子
        
    Examples
    --------
    >>> from factors import generate
    >>> roe = generate('ROE_ttm', financial_data)
    >>> beta = generate('Beta', price_data)
    """
    from .generator import generate_factor
    return generate_factor(factor_name, data, **kwargs)


def test(factor_name: str, **kwargs):
    """
    测试因子的统一入口
    
    Parameters
    ----------
    factor_name : str
        因子名称
    **kwargs
        测试参数
        
    Returns
    -------
    TestResult
        测试结果
        
    Examples
    --------
    >>> from factors import test
    >>> result = test('ROE_ttm')
    >>> print(f"IC: {result.ic_result.ic_mean:.4f}")
    """
    from .tester import test_factor
    return test_factor(factor_name, **kwargs)


def analyze(factor_names: list = None, preset: str = 'normal', **kwargs):
    """
    分析因子的统一入口
    
    Parameters
    ----------
    factor_names : list, optional
        要分析的因子列表，如果不指定则分析所有已测试的因子
    preset : str
        筛选预设（'loose', 'normal', 'strict'）
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        分析结果
        
    Examples
    --------
    >>> from factors import analyze
    >>> # 分析所有因子
    >>> results = analyze(preset='strict')
    >>> # 分析指定因子
    >>> results = analyze(['ROE_ttm', 'Beta'])
    """
    from .analyzer import FactorScreener
    screener = FactorScreener()
    
    if factor_names:
        # 分析指定因子
        return screener.analyze_factors(factor_names, **kwargs)
    else:
        # 筛选因子
        return screener.screen_factors(preset=preset, **kwargs)


def pipeline(factor_name: str, data, test: bool = True, analyze: bool = False, **kwargs):
    """
    完整的因子研究流水线
    
    生成 -> 测试 -> 分析
    
    Parameters
    ----------
    factor_name : str
        因子名称
    data : pd.DataFrame
        输入数据
    test : bool
        是否进行测试
    analyze : bool
        是否进行分析
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        包含生成、测试、分析结果的字典
        
    Examples
    --------
    >>> from factors import pipeline
    >>> results = pipeline('ROE_ttm', financial_data, test=True, analyze=True)
    >>> print(results['test'].ic_result.ic_mean)
    """
    results = {}
    
    # 生成因子
    factor_data = generate(factor_name, data, **kwargs)
    results['factor'] = factor_data
    
    # 测试因子
    if test:
        test_result = test(factor_name, **kwargs)
        results['test'] = test_result
        
    # 分析因子
    if analyze:
        analysis_result = analyze([factor_name], **kwargs)
        results['analysis'] = analysis_result
        
    return results