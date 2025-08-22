"""
因子模块 - 统一的因子生成、测试和分析框架

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
- base/        基础类和工具
- utils/       通用工具函数
"""

# 导入子模块
from . import generator, tester, analyzer, base, utils, calculator

# 从子模块导入主要接口
from .generator import (
    FactorGenerator,
    generate_factor,
    batch_generate_factors,
    list_available_factors
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

from .base import (
    FactorBase,
    MultiFactorBase,
    TimeSeriesProcessor,
    DataProcessingMixin
)

from .calculator import (
    FactorCalculator,
    FactorDataLoader
)

__all__ = [
    # 子模块
    'generator',
    'tester',
    'analyzer',
    'base',
    'utils',
    'calculator',
    
    # 生成接口
    'FactorGenerator',
    'generate_factor',
    'batch_generate_factors',
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
    
    # 基础类
    'FactorBase',
    'MultiFactorBase',
    'TimeSeriesProcessor',
    'DataProcessingMixin',
    
    # 工具类
    'FactorCalculator',
    'FactorDataLoader',
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