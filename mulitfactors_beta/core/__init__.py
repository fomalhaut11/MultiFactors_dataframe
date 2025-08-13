"""
核心模块统一入口
提供项目核心功能的便捷访问接口
"""

# 配置管理
from .config_manager import ConfigManager, get_config, get_path

# 数据库连接
from .database.connection_manager import DatabaseManager
from .database.sql_executor import SQLExecutor

# 单因子测试快捷入口
def test_single_factor(factor_name: str, **kwargs):
    """
    单因子测试快捷入口
    
    Parameters
    ----------
    factor_name : str
        因子名称
    **kwargs : dict
        测试配置参数
        - begin_date: 开始日期
        - end_date: 结束日期
        - group_nums: 分组数量
        - save_result: 是否保存结果
        - netral_base: 是否中性化
        - use_industry: 是否使用行业
        
    Returns
    -------
    TestResult
        测试结果对象
        
    Examples
    --------
    >>> from core import test_single_factor
    >>> result = test_single_factor('BP', begin_date='2024-01-01', end_date='2024-12-31')
    >>> print(f"IC: {result.ic_result.ic_mean:.4f}")
    """
    from factors.tester import SingleFactorTestPipeline
    
    pipeline = SingleFactorTestPipeline()
    return pipeline.run(factor_name, save_result=True, **kwargs)


def screen_factors(criteria=None, preset=None):
    """
    因子筛选快捷入口
    
    Parameters
    ----------
    criteria : dict, optional
        筛选标准
    preset : str, optional
        预设标准 ('strict', 'normal', 'loose')
        
    Returns
    -------
    list
        筛选出的因子列表
        
    Examples
    --------
    >>> from core import screen_factors
    >>> top_factors = screen_factors(preset='normal')
    >>> print(f"筛选出 {len(top_factors)} 个因子")
    """
    from factors.analyzer import FactorScreener
    
    screener = FactorScreener()
    screener.load_all_results()
    return screener.screen_factors(criteria=criteria, preset=preset)


def batch_test_factors(factor_list: list, **kwargs):
    """
    批量测试因子快捷入口
    
    Parameters
    ----------
    factor_list : list
        因子名称列表
    **kwargs : dict
        测试配置参数
        
    Returns
    -------
    dict
        因子名称到测试结果的映射
        
    Examples
    --------
    >>> from core import batch_test_factors
    >>> results = batch_test_factors(['BP', 'EP_ttm', 'ROE_ttm'])
    >>> for factor, result in results.items():
    ...     print(f"{factor}: IC={result.ic_result.ic_mean:.4f}")
    """
    from factors.tester import SingleFactorTestPipeline
    
    pipeline = SingleFactorTestPipeline()
    results = {}
    
    for factor_name in factor_list:
        try:
            print(f"测试因子: {factor_name}")
            result = pipeline.run(factor_name, save_result=True, **kwargs)
            results[factor_name] = result
        except Exception as e:
            print(f"测试 {factor_name} 失败: {e}")
            results[factor_name] = None
            
    return results


def load_factor_data(factor_name: str):
    """
    加载因子数据快捷入口
    
    Parameters
    ----------
    factor_name : str
        因子名称
        
    Returns
    -------
    pd.Series
        因子数据
        
    Examples
    --------
    >>> from core import load_factor_data
    >>> bp_factor = load_factor_data('BP')
    >>> print(f"因子形状: {bp_factor.shape}")
    """
    import pandas as pd
    from pathlib import Path
    
    factor_path = Path(get_path('raw_factors')) / f'{factor_name}.pkl'
    if factor_path.exists():
        return pd.read_pickle(factor_path)
    else:
        raise FileNotFoundError(f"因子文件不存在: {factor_path}")


def get_factor_test_result(factor_name: str):
    """
    获取因子最新测试结果快捷入口
    
    Parameters
    ----------
    factor_name : str
        因子名称
        
    Returns
    -------
    TestResult or None
        测试结果对象
        
    Examples
    --------
    >>> from core import get_factor_test_result
    >>> result = get_factor_test_result('BP')
    >>> if result:
    ...     print(f"IC: {result.ic_result.ic_mean:.4f}")
    """
    from factors.tester.core.result_manager import ResultManager
    from pathlib import Path
    import pickle
    
    manager = ResultManager()
    # 查找最新的测试结果
    test_path = Path(get_path('single_factor_test'))
    pattern = f"*/{factor_name}_*.pkl"
    files = list(test_path.glob(pattern))
    
    if files:
        # 按修改时间排序，获取最新的
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return manager.load(str(latest_file))
    
    return None


# 导出的函数和类
__all__ = [
    # 配置管理
    'ConfigManager',
    'get_config', 
    'get_path',
    # 数据库
    'DatabaseManager',
    'SQLExecutor',
    # 因子测试
    'test_single_factor',
    'screen_factors',
    'batch_test_factors',
    'load_factor_data',
    'get_factor_test_result',
]