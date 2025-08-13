"""
单因子测试模块

提供完整的单因子测试功能，包括：
- 回归分析测试
- 分组测试
- IC分析
- 批量测试
"""

from .core.pipeline import SingleFactorTestPipeline
from .base.test_result import TestResult
from .core.data_manager import DataManager
from .core.factor_tester import FactorTester
from .core.result_manager import ResultManager

__all__ = [
    # 主要接口
    'SingleFactorTestPipeline',
    'TestResult',
    
    # 管理器
    'DataManager',
    'FactorTester',
    'ResultManager',
    
    # 便捷函数
    'test_factor',
    'batch_test',
    'load_test_result',
]

# 版本信息
__version__ = '1.0.0'

# ========== 便捷函数 ==========

def test_factor(factor_name: str, **kwargs):
    """
    快速测试单个因子
    
    Parameters
    ----------
    factor_name : str
        因子名称
    **kwargs : dict
        测试参数
        
    Returns
    -------
    TestResult
        测试结果
        
    Examples
    --------
    >>> from factors.tester import test_factor
    >>> result = test_factor('BP', begin_date='2024-01-01', end_date='2024-12-31')
    >>> print(f"IC: {result.ic_result.ic_mean:.4f}")
    """
    pipeline = SingleFactorTestPipeline()
    return pipeline.run(factor_name, save_result=True, **kwargs)


def batch_test(factor_list: list, **kwargs):
    """
    批量测试多个因子
    
    Parameters
    ----------
    factor_list : list
        因子名称列表
    **kwargs : dict
        测试参数
        
    Returns
    -------
    dict
        因子名称到测试结果的映射
        
    Examples
    --------
    >>> from factors.tester import batch_test
    >>> results = batch_test(['BP', 'EP_ttm', 'ROE_ttm'])
    >>> for factor, result in results.items():
    ...     print(f"{factor}: IC={result.ic_result.ic_mean:.4f}")
    """
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


def load_test_result(factor_name: str, test_id: str = None):
    """
    加载因子测试结果
    
    Parameters
    ----------
    factor_name : str
        因子名称
    test_id : str, optional
        测试ID，如果不指定则加载最新的
        
    Returns
    -------
    TestResult or None
        测试结果
        
    Examples
    --------
    >>> from factors.tester import load_test_result
    >>> result = load_test_result('BP')
    >>> if result:
    ...     print(f"IC: {result.ic_result.ic_mean:.4f}")
    """
    from pathlib import Path
    from ..core.config_manager import get_path
    
    manager = ResultManager()
    test_path = Path(get_path('single_factor_test'))
    
    if test_id:
        # 加载指定ID的结果
        pattern = f"*/{factor_name}_*_{test_id}.pkl"
    else:
        # 加载最新的结果
        pattern = f"*/{factor_name}_*.pkl"
    
    files = list(test_path.glob(pattern))
    
    if files:
        # 按修改时间排序，获取最新的
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return manager.load(str(latest_file))
    
    return None