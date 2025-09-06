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
    'get_test_summary',
    'compare_factors',
    
    # 配置管理
    'TestConfig',
    'set_default_config',
    'get_default_config',
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
    from config import get_config
    
    manager = ResultManager()
    test_path = Path(get_config('main.paths.single_factor_test'))
    
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


def get_test_summary(factor_names: list, **kwargs):
    """
    获取多个因子的测试摘要
    
    Parameters
    ----------
    factor_names : list
        因子名称列表
    **kwargs : dict
        其他参数
        
    Returns
    -------
    pd.DataFrame
        测试摘要表格
        
    Examples
    --------
    >>> from factors.tester import get_test_summary
    >>> summary = get_test_summary(['BP', 'EP_ttm', 'ROE_ttm'])
    >>> print(summary[['ic_mean', 'icir', 'sharpe']])
    """
    import pandas as pd
    
    summaries = []
    for factor_name in factor_names:
        result = load_test_result(factor_name)
        if result:
            summary = {
                'factor': factor_name,
                'ic_mean': result.performance_metrics.get('ic_mean', 0),
                'icir': result.performance_metrics.get('icir', 0),
                'rank_ic': result.performance_metrics.get('rank_ic_mean', 0),
                'sharpe': result.performance_metrics.get('long_short_sharpe', 0),
                'turnover': result.performance_metrics.get('avg_turnover', 0),
                'cost': result.performance_metrics.get('avg_turnover_cost', 0)
            }
            summaries.append(summary)
    
    return pd.DataFrame(summaries)


def compare_factors(factor_results: dict, metrics: list = None):
    """
    比较多个因子的测试结果
    
    Parameters
    ----------
    factor_results : dict
        因子名称到TestResult的映射
    metrics : list, optional
        要比较的指标列表
        
    Returns
    -------
    pd.DataFrame
        比较结果表格
        
    Examples
    --------
    >>> from factors.tester import batch_test, compare_factors
    >>> results = batch_test(['BP', 'EP_ttm'])
    >>> comparison = compare_factors(results)
    >>> print(comparison)
    """
    import pandas as pd
    
    if metrics is None:
        metrics = ['ic_mean', 'icir', 'rank_ic_mean', 'long_short_sharpe', 
                  'monotonicity_score', 'avg_turnover', 'avg_turnover_cost']
    
    comparison_data = []
    for factor_name, result in factor_results.items():
        if result is None:
            continue
            
        row = {'factor': factor_name}
        for metric in metrics:
            if metric in result.performance_metrics:
                row[metric] = result.performance_metrics[metric]
            else:
                row[metric] = None
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.set_index('factor')
    
    # 按ICIR排序
    if 'icir' in df.columns:
        df = df.sort_values('icir', ascending=False)
    
    return df


# ========== 配置管理 ==========

class TestConfig:
    """测试配置类"""
    
    DEFAULT_CONFIG = {
        'group_nums': 10,
        'outlier_method': 'IQR',
        'outlier_param': 5,
        'normalization_method': 'zscore',
        'backtest_type': 'all',
        'ic_decay_periods': 20,
        'turnover_cost_rate': 0.0015,
    }
    
    _current_config = DEFAULT_CONFIG.copy()
    
    @classmethod
    def get(cls, key: str = None):
        """获取配置"""
        if key:
            return cls._current_config.get(key)
        return cls._current_config.copy()
    
    @classmethod
    def set(cls, key: str, value):
        """设置配置"""
        cls._current_config[key] = value
    
    @classmethod
    def update(cls, config: dict):
        """批量更新配置"""
        cls._current_config.update(config)
    
    @classmethod
    def reset(cls):
        """重置为默认配置"""
        cls._current_config = cls.DEFAULT_CONFIG.copy()


def set_default_config(config: dict):
    """
    设置默认测试配置
    
    Parameters
    ----------
    config : dict
        配置字典
        
    Examples
    --------
    >>> from factors.tester import set_default_config
    >>> set_default_config({'group_nums': 5, 'outlier_method': 'MAD'})
    """
    TestConfig.update(config)


def get_default_config():
    """
    获取默认测试配置
    
    Returns
    -------
    dict
        当前配置
        
    Examples
    --------
    >>> from factors.tester import get_default_config
    >>> config = get_default_config()
    >>> print(config['group_nums'])
    """
    return TestConfig.get()