"""
因子库模块

基于装饰器的因子注册系统，提供标准化的因子接口和元数据管理

主要功能：
- 因子注册和管理
- 标准化的因子接口
- 依赖检查和验证
- 元数据和搜索功能

使用方式：
```python
# 导入注册的因子
from factors.library import get_factor, list_factors

# 获取因子计算函数
roe_calc = get_factor('ROE_ttm')
result = roe_calc(financial_data)

# 列出可用因子
factors = list_factors()
print(factors['profitability'])  # ['ROE_ttm', 'ROA_ttm', ...]
```
"""

# 导入核心注册系统
from .factor_registry import (
    FactorRegistry,
    factor_registry,
    register_factor,
    get_factor,
    list_factors,
    search_factors,
    get_factor_info
)

# 导入注册的因子模块（这会触发因子注册）
from . import financial_factors

# 便捷函数
def calculate_factor(name: str, data, **kwargs):
    """
    计算单个因子
    
    Parameters
    ----------
    name : str
        因子名称
    data : pd.DataFrame
        输入数据
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series
        因子计算结果
        
    Examples
    --------
    >>> from factors.library import calculate_factor
    >>> roe = calculate_factor('ROE_ttm', financial_data)
    """
    factor_func = get_factor(name)
    if factor_func is None:
        raise ValueError(f"因子 {name} 未注册")
    
    return factor_func(data, **kwargs)


def batch_calculate_factors(factor_names: list, data, **kwargs):
    """
    批量计算多个因子
    
    Parameters
    ----------
    factor_names : list
        因子名称列表
    data : pd.DataFrame
        输入数据
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        因子名称到计算结果的映射
        
    Examples
    --------
    >>> from factors.library import batch_calculate_factors
    >>> results = batch_calculate_factors(['ROE_ttm', 'ROA_ttm'], financial_data)
    """
    results = {}
    for name in factor_names:
        try:
            results[name] = calculate_factor(name, data, **kwargs)
        except Exception as e:
            print(f"计算因子 {name} 失败: {e}")
            results[name] = None
    
    return results


def get_factor_dependencies(name: str) -> list:
    """
    获取因子的数据依赖
    
    Parameters
    ----------
    name : str
        因子名称
        
    Returns
    -------
    list
        依赖的数据字段列表
    """
    metadata = get_factor_info(name)
    return metadata.get('dependencies', [])


def validate_data_for_factor(name: str, data) -> tuple:
    """
    验证数据是否满足因子计算要求
    
    Parameters
    ----------
    name : str
        因子名称
    data : pd.DataFrame
        输入数据
        
    Returns
    -------
    tuple
        (is_valid, missing_fields)
    """
    return factor_registry.validate_dependencies(name, data)


def get_factors_by_category(category: str) -> list:
    """
    获取指定类别的所有因子
    
    Parameters
    ----------
    category : str
        因子类别
        
    Returns
    -------
    list
        因子名称列表
    """
    return list_factors(category)


def get_registered_count() -> int:
    """获取已注册的因子数量"""
    return len(factor_registry)


def get_factor_summary() -> dict:
    """
    获取因子库概览信息
    
    Returns
    -------
    dict
        包含因子数量、类别分布等信息
    """
    all_factors = list_factors()
    
    summary = {
        'total_factors': len(factor_registry),
        'categories': list(all_factors.keys()),
        'category_counts': {cat: len(factors) for cat, factors in all_factors.items()},
        'registry_version': '1.0.0'
    }
    
    return summary


# 主要导出接口
__all__ = [
    # 核心注册系统
    'FactorRegistry',
    'factor_registry',
    'register_factor',
    'get_factor',
    'list_factors',
    'search_factors',
    'get_factor_info',
    
    # 便捷计算函数
    'calculate_factor',
    'batch_calculate_factors',
    
    # 工具函数
    'get_factor_dependencies',
    'validate_data_for_factor',
    'get_factors_by_category',
    'get_registered_count',
    'get_factor_summary',
]

# 模块信息
__version__ = '1.0.0'