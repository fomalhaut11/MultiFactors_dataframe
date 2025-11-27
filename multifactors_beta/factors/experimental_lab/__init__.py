"""
新因子实验室模块

专门管理新因子的完整生命周期：
- 因子注册和状态管理
- 计算引擎和测试协调
- 结果跟踪和筛选支持

核心设计理念：
1. 与正式repository隔离，避免污染
2. 完整的实验可追溯性
3. 为analyzer模块提供结构化数据源
"""

from .factor_registry import ExperimentalFactorRegistry
from .calculation_engine import FactorCalculationEngine
from .test_coordinator import TestCoordinator
from .results_tracker import ResultsTracker
from .experimental_manager import ExperimentalFactorManager

__all__ = [
    # 核心管理器
    'ExperimentalFactorManager',
    
    # 子组件
    'ExperimentalFactorRegistry',
    'FactorCalculationEngine', 
    'TestCoordinator',
    'ResultsTracker',
    
    # 便捷函数
    'register_experimental_factor',
    'calculate_experimental_factor',
    'test_experimental_factor',
    'get_experimental_results',
    'promote_to_production',
    'archive_experimental_factor'
]

__version__ = '1.0.0'

# 便捷函数
def register_experimental_factor(name: str, calculation_func, description: str = "", **metadata):
    """
    快速注册新因子到实验室
    
    Parameters:
    -----------
    name : str
        因子名称
    calculation_func : callable
        因子计算函数
    description : str
        因子描述
    **metadata : dict
        其他元数据
    """
    manager = ExperimentalFactorManager()
    return manager.register_factor(name, calculation_func, description, **metadata)


def calculate_experimental_factor(name: str, **kwargs):
    """
    计算实验因子
    
    Parameters:
    -----------
    name : str
        因子名称
    **kwargs : dict
        计算参数
    """
    manager = ExperimentalFactorManager()
    return manager.calculate_factor(name, **kwargs)


def test_experimental_factor(name: str, **test_params):
    """
    测试实验因子
    
    Parameters:
    -----------
    name : str
        因子名称
    **test_params : dict
        测试参数
    """
    manager = ExperimentalFactorManager()
    return manager.test_factor(name, **test_params)


def get_experimental_results(name: str = None, status: str = None):
    """
    获取实验结果
    
    Parameters:
    -----------
    name : str, optional
        因子名称，为空则获取所有
    status : str, optional 
        状态筛选
    """
    manager = ExperimentalFactorManager()
    return manager.get_results(name, status)


def promote_to_production(name: str, target_category: str):
    """
    将成功的实验因子提升到正式repository
    
    Parameters:
    -----------
    name : str
        因子名称
    target_category : str
        目标分类 (technical/financial/mixed/value等)
    """
    manager = ExperimentalFactorManager()
    return manager.promote_factor(name, target_category)


def archive_experimental_factor(name: str, reason: str = ""):
    """
    归档失败的实验因子
    
    Parameters:
    -----------
    name : str
        因子名称
    reason : str
        归档原因
    """
    manager = ExperimentalFactorManager()
    return manager.archive_factor(name, reason)