"""
因子注册系统核心模块

提供装饰器和注册管理功能，将纯工具函数包装为标准的因子接口
"""
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional, Union
import logging
from functools import wraps
import inspect

logger = logging.getLogger(__name__)


class FactorRegistry:
    """因子注册表管理器"""
    
    def __init__(self):
        self._registry = {}
        self._metadata = {}
        self._categories = {}
        self._file_registry = {}  # 新增：跟踪从文件加载的因子
    
    def register(self, 
                 name: str, 
                 category: str,
                 description: str = None,
                 dependencies: list = None,
                 **metadata):
        """
        注册因子装饰器
        
        Parameters
        ----------
        name : str
            因子名称
        category : str
            因子类别 (profitability, value, quality, technical, etc.)
        description : str, optional
            因子描述
        dependencies : list, optional
            依赖的数据字段列表
        **metadata
            其他元数据
            
        Returns
        -------
        decorator
            装饰器函数
            
        Examples
        --------
        @factor_registry.register('ROE_ttm', 'profitability', 
                                 description='TTM净资产收益率')
        def calculate_roe_ttm(financial_data, **kwargs):
            # 计算逻辑
            return result
        """
        def decorator(func: Callable):
            # 创建因子包装器
            factor_func = self._create_factor_wrapper(
                func, name, category, description, dependencies, **metadata
            )
            
            # 注册到registry
            self._registry[name] = factor_func
            self._metadata[name] = {
                'category': category,
                'description': description or func.__doc__ or '',
                'dependencies': dependencies or [],
                'function_name': func.__name__,
                **metadata
            }
            
            # 更新类别索引
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
            
            return factor_func
        
        return decorator
    
    def _create_factor_wrapper(self, func, name, category, description, dependencies, **metadata):
        """
        创建因子包装器，提供标准化接口
        
        Parameters
        ----------
        func : callable
            原始计算函数
        name, category, description, dependencies
            因子元数据
            
        Returns
        -------
        callable
            包装后的因子函数
        """
        @wraps(func)
        def factor_wrapper(*args, **kwargs):
            # 添加标准化的错误处理
            try:
                # 检查函数签名并预处理参数
                sig = inspect.signature(func)
                bound_args = sig.bind_partial(*args, **kwargs)
                bound_args.apply_defaults()
                
                # 调用原始计算函数
                result = func(*bound_args.args, **bound_args.kwargs)
                
                # 标准化输出格式
                if isinstance(result, pd.Series):
                    # 确保Series有正确的名称
                    result.name = name
                    return result
                elif isinstance(result, pd.DataFrame):
                    # 如果是DataFrame，尝试转换为Series
                    if result.shape[1] == 1:
                        series_result = result.iloc[:, 0]
                        series_result.name = name
                        return series_result
                    else:
                        logger.warning(f"因子 {name} 返回多列DataFrame，返回原始结果")
                        return result
                else:
                    logger.warning(f"因子 {name} 返回非标准格式: {type(result)}")
                    return result
                    
            except Exception as e:
                logger.error(f"因子 {name} 计算失败: {e}")
                # 返回空Series而不是抛出异常
                return pd.Series(dtype=float, name=name)
        
        # 添加元数据属性
        factor_wrapper.factor_name = name
        factor_wrapper.factor_category = category
        factor_wrapper.factor_description = description
        factor_wrapper.factor_dependencies = dependencies or []
        factor_wrapper.original_function = func
        
        return factor_wrapper
    
    def get(self, name: str) -> Optional[Callable]:
        """
        获取注册的因子函数
        
        Parameters
        ----------
        name : str
            因子名称
            
        Returns
        -------
        callable or None
            因子计算函数
        """
        return self._registry.get(name)
    
    def list_factors(self, category: str = None) -> Union[list, dict]:
        """
        列出注册的因子
        
        Parameters
        ----------
        category : str, optional
            指定类别，如果不指定则返回所有类别
            
        Returns
        -------
        list or dict
            因子列表或按类别分组的字典
        """
        if category:
            return self._categories.get(category, [])
        else:
            return self._categories.copy()
    
    def get_metadata(self, name: str) -> dict:
        """
        获取因子元数据
        
        Parameters
        ----------
        name : str
            因子名称
            
        Returns
        -------
        dict
            因子元数据
        """
        return self._metadata.get(name, {})
    
    def search(self, keyword: str = None, category: str = None) -> list:
        """
        搜索因子
        
        Parameters
        ----------
        keyword : str, optional
            关键词搜索
        category : str, optional
            类别筛选
            
        Returns
        -------
        list
            匹配的因子名称列表
        """
        results = []
        
        for name, metadata in self._metadata.items():
            # 类别筛选
            if category and metadata.get('category') != category:
                continue
            
            # 关键词搜索
            if keyword:
                keyword_lower = keyword.lower()
                if (keyword_lower in name.lower() or 
                    keyword_lower in metadata.get('description', '').lower()):
                    results.append(name)
            else:
                results.append(name)
        
        return results
    
    def validate_dependencies(self, name: str, data: pd.DataFrame) -> tuple:
        """
        验证因子依赖的数据字段
        
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
        metadata = self.get_metadata(name)
        dependencies = metadata.get('dependencies', [])
        
        if not dependencies:
            return True, []
        
        available_fields = set(data.columns)
        required_fields = set(dependencies)
        missing_fields = required_fields - available_fields
        
        return len(missing_fields) == 0, list(missing_fields)
    
    def __len__(self):
        """返回注册因子数量"""
        return len(self._registry)
    
    def __contains__(self, name):
        """检查因子是否已注册"""
        return name in self._registry
    
    def __iter__(self):
        """迭代因子名称"""
        return iter(self._registry.keys())
    
    def register_from_file(self, 
                           name: str,
                           category: str, 
                           description: str,
                           dependencies: list,
                           calculate_func: callable,
                           file_path: str = None,
                           **metadata):
        """
        从文件注册因子
        
        Parameters
        ----------
        name : str
            因子名称
        category : str
            因子类别
        description : str
            因子描述
        dependencies : list
            数据依赖
        calculate_func : callable
            计算函数
        file_path : str, optional
            源文件路径
        **metadata
            其他元数据
        """
        # 创建标准化的因子函数包装器
        factor_func = self._create_factor_wrapper(
            calculate_func, name, category, description, dependencies, **metadata
        )
        
        # 注册到registry
        self._registry[name] = factor_func
        self._metadata[name] = {
            'category': category,
            'description': description,
            'dependencies': dependencies or [],
            'function_name': calculate_func.__name__,
            'source': 'file',
            'file_path': file_path,
            **metadata
        }
        
        # 更新类别索引
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        # 记录文件注册信息
        if file_path:
            self._file_registry[name] = file_path
        
        logger.info(f"从文件注册因子: {name} <- {file_path}")
        
        return factor_func
    
    def get_file_factors(self) -> Dict[str, str]:
        """
        获取从文件注册的因子列表
        
        Returns
        -------
        dict
            因子名称到文件路径的映射
        """
        return self._file_registry.copy()
    
    def is_file_factor(self, name: str) -> bool:
        """
        检查因子是否从文件加载
        
        Parameters
        ----------
        name : str
            因子名称
            
        Returns
        -------
        bool
            是否从文件加载
        """
        metadata = self._metadata.get(name, {})
        return metadata.get('source') == 'file'


# 全局注册表实例
factor_registry = FactorRegistry()


def register_factor(name: str, 
                   category: str,
                   description: str = None,
                   dependencies: list = None,
                   **metadata):
    """
    因子注册装饰器的便捷接口
    
    Parameters
    ----------
    name : str
        因子名称
    category : str
        因子类别
    description : str, optional
        因子描述
    dependencies : list, optional
        依赖的数据字段
    **metadata
        其他元数据
        
    Examples
    --------
    @register_factor('ROE_ttm', 'profitability',
                     description='TTM净资产收益率',
                     dependencies=['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH'])
    def calculate_roe_ttm(financial_data, **kwargs):
        # 计算逻辑
        return result
    """
    return factor_registry.register(name, category, description, dependencies, **metadata)


def get_factor(name: str) -> Optional[Callable]:
    """获取因子计算函数"""
    return factor_registry.get(name)


def list_factors(category: str = None) -> Union[list, dict]:
    """列出可用因子"""
    return factor_registry.list_factors(category)


def search_factors(keyword: str = None, category: str = None) -> list:
    """搜索因子"""
    return factor_registry.search(keyword, category)


def get_factor_info(name: str) -> dict:
    """获取因子信息"""
    return factor_registry.get_metadata(name)


# 导出列表
__all__ = [
    'FactorRegistry',
    'factor_registry',
    'register_factor',
    'get_factor',
    'list_factors',
    'search_factors',
    'get_factor_info',
]