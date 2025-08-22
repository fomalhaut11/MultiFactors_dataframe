"""
因子池管理

管理因子池的存储、更新和查询
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class FactorPool:
    """
    因子池管理器
    
    管理因子的存储、更新、查询和元数据
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化因子池
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.factors = {}  # 因子数据存储
        self.metadata = {}  # 因子元数据
        self.categories = defaultdict(list)  # 因子分类
        self.update_history = []  # 更新历史
        
        # 配置参数
        self.max_factors = self.config.get('max_factors', 1000)
        self.auto_cleanup = self.config.get('auto_cleanup', True)
        self.max_history = self.config.get('max_history', 100)
        
        logger.info("Initialized FactorPool")
    
    def add_factor(self,
                   name: str,
                   factor_data: pd.Series,
                   category: Optional[str] = None,
                   description: Optional[str] = None,
                   source: Optional[str] = None,
                   **metadata) -> bool:
        """
        添加因子到池中
        
        Parameters
        ----------
        name : str
            因子名称
        factor_data : pd.Series
            因子数据，必须是MultiIndex Series
        category : str, optional
            因子类别
        description : str, optional
            因子描述
        source : str, optional
            因子来源
        **metadata : dict
            其他元数据
            
        Returns
        -------
        bool
            是否成功添加
        """
        try:
            # 验证因子数据
            self._validate_factor_data(name, factor_data)
            
            # 检查因子池容量
            if len(self.factors) >= self.max_factors:
                if self.auto_cleanup:
                    self._cleanup_old_factors()
                else:
                    logger.warning(f"Factor pool is full ({self.max_factors} factors)")
                    return False
            
            # 存储因子数据
            self.factors[name] = factor_data.copy()
            
            # 存储元数据
            self.metadata[name] = {
                'name': name,
                'category': category or 'uncategorized',
                'description': description or '',
                'source': source or 'unknown',
                'added_time': datetime.now(),
                'data_shape': factor_data.shape,
                'data_range': self._get_data_range(factor_data),
                'coverage': self._calculate_coverage(factor_data),
                **metadata
            }
            
            # 更新分类索引
            if category:
                if name not in self.categories[category]:
                    self.categories[category].append(name)
            
            # 记录更新历史
            self.update_history.append({
                'timestamp': datetime.now(),
                'action': 'add',
                'factor_name': name,
                'category': category
            })
            
            # 限制历史记录数量
            if len(self.update_history) > self.max_history:
                self.update_history = self.update_history[-self.max_history:]
            
            logger.info(f"Added factor '{name}' to pool (category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add factor '{name}': {e}")
            return False
    
    def remove_factor(self, name: str) -> bool:
        """
        从池中移除因子
        
        Parameters
        ----------
        name : str
            因子名称
            
        Returns
        -------
        bool
            是否成功移除
        """
        try:
            if name not in self.factors:
                logger.warning(f"Factor '{name}' not found in pool")
                return False
            
            # 获取类别信息
            category = self.metadata.get(name, {}).get('category', 'uncategorized')
            
            # 移除因子数据和元数据
            del self.factors[name]
            if name in self.metadata:
                del self.metadata[name]
            
            # 更新分类索引
            if category in self.categories and name in self.categories[category]:
                self.categories[category].remove(name)
                if not self.categories[category]:  # 如果类别为空，删除类别
                    del self.categories[category]
            
            # 记录更新历史
            self.update_history.append({
                'timestamp': datetime.now(),
                'action': 'remove',
                'factor_name': name,
                'category': category
            })
            
            logger.info(f"Removed factor '{name}' from pool")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove factor '{name}': {e}")
            return False
    
    def get_factor(self, name: str) -> Optional[pd.Series]:
        """
        获取因子数据
        
        Parameters
        ----------
        name : str
            因子名称
            
        Returns
        -------
        pd.Series or None
            因子数据
        """
        return self.factors.get(name)
    
    def get_factors(self,
                    names: Optional[List[str]] = None,
                    category: Optional[str] = None,
                    pattern: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        获取多个因子数据
        
        Parameters
        ----------
        names : List[str], optional
            指定的因子名称列表
        category : str, optional
            指定类别的所有因子
        pattern : str, optional
            名称匹配模式
            
        Returns
        -------
        Dict[str, pd.Series]
            因子数据字典
        """
        if names is not None:
            # 返回指定名称的因子
            return {name: self.factors[name] for name in names if name in self.factors}
        
        elif category is not None:
            # 返回指定类别的因子
            if category in self.categories:
                return {name: self.factors[name] for name in self.categories[category]
                        if name in self.factors}
            else:
                return {}
        
        elif pattern is not None:
            # 返回名称匹配模式的因子
            import re
            matched_names = [name for name in self.factors.keys()
                             if re.search(pattern, name)]
            return {name: self.factors[name] for name in matched_names}
        
        else:
            # 返回所有因子
            return self.factors.copy()
    
    def get_metadata(self,
                     name: Optional[str] = None,
                     category: Optional[str] = None) -> Dict[str, Any]:
        """
        获取因子元数据
        
        Parameters
        ----------
        name : str, optional
            特定因子名称
        category : str, optional
            特定类别
            
        Returns
        -------
        Dict[str, Any]
            元数据字典
        """
        if name is not None:
            return self.metadata.get(name, {})
        
        elif category is not None:
            if category in self.categories:
                return {name: self.metadata.get(name, {})
                        for name in self.categories[category]}
            else:
                return {}
        
        else:
            return self.metadata.copy()
    
    def list_factors(self,
                     category: Optional[str] = None,
                     sort_by: str = 'name') -> List[str]:
        """
        列出因子名称
        
        Parameters
        ----------
        category : str, optional
            指定类别
        sort_by : str
            排序方式：'name', 'added_time', 'category'
            
        Returns
        -------
        List[str]
            因子名称列表
        """
        if category is not None:
            factor_names = self.categories.get(category, [])
        else:
            factor_names = list(self.factors.keys())
        
        # 排序
        if sort_by == 'name':
            factor_names.sort()
        elif sort_by == 'added_time':
            factor_names.sort(key=lambda x: self.metadata.get(x, {}).get('added_time', datetime.min))
        elif sort_by == 'category':
            factor_names.sort(key=lambda x: self.metadata.get(x, {}).get('category', ''))
        
        return factor_names
    
    def list_categories(self) -> List[str]:
        """
        列出所有类别
        
        Returns
        -------
        List[str]
            类别列表
        """
        return list(self.categories.keys())
    
    def search_factors(self,
                       keyword: Optional[str] = None,
                       category: Optional[str] = None,
                       min_coverage: Optional[float] = None,
                       date_range: Optional[Tuple[str, str]] = None) -> List[str]:
        """
        搜索因子
        
        Parameters
        ----------
        keyword : str, optional
            关键词搜索（名称或描述）
        category : str, optional
            类别筛选
        min_coverage : float, optional
            最小覆盖率要求
        date_range : Tuple[str, str], optional
            日期范围要求
            
        Returns
        -------
        List[str]
            匹配的因子名称列表
        """
        matched_factors = []
        
        for name, metadata in self.metadata.items():
            # 关键词匹配
            if keyword is not None:
                if (keyword.lower() not in name.lower() and
                    keyword.lower() not in metadata.get('description', '').lower()):
                    continue
            
            # 类别匹配
            if category is not None and metadata.get('category') != category:
                continue
            
            # 覆盖率筛选
            if min_coverage is not None:
                factor_coverage = metadata.get('coverage', 0)
                if factor_coverage < min_coverage:
                    continue
            
            # 日期范围筛选
            if date_range is not None:
                # 这里需要检查因子数据的日期范围
                factor_data = self.factors.get(name)
                if factor_data is not None:
                    factor_dates = factor_data.index.get_level_values(0).unique()
                    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    if factor_dates.min() > end_date or factor_dates.max() < start_date:
                        continue
            
            matched_factors.append(name)
        
        return matched_factors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取因子池统计信息
        
        Returns
        -------
        Dict[str, Any]
            统计信息
        """
        stats = {
            'total_factors': len(self.factors),
            'categories': dict(self.categories),
            'category_counts': {cat: len(factors) for cat, factors in self.categories.items()},
            'pool_capacity': self.max_factors,
            'capacity_usage': len(self.factors) / self.max_factors if self.max_factors > 0 else 0
        }
        
        # 计算覆盖率统计
        if self.metadata:
            coverages = [meta.get('coverage', 0) for meta in self.metadata.values()]
            stats['coverage_stats'] = {
                'mean': float(np.mean(coverages)),
                'std': float(np.std(coverages)),
                'min': float(np.min(coverages)),
                'max': float(np.max(coverages))
            }
        
        # 计算数据量统计
        if self.factors:
            data_sizes = [len(factor) for factor in self.factors.values()]
            stats['data_size_stats'] = {
                'mean': float(np.mean(data_sizes)),
                'std': float(np.std(data_sizes)),
                'min': int(np.min(data_sizes)),
                'max': int(np.max(data_sizes)),
                'total': int(np.sum(data_sizes))
            }
        
        return stats
    
    def _validate_factor_data(self, name: str, factor_data: pd.Series):
        """
        验证因子数据格式
        
        Parameters
        ----------
        name : str
            因子名称
        factor_data : pd.Series
            因子数据
        """
        if not isinstance(factor_data, pd.Series):
            raise TypeError(f"Factor {name} must be pd.Series")
        
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError(f"Factor {name} must have MultiIndex")
        
        if factor_data.index.nlevels != 2:
            raise ValueError(f"Factor {name} must have 2-level MultiIndex (dates, stocks)")
        
        if len(factor_data) == 0:
            raise ValueError(f"Factor {name} is empty")
        
        if not pd.api.types.is_numeric_dtype(factor_data.dtype):
            raise TypeError(f"Factor {name} must have numeric dtype")
    
    def _get_data_range(self, factor_data: pd.Series) -> Dict[str, Any]:
        """
        获取数据范围信息
        
        Parameters
        ----------
        factor_data : pd.Series
            因子数据
            
        Returns
        -------
        Dict[str, Any]
            数据范围信息
        """
        try:
            dates = factor_data.index.get_level_values(0).unique()
            stocks = factor_data.index.get_level_values(1).unique()
            values = factor_data.dropna()
            
            return {
                'date_range': (dates.min(), dates.max()),
                'stock_count': len(stocks),
                'date_count': len(dates),
                'value_range': (float(values.min()), float(values.max())) if len(values) > 0 else (0, 0),
                'total_observations': len(factor_data)
            }
        except Exception as e:
            logger.debug(f"Error calculating data range: {e}")
            return {}
    
    def _calculate_coverage(self, factor_data: pd.Series) -> float:
        """
        计算因子覆盖率
        
        Parameters
        ----------
        factor_data : pd.Series
            因子数据
            
        Returns
        -------
        float
            覆盖率（0-1）
        """
        try:
            total_count = len(factor_data)
            non_null_count = factor_data.count()
            return non_null_count / total_count if total_count > 0 else 0
        except:
            return 0
    
    def _cleanup_old_factors(self, max_factors_to_keep: Optional[int] = None):
        """
        清理旧因子
        
        Parameters
        ----------
        max_factors_to_keep : int, optional
            保留的最大因子数
        """
        max_keep = max_factors_to_keep or int(self.max_factors * 0.8)
        
        if len(self.factors) <= max_keep:
            return
        
        # 按添加时间排序，保留最新的因子
        factor_times = []
        for name, metadata in self.metadata.items():
            added_time = metadata.get('added_time', datetime.min)
            factor_times.append((added_time, name))
        
        factor_times.sort()  # 旧的在前
        
        # 移除最旧的因子
        factors_to_remove = len(self.factors) - max_keep
        for i in range(factors_to_remove):
            _, name_to_remove = factor_times[i]
            self.remove_factor(name_to_remove)
        
        logger.info(f"Cleaned up {factors_to_remove} old factors")
    
    def clear(self):
        """清空因子池"""
        self.factors.clear()
        self.metadata.clear()
        self.categories.clear()
        self.update_history.append({
            'timestamp': datetime.now(),
            'action': 'clear',
            'factor_name': 'all',
            'category': 'all'
        })
        logger.info("Cleared factor pool")
    
    def export_metadata(self) -> pd.DataFrame:
        """
        导出元数据为DataFrame
        
        Returns
        -------
        pd.DataFrame
            元数据DataFrame
        """
        if not self.metadata:
            return pd.DataFrame()
        
        # 转换为DataFrame
        metadata_list = []
        for name, metadata in self.metadata.items():
            flattened = {'factor_name': name}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # 展平嵌套字典
                    for subkey, subvalue in value.items():
                        flattened[f"{key}_{subkey}"] = subvalue
                else:
                    flattened[key] = value
            metadata_list.append(flattened)
        
        return pd.DataFrame(metadata_list)