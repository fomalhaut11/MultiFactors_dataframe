"""
因子组合器基类

提供因子组合的基础功能和接口定义
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CombinerBase(ABC):
    """
    因子组合器基类
    
    定义了因子组合的标准接口和通用功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化组合器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.combination_history = []
        self._validate_config()
        
        # 默认配置
        self.min_weight = self.config.get('min_weight', 0.0)
        self.max_weight = self.config.get('max_weight', 1.0)
        self.normalize_weights = self.config.get('normalize', True)
        self.handle_missing = self.config.get('handle_missing', 'forward_fill')
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def combine(self, 
               factors: Dict[str, pd.Series],
               weights: Optional[Dict[str, float]] = None,
               **kwargs) -> pd.Series:
        """
        组合多个因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典，key为因子名，value为MultiIndex Series
        weights : Dict[str, float], optional
            因子权重，如果为None则需要计算
        **kwargs : dict
            其他参数
            
        Returns
        -------
        pd.Series
            组合后的因子，MultiIndex Series格式
        """
        pass
    
    @abstractmethod
    def calculate_weights(self,
                         factors: Dict[str, pd.Series],
                         evaluation_results: Optional[Dict] = None,
                         **kwargs) -> Dict[str, float]:
        """
        计算因子权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
        evaluation_results : Dict, optional
            因子评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            因子权重字典
        """
        pass
    
    def validate_factors(self, factors: Dict[str, pd.Series]) -> bool:
        """
        验证因子格式
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            待验证的因子字典
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        ValueError
            如果因子格式不正确
        """
        if not factors:
            raise ValueError("Factors dictionary cannot be empty")
        
        for name, factor in factors.items():
            # 检查类型
            if not isinstance(factor, pd.Series):
                raise TypeError(f"Factor {name} must be pd.Series, got {type(factor)}")
            
            # 检查索引
            if not isinstance(factor.index, pd.MultiIndex):
                raise ValueError(f"Factor {name} must have MultiIndex")
            
            # 检查索引层级
            if factor.index.nlevels != 2:
                raise ValueError(
                    f"Factor {name} must have 2-level MultiIndex (dates, stocks), "
                    f"got {factor.index.nlevels} levels"
                )
            
            # 检查是否有数据
            if len(factor) == 0:
                raise ValueError(f"Factor {name} is empty")
            
            # 检查数值类型
            if not pd.api.types.is_numeric_dtype(factor.dtype):
                raise TypeError(f"Factor {name} must have numeric dtype, got {factor.dtype}")
        
        logger.debug(f"Validated {len(factors)} factors")
        return True
    
    def align_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        对齐因子索引
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子字典
            
        Returns
        -------
        Dict[str, pd.Series]
            对齐后的因子字典
        """
        if len(factors) == 1:
            return factors
        
        # 找到公共索引
        common_index = None
        for name, factor in factors.items():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        if len(common_index) == 0:
            raise ValueError("No common index found among factors")
        
        # 对齐所有因子
        aligned = {}
        for name, factor in factors.items():
            aligned[name] = factor.reindex(common_index)
            
            # 处理缺失值
            if self.handle_missing == 'forward_fill':
                aligned[name] = aligned[name].fillna(method='ffill')
            elif self.handle_missing == 'drop':
                aligned[name] = aligned[name].dropna()
            elif self.handle_missing == 'zero':
                aligned[name] = aligned[name].fillna(0)
            # 如果是None或其他值，保持NaN
        
        logger.info(f"Aligned {len(factors)} factors to {len(common_index)} common observations")
        return aligned
    
    def normalize_weights_dict(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        归一化权重
        
        Parameters
        ----------
        weights : Dict[str, float]
            原始权重
            
        Returns
        -------
        Dict[str, float]
            归一化后的权重
        """
        if not weights:
            return weights
        
        # 应用权重约束
        constrained = {}
        for name, weight in weights.items():
            constrained[name] = np.clip(weight, self.min_weight, self.max_weight)
        
        # 归一化
        if self.normalize_weights:
            total = sum(constrained.values())
            if total > 0:
                normalized = {k: v/total for k, v in constrained.items()}
            else:
                # 如果所有权重都是0，使用等权
                n = len(constrained)
                normalized = {k: 1.0/n for k in constrained.keys()}
        else:
            normalized = constrained
        
        return normalized
    
    def _combine_linear(self, 
                       factors: Dict[str, pd.Series],
                       weights: Dict[str, float]) -> pd.Series:
        """
        线性组合因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
        weights : Dict[str, float]
            归一化后的权重
            
        Returns
        -------
        pd.Series
            组合后的因子
        """
        # 初始化结果
        result = None
        
        for name, factor in factors.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            weighted_factor = factor * weight
            
            if result is None:
                result = weighted_factor
            else:
                # 对齐索引后相加
                result = result.add(weighted_factor, fill_value=0)
        
        if result is None:
            # 如果所有权重都是0，返回空Series
            first_factor = next(iter(factors.values()))
            result = pd.Series(0, index=first_factor.index)
        
        result.name = 'composite_factor'
        return result
    
    def _normalize(self, factor: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        标准化因子
        
        Parameters
        ----------
        factor : pd.Series
            原始因子
        method : str
            标准化方法：'zscore', 'minmax', 'rank'
            
        Returns
        -------
        pd.Series
            标准化后的因子
        """
        if method == 'zscore':
            # Z-Score标准化
            grouped = factor.groupby(level=0)  # 按日期分组
            result = grouped.transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
        elif method == 'minmax':
            # Min-Max标准化
            grouped = factor.groupby(level=0)
            result = grouped.transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
            )
        elif method == 'rank':
            # 排名标准化
            grouped = factor.groupby(level=0)
            result = grouped.rank(pct=True)
        else:
            result = factor
        
        return result
    
    def _validate_config(self):
        """验证配置参数"""
        # 验证权重约束
        if 'min_weight' in self.config:
            if not isinstance(self.config['min_weight'], (int, float)):
                raise TypeError("min_weight must be numeric")
            if self.config['min_weight'] < 0:
                raise ValueError("min_weight cannot be negative")
        
        if 'max_weight' in self.config:
            if not isinstance(self.config['max_weight'], (int, float)):
                raise TypeError("max_weight must be numeric")
            if self.config['max_weight'] < 0:
                raise ValueError("max_weight cannot be negative")
        
        if 'min_weight' in self.config and 'max_weight' in self.config:
            if self.config['min_weight'] > self.config['max_weight']:
                raise ValueError("min_weight cannot be greater than max_weight")
        
        # 验证缺失值处理方法
        valid_missing_methods = ['forward_fill', 'drop', 'zero', None]
        if 'handle_missing' in self.config:
            if self.config['handle_missing'] not in valid_missing_methods:
                raise ValueError(
                    f"Invalid handle_missing method: {self.config['handle_missing']}. "
                    f"Must be one of {valid_missing_methods}"
                )
    
    def get_combination_info(self) -> Dict[str, Any]:
        """
        获取组合信息
        
        Returns
        -------
        Dict[str, Any]
            组合器的配置和状态信息
        """
        return {
            'class': self.__class__.__name__,
            'config': self.config,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'normalize_weights': self.normalize_weights,
            'handle_missing': self.handle_missing,
            'history_count': len(self.combination_history)
        }
    
    def save_combination_history(self, 
                                factors: List[str],
                                weights: Dict[str, float],
                                result_stats: Optional[Dict] = None):
        """
        保存组合历史
        
        Parameters
        ----------
        factors : List[str]
            因子名称列表
        weights : Dict[str, float]
            使用的权重
        result_stats : Dict, optional
            结果统计信息
        """
        record = {
            'timestamp': datetime.now(),
            'factors': factors,
            'weights': weights.copy(),
            'stats': result_stats or {}
        }
        self.combination_history.append(record)
        
        # 限制历史记录数量
        max_history = self.config.get('max_history', 100)
        if len(self.combination_history) > max_history:
            self.combination_history = self.combination_history[-max_history:]