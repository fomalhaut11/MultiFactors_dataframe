"""
因子选择器基类

定义了因子选择的标准接口和通用功能
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SelectorBase(ABC):
    """
    因子选择器基类
    
    定义了因子选择的标准接口和通用功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化选择器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.selection_history = []
        self._validate_config()
        
        # 默认配置
        self.min_weight = self.config.get('min_weight', 0.0)
        self.max_weight = self.config.get('max_weight', 1.0)
        self.min_factors = self.config.get('min_factors', 1)
        self.max_factors = self.config.get('max_factors', 20)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def select(self,
               factors_pool: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               constraints: Optional[Dict] = None,
               **kwargs) -> Dict[str, Any]:
        """
        选择因子
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池，key为因子名，value为MultiIndex Series
        evaluation_results : Dict, optional
            评估结果，用于计算选择权重
        constraints : Dict, optional
            选择约束条件
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            选择结果，包含selected_factors, factors_data等
        """
        pass
    
    @abstractmethod
    def score_factors(self,
                      factors_pool: Dict[str, pd.Series],
                      evaluation_results: Optional[Dict] = None,
                      **kwargs) -> Dict[str, float]:
        """
        为因子打分
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            因子得分字典
        """
        pass
    
    def validate_factors(self, factors_pool: Dict[str, pd.Series]) -> bool:
        """
        验证因子格式
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            待验证的因子池
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        ValueError
            如果因子格式不正确
        """
        if not factors_pool:
            raise ValueError("Factors pool cannot be empty")
        
        for name, factor in factors_pool.items():
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
        
        logger.debug(f"Validated {len(factors_pool)} factors")
        return True
    
    def apply_filters(self,
                      factors_pool: Dict[str, pd.Series],
                      filters: List[Any],
                      evaluation_results: Optional[Dict] = None) -> Dict[str, pd.Series]:
        """
        应用筛选器
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            原始因子池
        filters : List[Any]
            筛选器列表
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        Dict[str, pd.Series]
            筛选后的因子池
        """
        filtered = factors_pool.copy()
        
        for filter_obj in filters:
            try:
                if hasattr(filter_obj, 'filter'):
                    # 检查筛选器的filter方法签名
                    if 'evaluation_results' in filter_obj.filter.__code__.co_varnames:
                        filtered = filter_obj.filter(filtered, evaluation_results)
                    else:
                        filtered = filter_obj.filter(filtered)
                else:
                    logger.warning(f"Filter {filter_obj} does not have filter method")
            except Exception as e:
                logger.error(f"Error applying filter {filter_obj}: {e}")
                continue
        
        logger.info(f"Applied {len(filters)} filters, {len(filtered)} factors remaining")
        return filtered
    
    def check_diversity(self,
                        selected_factors: List[str],
                        correlation_matrix: pd.DataFrame) -> float:
        """
        检查因子多样性
        
        Parameters
        ----------
        selected_factors : List[str]
            选中的因子列表
        correlation_matrix : pd.DataFrame
            相关性矩阵
            
        Returns
        -------
        float
            多样性得分（0-1，越高越多样）
        """
        if len(selected_factors) < 2:
            return 1.0
        
        # 计算选中因子间的平均相关性
        correlations = []
        for i, f1 in enumerate(selected_factors):
            for f2 in selected_factors[i+1:]:
                if (f1 in correlation_matrix.index and 
                    f2 in correlation_matrix.index):
                    correlations.append(abs(correlation_matrix.loc[f1, f2]))
        
        if not correlations:
            return 1.0
        
        avg_correlation = np.mean(correlations)
        diversity = 1 - avg_correlation
        return max(0, diversity)
    
    def extract_scores(self,
                       evaluation_results: Dict,
                       score_type: str = 'total_score') -> Dict[str, float]:
        """
        从评估结果中提取得分
        
        Parameters
        ----------
        evaluation_results : Dict
            评估结果字典
        score_type : str
            得分类型：'total_score', 'ic_mean', 'icir'等
            
        Returns
        -------
        Dict[str, float]
            因子得分字典
        """
        scores = {}
        
        for factor_name, result in evaluation_results.items():
            try:
                score = None
                
                # 尝试从不同属性获取分数
                if hasattr(result, score_type):
                    score = getattr(result, score_type)
                elif hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                    score = result.metrics.get(score_type)
                elif isinstance(result, dict):
                    score = result.get(score_type)
                
                if score is not None and np.isfinite(score):
                    scores[factor_name] = float(score)
                    
            except Exception as e:
                logger.warning(f"Failed to extract {score_type} for {factor_name}: {e}")
                continue
        
        return scores
    
    def _validate_config(self):
        """验证配置参数"""
        # 验证因子数量约束
        if 'min_factors' in self.config:
            if not isinstance(self.config['min_factors'], int) or self.config['min_factors'] < 1:
                raise ValueError("min_factors must be positive integer")
        
        if 'max_factors' in self.config:
            if not isinstance(self.config['max_factors'], int) or self.config['max_factors'] < 1:
                raise ValueError("max_factors must be positive integer")
        
        if ('min_factors' in self.config and 'max_factors' in self.config and
            self.config['min_factors'] > self.config['max_factors']):
            raise ValueError("min_factors cannot be greater than max_factors")
    
    def save_selection_history(self,
                               selected_factors: List[str],
                               scores: Dict[str, float],
                               method: str,
                               stats: Optional[Dict] = None):
        """
        保存选择历史
        
        Parameters
        ----------
        selected_factors : List[str]
            选中的因子
        scores : Dict[str, float]
            因子得分
        method : str
            选择方法
        stats : Dict, optional
            统计信息
        """
        record = {
            'timestamp': datetime.now(),
            'selected_factors': selected_factors.copy(),
            'scores': scores.copy(),
            'method': method,
            'stats': stats or {}
        }
        self.selection_history.append(record)
        
        # 限制历史记录数量
        max_history = self.config.get('max_history', 50)
        if len(self.selection_history) > max_history:
            self.selection_history = self.selection_history[-max_history:]
    
    def get_selection_info(self) -> Dict[str, Any]:
        """
        获取选择器信息
        
        Returns
        -------
        Dict[str, Any]
            选择器的配置和状态信息
        """
        return {
            'class': self.__class__.__name__,
            'config': self.config,
            'min_factors': self.min_factors,
            'max_factors': self.max_factors,
            'history_count': len(self.selection_history)
        }