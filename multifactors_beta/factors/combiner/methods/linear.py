"""
线性组合方法
"""

from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LinearCombiner:
    """
    线性组合器
    
    实现因子的线性加权组合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化线性组合器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.handle_missing = self.config.get('handle_missing', 'forward_fill')
        self.normalize_before = self.config.get('normalize_before', False)
        self.normalize_after = self.config.get('normalize_after', True)
        self.normalize_method = self.config.get('normalize_method', 'zscore')
    
    def combine(self,
               factors: Dict[str, pd.Series],
               weights: Dict[str, float]) -> pd.Series:
        """
        线性组合因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
        weights : Dict[str, float]
            权重字典
            
        Returns
        -------
        pd.Series
            组合后的因子
        """
        if not factors:
            raise ValueError("No factors to combine")
        
        if not weights:
            raise ValueError("No weights provided")
        
        # 预处理：标准化
        if self.normalize_before:
            factors = self._normalize_factors(factors)
        
        # 对齐因子
        aligned_factors = self._align_factors(factors)
        
        # 线性组合
        result = self._weighted_sum(aligned_factors, weights)
        
        # 后处理：标准化
        if self.normalize_after:
            result = self._normalize_series(result)
        
        result.name = 'composite_factor'
        
        logger.info(f"Combined {len(factors)} factors using linear method")
        return result
    
    def _align_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        对齐因子索引
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子
            
        Returns
        -------
        Dict[str, pd.Series]
            对齐后的因子
        """
        # 找到公共索引
        common_index = None
        for factor in factors.values():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        if len(common_index) == 0:
            raise ValueError("No common index found among factors")
        
        # 对齐并处理缺失值
        aligned = {}
        for name, factor in factors.items():
            aligned_factor = factor.reindex(common_index)
            
            # 处理缺失值
            if self.handle_missing == 'forward_fill':
                aligned_factor = aligned_factor.fillna(method='ffill')
            elif self.handle_missing == 'drop':
                # 标记需要删除的索引
                pass  # 在后续步骤统一处理
            elif self.handle_missing == 'zero':
                aligned_factor = aligned_factor.fillna(0)
            
            aligned[name] = aligned_factor
        
        # 如果选择drop，删除任何因子有缺失值的观测
        if self.handle_missing == 'drop':
            # 找出所有因子都有值的索引
            valid_mask = pd.Series(True, index=common_index)
            for factor in aligned.values():
                valid_mask &= ~factor.isna()
            
            valid_index = common_index[valid_mask]
            aligned = {name: factor.loc[valid_index] for name, factor in aligned.items()}
        
        return aligned
    
    def _weighted_sum(self,
                     factors: Dict[str, pd.Series],
                     weights: Dict[str, float]) -> pd.Series:
        """
        计算加权和
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
        weights : Dict[str, float]
            权重
            
        Returns
        -------
        pd.Series
            加权和
        """
        # 初始化结果
        result = None
        total_weight = 0
        
        for name, factor in factors.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            weighted_factor = factor * weight
            total_weight += weight
            
            if result is None:
                result = weighted_factor.copy()
            else:
                result = result + weighted_factor
        
        # 如果没有有效权重，返回零值
        if result is None:
            first_factor = next(iter(factors.values()))
            result = pd.Series(0, index=first_factor.index)
        
        return result
    
    def _normalize_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        标准化因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子
            
        Returns
        -------
        Dict[str, pd.Series]
            标准化后的因子
        """
        normalized = {}
        for name, factor in factors.items():
            normalized[name] = self._normalize_series(factor)
        return normalized
    
    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """
        标准化单个Series
        
        Parameters
        ----------
        series : pd.Series
            原始Series
            
        Returns
        -------
        pd.Series
            标准化后的Series
        """
        if self.normalize_method == 'zscore':
            # Z-Score标准化（按日期）
            grouped = series.groupby(level=0)
            result = grouped.transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        elif self.normalize_method == 'minmax':
            # Min-Max标准化（按日期）
            grouped = series.groupby(level=0)
            result = grouped.transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) 
                if x.max() > x.min() else 0.5
            )
        elif self.normalize_method == 'rank':
            # 排名标准化（按日期）
            grouped = series.groupby(level=0)
            result = grouped.rank(pct=True)
        elif self.normalize_method == 'robust':
            # 稳健标准化（使用中位数和MAD）
            grouped = series.groupby(level=0)
            result = grouped.transform(
                lambda x: (x - x.median()) / (1.4826 * (x - x.median()).abs().median())
                if (x - x.median()).abs().median() > 0 else 0
            )
        else:
            result = series
        
        return result