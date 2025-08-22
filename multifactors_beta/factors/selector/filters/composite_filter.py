"""
复合筛选器

组合多个筛选器进行综合筛选
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class CompositeFilter(BaseFilter):
    """
    复合筛选器
    
    组合多个筛选器，支持AND/OR逻辑组合
    """
    
    def __init__(self,
                 filters: List[BaseFilter],
                 logic: str = 'AND',
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化复合筛选器
        
        Parameters
        ----------
        filters : List[BaseFilter]
            子筛选器列表
        logic : str
            逻辑组合方式：'AND', 'OR'
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        self.filters = filters or []
        self.logic = logic.upper()
        
        if self.logic not in ['AND', 'OR']:
            raise ValueError(f"Invalid logic: {logic}. Must be 'AND' or 'OR'")
        
        if not self.filters:
            logger.warning("No filters provided for CompositeFilter")
        
        logger.info(f"Initialized CompositeFilter with {len(self.filters)} filters, logic: {self.logic}")
    
    def filter(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               **kwargs) -> Dict[str, pd.Series]:
        """
        应用复合筛选
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            待筛选的因子
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            筛选后的因子
        """
        self.validate_inputs(factors, evaluation_results)
        
        if not self.filters:
            logger.warning("No filters to apply, returning original factors")
            return factors
        
        if len(self.filters) == 1:
            return self.filters[0].filter(factors, evaluation_results, **kwargs)
        
        if self.logic == 'AND':
            return self._apply_and_logic(factors, evaluation_results, **kwargs)
        else:  # OR logic
            return self._apply_or_logic(factors, evaluation_results, **kwargs)
    
    def _apply_and_logic(self,
                         factors: Dict[str, pd.Series],
                         evaluation_results: Optional[Dict] = None,
                         **kwargs) -> Dict[str, pd.Series]:
        """
        应用AND逻辑：因子必须通过所有筛选器
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            待筛选的因子
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            筛选后的因子
        """
        current_factors = factors.copy()
        filter_results = []
        
        for i, filter_obj in enumerate(self.filters):
            try:
                filtered = filter_obj.filter(current_factors, evaluation_results, **kwargs)
                filter_results.append({
                    'filter_index': i,
                    'filter_type': filter_obj.__class__.__name__,
                    'input_count': len(current_factors),
                    'output_count': len(filtered),
                    'removed_factors': set(current_factors.keys()) - set(filtered.keys())
                })
                current_factors = filtered
                
                logger.debug(
                    f"AND filter {i+1}/{len(self.filters)} ({filter_obj.__class__.__name__}): "
                    f"{filter_results[-1]['input_count']} -> {filter_results[-1]['output_count']} factors"
                )
                
                # 如果没有因子剩余，提前结束
                if not current_factors:
                    logger.warning("No factors remaining after AND filtering")
                    break
                    
            except Exception as e:
                logger.error(f"Error in filter {i} ({filter_obj.__class__.__name__}): {e}")
                continue
        
        # 记录筛选历史
        self.filter_history.append({
            'logic': self.logic,
            'original_count': len(factors),
            'final_count': len(current_factors),
            'filter_results': filter_results
        })
        
        logger.info(
            f"CompositeFilter (AND): {len(factors)} -> {len(current_factors)} factors "
            f"after {len(self.filters)} filters"
        )
        
        return current_factors
    
    def _apply_or_logic(self,
                        factors: Dict[str, pd.Series],
                        evaluation_results: Optional[Dict] = None,
                        **kwargs) -> Dict[str, pd.Series]:
        """
        应用OR逻辑：因子只需通过任一筛选器
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            待筛选的因子
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            筛选后的因子
        """
        passed_factors = set()
        filter_results = []
        
        for i, filter_obj in enumerate(self.filters):
            try:
                filtered = filter_obj.filter(factors, evaluation_results, **kwargs)
                passed_in_this_filter = set(filtered.keys())
                passed_factors.update(passed_in_this_filter)
                
                filter_results.append({
                    'filter_index': i,
                    'filter_type': filter_obj.__class__.__name__,
                    'input_count': len(factors),
                    'output_count': len(filtered),
                    'passed_factors': passed_in_this_filter
                })
                
                logger.debug(
                    f"OR filter {i+1}/{len(self.filters)} ({filter_obj.__class__.__name__}): "
                    f"passed {len(passed_in_this_filter)} factors"
                )
                
            except Exception as e:
                logger.error(f"Error in filter {i} ({filter_obj.__class__.__name__}): {e}")
                continue
        
        # 构建最终结果
        final_factors = {
            name: factor for name, factor in factors.items()
            if name in passed_factors
        }
        
        # 记录筛选历史
        self.filter_history.append({
            'logic': self.logic,
            'original_count': len(factors),
            'final_count': len(final_factors),
            'filter_results': filter_results
        })
        
        logger.info(
            f"CompositeFilter (OR): {len(factors)} -> {len(final_factors)} factors "
            f"after {len(self.filters)} filters"
        )
        
        return final_factors
    
    def add_filter(self, filter_obj: BaseFilter):
        """
        添加筛选器
        
        Parameters
        ----------
        filter_obj : BaseFilter
            要添加的筛选器
        """
        if not isinstance(filter_obj, BaseFilter):
            raise TypeError("Filter must be instance of BaseFilter")
        
        self.filters.append(filter_obj)
        logger.info(f"Added filter: {filter_obj.__class__.__name__}")
    
    def remove_filter(self, index: int):
        """
        移除筛选器
        
        Parameters
        ----------
        index : int
            要移除的筛选器索引
        """
        if 0 <= index < len(self.filters):
            removed = self.filters.pop(index)
            logger.info(f"Removed filter at index {index}: {removed.__class__.__name__}")
        else:
            raise IndexError(f"Filter index {index} out of range")
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        获取筛选器摘要信息
        
        Returns
        -------
        Dict[str, Any]
            筛选器摘要
        """
        summary = {
            'filter_type': 'composite',
            'logic': self.logic,
            'num_filters': len(self.filters),
            'sub_filters': [
                {
                    'index': i,
                    'type': filter_obj.__class__.__name__,
                    'config': getattr(filter_obj, 'config', {})
                }
                for i, filter_obj in enumerate(self.filters)
            ]
        }
        
        if self.filter_history:
            recent = self.filter_history[-1]
            summary['last_filtering'] = {
                'original_count': recent['original_count'],
                'final_count': recent['final_count'],
                'reduction_rate': 1 - (recent['final_count'] / recent['original_count'])
                if recent['original_count'] > 0 else 0,
                'filter_details': recent['filter_results']
            }
        
        return summary
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        获取详细的筛选结果
        
        Returns
        -------
        Dict[str, Any]
            详细筛选结果
        """
        if not self.filter_history:
            return {'message': 'No filtering history available'}
        
        recent = self.filter_history[-1]
        
        result = {
            'logic': recent['logic'],
            'original_count': recent['original_count'],
            'final_count': recent['final_count'],
            'overall_reduction_rate': 1 - (recent['final_count'] / recent['original_count'])
            if recent['original_count'] > 0 else 0
        }
        
        if self.logic == 'AND':
            # AND逻辑：显示每个筛选器的逐步过滤结果
            result['step_by_step'] = []
            for filter_result in recent['filter_results']:
                step = {
                    'filter': filter_result['filter_type'],
                    'input_count': filter_result['input_count'],
                    'output_count': filter_result['output_count'],
                    'removed_count': filter_result['input_count'] - filter_result['output_count'],
                    'removal_rate': (filter_result['input_count'] - filter_result['output_count']) / 
                                  filter_result['input_count'] if filter_result['input_count'] > 0 else 0
                }
                result['step_by_step'].append(step)
        
        else:  # OR逻辑
            # OR逻辑：显示每个筛选器通过的因子数
            result['filter_contributions'] = []
            for filter_result in recent['filter_results']:
                contribution = {
                    'filter': filter_result['filter_type'],
                    'passed_count': filter_result['output_count'],
                    'pass_rate': filter_result['output_count'] / recent['original_count']
                    if recent['original_count'] > 0 else 0
                }
                result['filter_contributions'].append(contribution)
        
        return result