"""
稳定性筛选器

基于因子稳定性指标进行筛选
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class StabilityFilter(BaseFilter):
    """
    稳定性筛选器
    
    基于因子的稳定性指标进行筛选，包括IC稳定性、收益稳定性等
    """
    
    def __init__(self,
                 min_stability_score: Optional[float] = None,
                 max_ic_volatility: Optional[float] = None,
                 min_positive_periods: Optional[float] = None,
                 max_decay_rate: Optional[float] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化稳定性筛选器
        
        Parameters
        ----------
        min_stability_score : float, optional
            最低稳定性得分要求
        max_ic_volatility : float, optional
            最大IC波动率要求
        min_positive_periods : float, optional
            最小正向期数占比要求
        max_decay_rate : float, optional
            最大衰减率要求
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        # 从参数或配置中获取筛选条件
        self.min_stability_score = min_stability_score or self.config.get('min_stability_score', 50.0)
        self.max_ic_volatility = max_ic_volatility or self.config.get('max_ic_volatility', 0.5)
        self.min_positive_periods = min_positive_periods or self.config.get('min_positive_periods', 0.4)
        self.max_decay_rate = max_decay_rate or self.config.get('max_decay_rate', 0.3)
        
        # 是否要求所有条件都满足
        self.require_all = self.config.get('require_all', False)
    
    def filter(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               **kwargs) -> Dict[str, pd.Series]:
        """
        基于稳定性指标筛选因子
        
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
        
        if not evaluation_results:
            logger.warning("No evaluation results provided, returning original factors")
            return factors
        
        filtered = {}
        rejected_count = 0
        rejection_reasons = {}
        
        for name, factor in factors.items():
            if name not in evaluation_results:
                logger.debug(f"No evaluation result for factor {name}, skipping")
                continue
            
            result = evaluation_results[name]
            reasons = []
            passed_checks = []
            
            # 检查稳定性得分
            if self.min_stability_score is not None:
                stability_score = self._extract_stability_metric(result, 'stability_score')
                if stability_score is not None:
                    passed = stability_score >= self.min_stability_score
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"stability_score {stability_score:.2f} < {self.min_stability_score}")
            
            # 检查IC波动率
            if self.max_ic_volatility is not None:
                ic_volatility = self._extract_stability_metric(result, 'ic_volatility')
                if ic_volatility is not None:
                    passed = ic_volatility <= self.max_ic_volatility
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"ic_volatility {ic_volatility:.4f} > {self.max_ic_volatility}")
            
            # 检查正向期数占比
            if self.min_positive_periods is not None:
                positive_ratio = self._extract_stability_metric(result, 'positive_periods_ratio')
                if positive_ratio is not None:
                    passed = positive_ratio >= self.min_positive_periods
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"positive_periods_ratio {positive_ratio:.4f} < {self.min_positive_periods}")
            
            # 检查衰减率
            if self.max_decay_rate is not None:
                decay_rate = self._extract_stability_metric(result, 'decay_rate')
                if decay_rate is not None:
                    passed = abs(decay_rate) <= self.max_decay_rate
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"|decay_rate| {abs(decay_rate):.4f} > {self.max_decay_rate}")
            
            # 决定是否通过筛选
            if not passed_checks:
                # 如果没有检查项，默认通过
                filtered[name] = factor
            elif self.require_all:
                # 要求所有条件都满足
                if all(passed_checks):
                    filtered[name] = factor
                else:
                    rejected_count += 1
                    rejection_reasons[name] = "; ".join(reasons)
            else:
                # 满足任一条件即可
                if any(passed_checks):
                    filtered[name] = factor
                else:
                    rejected_count += 1
                    rejection_reasons[name] = "; ".join(reasons)
        
        # 记录筛选结果
        self.filter_history.append({
            'original_count': len(factors),
            'filtered_count': len(filtered),
            'rejected_count': rejected_count,
            'rejection_reasons': rejection_reasons
        })
        
        logger.info(
            f"Stability filter: {len(factors)} -> {len(filtered)} factors "
            f"(rejected {rejected_count})"
        )
        
        if rejected_count > 0:
            logger.debug(f"Rejection reasons: {rejection_reasons}")
        
        return filtered
    
    def _extract_stability_metric(self, result: Any, metric_name: str) -> Optional[float]:
        """
        从评估结果中提取稳定性指标
        
        Parameters
        ----------
        result : Any
            评估结果对象
        metric_name : str
            指标名称
            
        Returns
        -------
        float or None
            指标值
        """
        try:
            # 尝试从stability_result中获取
            if hasattr(result, 'stability_result') and result.stability_result:
                stability_result = result.stability_result
                
                # 直接属性访问
                if hasattr(stability_result, metric_name):
                    value = getattr(stability_result, metric_name)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        return float(value)
                
                # 从metrics字典获取
                if hasattr(stability_result, 'metrics') and isinstance(stability_result.metrics, dict):
                    value = stability_result.metrics.get(metric_name)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        return float(value)
            
            # 尝试从主result的metrics获取
            if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                value = result.metrics.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 尝试直接属性访问
            if hasattr(result, metric_name):
                value = getattr(result, metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 尝试字典访问
            if isinstance(result, dict):
                value = result.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 特殊处理：从IC时间序列计算稳定性指标
            if metric_name == 'ic_volatility':
                return self._calculate_ic_volatility(result)
            elif metric_name == 'positive_periods_ratio':
                return self._calculate_positive_periods_ratio(result)
            elif metric_name == 'stability_score':
                return self._calculate_stability_score(result)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract {metric_name}: {e}")
            return None
    
    def _calculate_ic_volatility(self, result: Any) -> Optional[float]:
        """
        计算IC波动率
        
        Parameters
        ----------
        result : Any
            评估结果
            
        Returns
        -------
        float or None
            IC波动率
        """
        try:
            # 尝试获取IC时间序列
            ic_series = None
            
            if hasattr(result, 'ic_result') and result.ic_result:
                if hasattr(result.ic_result, 'ic_series'):
                    ic_series = result.ic_result.ic_series
            
            if ic_series is not None and len(ic_series) > 1:
                return float(ic_series.std())
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to calculate IC volatility: {e}")
            return None
    
    def _calculate_positive_periods_ratio(self, result: Any) -> Optional[float]:
        """
        计算正向期数占比
        
        Parameters
        ----------
        result : Any
            评估结果
            
        Returns
        -------
        float or None
            正向期数占比
        """
        try:
            # 尝试获取IC时间序列
            ic_series = None
            
            if hasattr(result, 'ic_result') and result.ic_result:
                if hasattr(result.ic_result, 'ic_series'):
                    ic_series = result.ic_result.ic_series
            
            if ic_series is not None and len(ic_series) > 0:
                positive_count = (ic_series > 0).sum()
                return float(positive_count / len(ic_series))
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to calculate positive periods ratio: {e}")
            return None
    
    def _calculate_stability_score(self, result: Any) -> Optional[float]:
        """
        计算综合稳定性得分
        
        Parameters
        ----------
        result : Any
            评估结果
            
        Returns
        -------
        float or None
            稳定性得分
        """
        try:
            # 如果已有稳定性维度分数，直接使用
            if hasattr(result, 'dimension_scores') and isinstance(result.dimension_scores, dict):
                stability_score = result.dimension_scores.get('stability')
                if isinstance(stability_score, (int, float)) and np.isfinite(stability_score):
                    return float(stability_score)
            
            # 否则基于IC统计计算
            ic_volatility = self._calculate_ic_volatility(result)
            positive_ratio = self._calculate_positive_periods_ratio(result)
            
            if ic_volatility is not None and positive_ratio is not None:
                # 简单的稳定性评分公式
                volatility_score = max(0, 100 * (1 - min(ic_volatility, 1.0)))
                consistency_score = 100 * positive_ratio
                
                stability_score = 0.5 * volatility_score + 0.5 * consistency_score
                return float(stability_score)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to calculate stability score: {e}")
            return None
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        获取筛选器摘要信息
        
        Returns
        -------
        Dict[str, Any]
            筛选器摘要
        """
        summary = {
            'filter_type': 'stability',
            'criteria': {
                'min_stability_score': self.min_stability_score,
                'max_ic_volatility': self.max_ic_volatility,
                'min_positive_periods': self.min_positive_periods,
                'max_decay_rate': self.max_decay_rate
            },
            'require_all': self.require_all
        }
        
        if self.filter_history:
            recent = self.filter_history[-1]
            summary['last_filtering'] = {
                'original_count': recent['original_count'],
                'filtered_count': recent['filtered_count'],
                'rejection_rate': recent['rejected_count'] / recent['original_count']
                if recent['original_count'] > 0 else 0
            }
        
        return summary