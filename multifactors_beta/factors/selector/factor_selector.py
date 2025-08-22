"""
因子选择器主类

整合各种选择策略和筛选器
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .base.selector_base import SelectorBase
from .filters import (
    PerformanceFilter,
    CorrelationFilter,
    StabilityFilter,
    CompositeFilter
)
from .strategies import TopNSelector

logger = logging.getLogger(__name__)


class FactorSelector:
    """
    因子选择器主类
    
    提供完整的因子选择功能，包括筛选和策略选择
    """
    
    def __init__(self,
                 method: str = 'top_n',
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化因子选择器
        
        Parameters
        ----------
        method : str
            选择方法：'top_n', 'threshold', 'clustering'
        config : Dict[str, Any], optional
            配置参数
        """
        self.method = method
        self.config = config or {}
        self.filters = []
        self.selection_history = []
        
        # 初始化选择器组件
        self._init_selector()
        
        logger.info(f"Initialized FactorSelector with method: {method}")
    
    def _init_selector(self):
        """初始化选择器组件"""
        # 创建策略选择器
        if self.method == 'top_n':
            self.strategy_selector = TopNSelector(
                n_factors=self.config.get('n_factors', 10),
                score_metric=self.config.get('score_metric', 'total_score'),
                tie_breaker=self.config.get('tie_breaker', 'ic_mean'),
                config=self.config
            )
        else:
            raise ValueError(f"Unsupported selection method: {self.method}")
        
        # 初始化默认筛选器
        self._init_default_filters()
    
    def _init_default_filters(self):
        """初始化默认筛选器"""
        filter_config = self.config.get('filters', {})
        
        # 性能筛选器
        if filter_config.get('use_performance_filter', True):
            perf_config = filter_config.get('performance', {})
            self.filters.append(PerformanceFilter(
                min_score=perf_config.get('min_score', 50.0),
                min_ic=perf_config.get('min_ic', 0.01),
                min_ir=perf_config.get('min_ir', 0.2),
                config=perf_config
            ))
        
        # 稳定性筛选器
        if filter_config.get('use_stability_filter', True):
            stab_config = filter_config.get('stability', {})
            self.filters.append(StabilityFilter(
                min_stability_score=stab_config.get('min_stability_score', 40.0),
                max_ic_volatility=stab_config.get('max_ic_volatility', 0.6),
                config=stab_config
            ))
        
        # 相关性筛选器
        if filter_config.get('use_correlation_filter', True):
            corr_config = filter_config.get('correlation', {})
            self.filters.append(CorrelationFilter(
                max_correlation=corr_config.get('max_correlation', 0.7),
                method=corr_config.get('method', 'hierarchical'),
                config=corr_config
            ))
    
    def select(self,
               factors_pool: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               test_results: Optional[Dict] = None,
               correlation_matrix: Optional[pd.DataFrame] = None,
               custom_filters: Optional[List] = None,
               **kwargs) -> Dict[str, Any]:
        """
        选择最优因子组合
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            候选因子池，key为因子名，value为MultiIndex Series
        evaluation_results : Dict, optional
            因子评估结果
        test_results : Dict, optional
            因子测试结果
        correlation_matrix : pd.DataFrame, optional
            因子相关性矩阵
        custom_filters : List, optional
            自定义筛选器列表
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        start_time = datetime.now()
        
        # 验证输入
        if not factors_pool:
            raise ValueError("factors_pool cannot be empty")
        
        logger.info(f"Starting factor selection from {len(factors_pool)} candidates")
        
        # 第一步：应用筛选器
        filtered_factors = self._apply_filters(
            factors_pool, evaluation_results, correlation_matrix, custom_filters
        )
        
        if not filtered_factors:
            logger.warning("No factors passed filtering stage")
            return self._build_empty_result(factors_pool)
        
        # 第二步：策略选择
        selection_result = self.strategy_selector.select(
            filtered_factors, evaluation_results, **kwargs
        )
        
        # 第三步：增强结果信息
        enhanced_result = self._enhance_result(
            selection_result, factors_pool, evaluation_results, 
            len(filtered_factors), start_time
        )
        
        # 记录选择历史
        self.selection_history.append({
            'timestamp': start_time,
            'original_count': len(factors_pool),
            'filtered_count': len(filtered_factors),
            'selected_count': len(enhanced_result['selected_factors']),
            'method': self.method,
            'config': self.config.copy()
        })
        
        elapsed = datetime.now() - start_time
        logger.info(
            f"Factor selection completed in {elapsed.total_seconds():.2f}s: "
            f"{len(factors_pool)} -> {len(filtered_factors)} -> "
            f"{len(enhanced_result['selected_factors'])} factors"
        )
        
        return enhanced_result
    
    def _apply_filters(self,
                       factors_pool: Dict[str, pd.Series],
                       evaluation_results: Optional[Dict] = None,
                       correlation_matrix: Optional[pd.DataFrame] = None,
                       custom_filters: Optional[List] = None) -> Dict[str, pd.Series]:
        """
        应用筛选器
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            原始因子池
        evaluation_results : Dict, optional
            评估结果
        correlation_matrix : pd.DataFrame, optional
            相关性矩阵
        custom_filters : List, optional
            自定义筛选器
            
        Returns
        -------
        Dict[str, pd.Series]
            筛选后的因子池
        """
        current_factors = factors_pool.copy()
        
        # 应用默认筛选器
        for filter_obj in self.filters:
            try:
                if isinstance(filter_obj, CorrelationFilter) and correlation_matrix is not None:
                    # 相关性筛选器需要相关性矩阵
                    current_factors = filter_obj.filter(
                        current_factors, evaluation_results, 
                        correlation_matrix=correlation_matrix
                    )
                else:
                    current_factors = filter_obj.filter(current_factors, evaluation_results)
                
                if not current_factors:
                    logger.warning(f"No factors remaining after {filter_obj.__class__.__name__}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in filter {filter_obj.__class__.__name__}: {e}")
                continue
        
        # 应用自定义筛选器
        if custom_filters:
            for filter_obj in custom_filters:
                try:
                    current_factors = filter_obj.filter(current_factors, evaluation_results)
                    if not current_factors:
                        logger.warning(f"No factors remaining after custom filter {filter_obj.__class__.__name__}")
                        break
                except Exception as e:
                    logger.error(f"Error in custom filter {filter_obj.__class__.__name__}: {e}")
                    continue
        
        return current_factors
    
    def _enhance_result(self,
                        selection_result: Dict[str, Any],
                        original_factors: Dict[str, pd.Series],
                        evaluation_results: Optional[Dict],
                        filtered_count: int,
                        start_time: datetime) -> Dict[str, Any]:
        """
        增强选择结果
        
        Parameters
        ----------
        selection_result : Dict[str, Any]
            原始选择结果
        original_factors : Dict[str, pd.Series]
            原始因子池
        evaluation_results : Dict, optional
            评估结果
        filtered_count : int
            筛选后因子数量
        start_time : datetime
            开始时间
            
        Returns
        -------
        Dict[str, Any]
            增强后的结果
        """
        enhanced = selection_result.copy()
        
        # 添加筛选信息
        enhanced['filtering_summary'] = {
            'original_count': len(original_factors),
            'filtered_count': filtered_count,
            'filter_reduction_rate': 1 - (filtered_count / len(original_factors))
            if len(original_factors) > 0 else 0,
            'filters_applied': [f.__class__.__name__ for f in self.filters]
        }
        
        # 添加多样性分析
        if len(enhanced['selected_factors']) > 1 and evaluation_results:
            enhanced['diversity_analysis'] = self._analyze_diversity(
                enhanced['selected_factors'], evaluation_results
            )
        
        # 添加覆盖率分析
        enhanced['coverage_analysis'] = self._analyze_coverage(
            enhanced['factors_data']
        )
        
        # 添加性能统计
        enhanced['performance_stats'] = self._calculate_performance_stats(
            enhanced['selected_factors'], evaluation_results
        )
        
        # 添加执行信息
        enhanced['execution_info'] = {
            'start_time': start_time,
            'end_time': datetime.now(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'selector_method': self.method,
            'selector_config': self.config
        }
        
        return enhanced
    
    def _analyze_diversity(self,
                           selected_factors: List[str],
                           evaluation_results: Dict) -> Dict[str, Any]:
        """
        分析选中因子的多样性
        
        Parameters
        ----------
        selected_factors : List[str]
            选中的因子
        evaluation_results : Dict
            评估结果
            
        Returns
        -------
        Dict[str, Any]
            多样性分析结果
        """
        diversity_metrics = {}
        
        try:
            # 计算得分方差
            scores = []
            for factor in selected_factors:
                if factor in evaluation_results:
                    result = evaluation_results[factor]
                    if hasattr(result, 'total_score'):
                        scores.append(result.total_score)
            
            if scores:
                diversity_metrics['score_variance'] = float(np.var(scores))
                diversity_metrics['score_range'] = float(max(scores) - min(scores))
                diversity_metrics['score_std'] = float(np.std(scores))
            
            # 计算IC方差
            ics = []
            for factor in selected_factors:
                if factor in evaluation_results:
                    result = evaluation_results[factor]
                    if hasattr(result, 'metrics') and 'ic_mean' in result.metrics:
                        ics.append(result.metrics['ic_mean'])
            
            if ics:
                diversity_metrics['ic_variance'] = float(np.var(ics))
                diversity_metrics['ic_range'] = float(max(ics) - min(ics))
        
        except Exception as e:
            logger.debug(f"Error in diversity analysis: {e}")
            diversity_metrics['error'] = str(e)
        
        return diversity_metrics
    
    def _analyze_coverage(self, factors_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        分析因子覆盖率
        
        Parameters
        ----------
        factors_data : Dict[str, pd.Series]
            因子数据
            
        Returns
        -------
        Dict[str, Any]
            覆盖率分析结果
        """
        coverage_stats = {}
        
        try:
            if not factors_data:
                return coverage_stats
            
            # 计算每个因子的覆盖率
            factor_coverage = {}
            total_observations = 0
            
            for name, factor in factors_data.items():
                non_null_count = factor.count()
                total_count = len(factor)
                coverage = non_null_count / total_count if total_count > 0 else 0
                
                factor_coverage[name] = {
                    'coverage_rate': coverage,
                    'non_null_count': non_null_count,
                    'total_count': total_count
                }
                
                total_observations = max(total_observations, total_count)
            
            coverage_stats['factor_coverage'] = factor_coverage
            coverage_stats['avg_coverage'] = np.mean([
                stats['coverage_rate'] for stats in factor_coverage.values()
            ])
            coverage_stats['min_coverage'] = min([
                stats['coverage_rate'] for stats in factor_coverage.values()
            ]) if factor_coverage else 0
            coverage_stats['total_observations'] = total_observations
        
        except Exception as e:
            logger.debug(f"Error in coverage analysis: {e}")
            coverage_stats['error'] = str(e)
        
        return coverage_stats
    
    def _calculate_performance_stats(self,
                                     selected_factors: List[str],
                                     evaluation_results: Optional[Dict]) -> Dict[str, Any]:
        """
        计算选中因子的性能统计
        
        Parameters
        ----------
        selected_factors : List[str]
            选中的因子
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        Dict[str, Any]
            性能统计结果
        """
        stats = {}
        
        if not evaluation_results:
            return stats
        
        try:
            # 收集各种指标
            total_scores = []
            ic_means = []
            ir_values = []
            
            for factor in selected_factors:
                if factor in evaluation_results:
                    result = evaluation_results[factor]
                    
                    # 总分
                    if hasattr(result, 'total_score'):
                        total_scores.append(result.total_score)
                    
                    # IC和IR
                    if hasattr(result, 'metrics'):
                        metrics = result.metrics
                        if 'ic_mean' in metrics:
                            ic_means.append(metrics['ic_mean'])
                        if 'icir' in metrics:
                            ir_values.append(metrics['icir'])
            
            # 计算统计量
            if total_scores:
                stats['total_score'] = {
                    'mean': float(np.mean(total_scores)),
                    'std': float(np.std(total_scores)),
                    'min': float(np.min(total_scores)),
                    'max': float(np.max(total_scores))
                }
            
            if ic_means:
                stats['ic_mean'] = {
                    'mean': float(np.mean(ic_means)),
                    'std': float(np.std(ic_means)),
                    'min': float(np.min(ic_means)),
                    'max': float(np.max(ic_means))
                }
            
            if ir_values:
                stats['icir'] = {
                    'mean': float(np.mean(ir_values)),
                    'std': float(np.std(ir_values)),
                    'min': float(np.min(ir_values)),
                    'max': float(np.max(ir_values))
                }
        
        except Exception as e:
            logger.debug(f"Error in performance stats calculation: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _build_empty_result(self, original_factors: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        构建空选择结果
        
        Parameters
        ----------
        original_factors : Dict[str, pd.Series]
            原始因子池
            
        Returns
        -------
        Dict[str, Any]
            空结果
        """
        return {
            'selected_factors': [],
            'factors_data': {},
            'selection_scores': {},
            'selection_reasons': {},
            'rejected_factors': {name: "Filtered out" for name in original_factors.keys()},
            'selection_method': self.method,
            'summary': {
                'total_candidates': len(original_factors),
                'selected_count': 0,
                'avg_score': 0,
                'score_range': (0, 0)
            },
            'filtering_summary': {
                'original_count': len(original_factors),
                'filtered_count': 0,
                'filter_reduction_rate': 1.0
            }
        }
    
    def add_filter(self, filter_obj):
        """
        添加筛选器
        
        Parameters
        ----------
        filter_obj : BaseFilter
            要添加的筛选器
        """
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
            logger.info(f"Removed filter: {removed.__class__.__name__}")
        else:
            raise IndexError(f"Filter index {index} out of range")
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        获取选择器摘要信息
        
        Returns
        -------
        Dict[str, Any]
            选择器摘要
        """
        return {
            'method': self.method,
            'config': self.config,
            'filters': [f.__class__.__name__ for f in self.filters],
            'selection_history_count': len(self.selection_history),
            'strategy_selector': self.strategy_selector.__class__.__name__
        }