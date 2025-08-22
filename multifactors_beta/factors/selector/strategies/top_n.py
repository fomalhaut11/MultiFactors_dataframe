"""
TopN选择策略

选择评分最高的N个因子
"""

from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
import numpy as np
import logging

from ..base.selector_base import SelectorBase

logger = logging.getLogger(__name__)


class TopNSelector(SelectorBase):
    """
    TopN选择器
    
    根据评分选择排名前N的因子
    """
    
    def __init__(self,
                 n_factors: int = 10,
                 score_metric: str = 'total_score',
                 tie_breaker: str = 'ic_mean',
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化TopN选择器
        
        Parameters
        ----------
        n_factors : int
            选择的因子数量
        score_metric : str
            主要评分指标
        tie_breaker : str
            并列时的决胜指标
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        self.n_factors = n_factors
        self.score_metric = score_metric
        self.tie_breaker = tie_breaker
        
        # 验证参数
        if self.n_factors <= 0:
            raise ValueError("n_factors must be positive")
        
        logger.info(
            f"Initialized TopNSelector: n_factors={n_factors}, "
            f"score_metric={score_metric}, tie_breaker={tie_breaker}"
        )
    
    def select(self,
               factors_pool: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               constraints: Optional[Dict] = None,
               **kwargs) -> Dict[str, Any]:
        """
        选择TopN因子
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict, optional
            评估结果
        constraints : Dict, optional
            选择约束
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        self.validate_factors(factors_pool)
        
        if not evaluation_results:
            # 如果没有评估结果，随机选择
            logger.warning("No evaluation results provided, selecting randomly")
            factor_names = list(factors_pool.keys())
            selected_names = factor_names[:min(self.n_factors, len(factor_names))]
            scores = {name: 1.0 for name in selected_names}
        else:
            # 计算因子得分
            scores = self.score_factors(factors_pool, evaluation_results, **kwargs)
            
            # 选择TopN
            selected_names = self._select_top_n(scores, constraints)
        
        # 构建结果
        result = self._build_result(
            selected_names, factors_pool, scores, evaluation_results
        )
        
        # 记录历史
        self.save_selection_history(
            selected_names, scores, 'top_n',
            {'n_factors': self.n_factors, 'score_metric': self.score_metric}
        )
        
        return result
    
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
        if not evaluation_results:
            return {name: 1.0 for name in factors_pool.keys()}
        
        scores = {}
        
        for name in factors_pool.keys():
            if name not in evaluation_results:
                scores[name] = 0.0
                continue
            
            result = evaluation_results[name]
            
            # 提取主要得分
            primary_score = self._extract_score(result, self.score_metric)
            if primary_score is None:
                primary_score = 0.0
            
            scores[name] = primary_score
        
        return scores
    
    def _select_top_n(self,
                      scores: Dict[str, float],
                      constraints: Optional[Dict] = None) -> List[str]:
        """
        选择TopN因子
        
        Parameters
        ----------
        scores : Dict[str, float]
            因子得分
        constraints : Dict, optional
            约束条件
            
        Returns
        -------
        List[str]
            选中的因子名称
        """
        if not scores:
            return []
        
        # 应用约束
        filtered_scores = self._apply_constraints(scores, constraints)
        
        if not filtered_scores:
            logger.warning("No factors pass constraints")
            return []
        
        # 排序选择
        sorted_factors = self._sort_factors(filtered_scores)
        
        # 选择前N个
        n_to_select = min(self.n_factors, len(sorted_factors))
        selected = sorted_factors[:n_to_select]
        
        logger.info(f"Selected top {len(selected)} factors from {len(filtered_scores)} candidates")
        
        return selected
    
    def _sort_factors(self, scores: Dict[str, float]) -> List[str]:
        """
        按得分排序因子
        
        Parameters
        ----------
        scores : Dict[str, float]
            因子得分
            
        Returns
        -------
        List[str]
            排序后的因子名称列表
        """
        # 基本排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 处理并列情况
        if self.tie_breaker and hasattr(self, '_evaluation_results'):
            sorted_items = self._resolve_ties(sorted_items)
        
        return [name for name, _ in sorted_items]
    
    def _resolve_ties(self, sorted_items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        解决并列问题
        
        Parameters
        ----------
        sorted_items : List[Tuple[str, float]]
            排序的(因子名, 得分)列表
            
        Returns
        -------
        List[Tuple[str, float]]
            解决并列后的列表
        """
        if not hasattr(self, '_evaluation_results') or not self._evaluation_results:
            return sorted_items
        
        resolved = []
        i = 0
        
        while i < len(sorted_items):
            # 找到所有同分的因子
            current_score = sorted_items[i][1]
            tied_factors = []
            j = i
            
            while j < len(sorted_items) and sorted_items[j][1] == current_score:
                tied_factors.append(sorted_items[j])
                j += 1
            
            if len(tied_factors) > 1:
                # 使用决胜指标重新排序
                tie_breaker_scores = {}
                for name, _ in tied_factors:
                    if name in self._evaluation_results:
                        tie_score = self._extract_score(
                            self._evaluation_results[name], 
                            self.tie_breaker
                        )
                        tie_breaker_scores[name] = tie_score or 0.0
                    else:
                        tie_breaker_scores[name] = 0.0
                
                # 按决胜指标排序
                tied_sorted = sorted(
                    tied_factors,
                    key=lambda x: tie_breaker_scores.get(x[0], 0.0),
                    reverse=True
                )
                resolved.extend(tied_sorted)
            else:
                resolved.extend(tied_factors)
            
            i = j
        
        return resolved
    
    def _apply_constraints(self,
                           scores: Dict[str, float],
                           constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        应用选择约束
        
        Parameters
        ----------
        scores : Dict[str, float]
            原始得分
        constraints : Dict, optional
            约束条件
            
        Returns
        -------
        Dict[str, float]
            满足约束的得分
        """
        if not constraints:
            return scores
        
        filtered = scores.copy()
        
        # 最小得分约束
        min_score = constraints.get('min_score')
        if min_score is not None:
            filtered = {
                name: score for name, score in filtered.items()
                if score >= min_score
            }
        
        # 黑名单约束
        blacklist = constraints.get('blacklist', [])
        if blacklist:
            filtered = {
                name: score for name, score in filtered.items()
                if name not in blacklist
            }
        
        # 白名单约束
        whitelist = constraints.get('whitelist')
        if whitelist:
            filtered = {
                name: score for name, score in filtered.items()
                if name in whitelist
            }
        
        return filtered
    
    def _extract_score(self, result: Any, metric_name: str) -> Optional[float]:
        """
        从评估结果中提取得分
        
        Parameters
        ----------
        result : Any
            评估结果对象
        metric_name : str
            指标名称
            
        Returns
        -------
        float or None
            得分值
        """
        try:
            # 直接属性访问
            if hasattr(result, metric_name):
                value = getattr(result, metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 从metrics字典获取
            if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                value = result.metrics.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 字典访问
            if isinstance(result, dict):
                value = result.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract {metric_name}: {e}")
            return None
    
    def _build_result(self,
                      selected_names: List[str],
                      factors_pool: Dict[str, pd.Series],
                      scores: Dict[str, float],
                      evaluation_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        构建选择结果
        
        Parameters
        ----------
        selected_names : List[str]
            选中的因子名称
        factors_pool : Dict[str, pd.Series]
            因子池
        scores : Dict[str, float]
            因子得分
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        # 选中的因子数据
        selected_factors = {
            name: factors_pool[name] for name in selected_names
            if name in factors_pool
        }
        
        # 选择得分
        selection_scores = {
            name: scores.get(name, 0.0) for name in selected_names
        }
        
        # 选择原因
        selection_reasons = {
            name: f"Ranked #{i+1} with {self.score_metric}={scores.get(name, 0.0):.4f}"
            for i, name in enumerate(selected_names)
        }
        
        # 未选中的因子及原因
        rejected_factors = {}
        all_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(all_ranked):
            if name not in selected_names:
                if i < self.n_factors:
                    reason = f"Ranked #{i+1} but filtered out"
                else:
                    reason = f"Ranked #{i+1}, below top {self.n_factors}"
                rejected_factors[name] = reason
        
        result = {
            'selected_factors': selected_names,
            'factors_data': selected_factors,
            'selection_scores': selection_scores,
            'selection_reasons': selection_reasons,
            'rejected_factors': rejected_factors,
            'selection_method': 'top_n',
            'selection_params': {
                'n_factors': self.n_factors,
                'score_metric': self.score_metric,
                'tie_breaker': self.tie_breaker
            },
            'summary': {
                'total_candidates': len(factors_pool),
                'selected_count': len(selected_names),
                'avg_score': np.mean(list(selection_scores.values())) if selection_scores else 0,
                'score_range': (
                    min(selection_scores.values()) if selection_scores else 0,
                    max(selection_scores.values()) if selection_scores else 0
                )
            }
        }
        
        return result
    
    def update_params(self,
                      n_factors: Optional[int] = None,
                      score_metric: Optional[str] = None,
                      tie_breaker: Optional[str] = None):
        """
        更新选择参数
        
        Parameters
        ----------
        n_factors : int, optional
            新的因子数量
        score_metric : str, optional
            新的评分指标
        tie_breaker : str, optional
            新的决胜指标
        """
        if n_factors is not None:
            if n_factors <= 0:
                raise ValueError("n_factors must be positive")
            self.n_factors = n_factors
        
        if score_metric is not None:
            self.score_metric = score_metric
        
        if tie_breaker is not None:
            self.tie_breaker = tie_breaker
        
        logger.info(
            f"Updated TopNSelector params: n_factors={self.n_factors}, "
            f"score_metric={self.score_metric}, tie_breaker={self.tie_breaker}"
        )