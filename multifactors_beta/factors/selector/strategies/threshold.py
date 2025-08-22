"""
阈值选择策略

基于阈值条件选择因子
"""

from typing import Dict, Optional, Any, List, Union
import pandas as pd
import numpy as np
import logging

from ..base.selector_base import SelectorBase

logger = logging.getLogger(__name__)


class ThresholdSelector(SelectorBase):
    """
    阈值选择器
    
    基于多个阈值条件选择因子
    """
    
    def __init__(self,
                 thresholds: Dict[str, float],
                 logic: str = 'AND',
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化阈值选择器
        
        Parameters
        ----------
        thresholds : Dict[str, float]
            阈值条件，如 {'total_score': 70, 'ic_mean': 0.02}
        logic : str
            逻辑组合：'AND', 'OR'
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        self.thresholds = thresholds or {}
        self.logic = logic.upper()
        
        if self.logic not in ['AND', 'OR']:
            raise ValueError(f"Invalid logic: {logic}. Must be 'AND' or 'OR'")
        
        if not self.thresholds:
            logger.warning("No thresholds provided")
        
        logger.info(
            f"Initialized ThresholdSelector: {len(self.thresholds)} thresholds, "
            f"logic={self.logic}"
        )
    
    def select(self,
               factors_pool: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               constraints: Optional[Dict] = None,
               **kwargs) -> Dict[str, Any]:
        """
        基于阈值选择因子
        
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
            logger.warning("No evaluation results provided for threshold selection")
            return self._build_empty_result(factors_pool)
        
        # 计算因子得分
        scores = self.score_factors(factors_pool, evaluation_results, **kwargs)
        
        # 应用阈值选择
        selected_names = self._apply_thresholds(
            factors_pool, evaluation_results, constraints
        )
        
        # 构建结果
        result = self._build_result(
            selected_names, factors_pool, scores, evaluation_results
        )
        
        # 记录历史
        self.save_selection_history(
            selected_names, scores, 'threshold',
            {'thresholds': self.thresholds, 'logic': self.logic}
        )
        
        return result
    
    def score_factors(self,
                      factors_pool: Dict[str, pd.Series],
                      evaluation_results: Optional[Dict] = None,
                      **kwargs) -> Dict[str, float]:
        """
        为因子打分（基于是否通过阈值）
        
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
            因子得分字典（通过阈值的为1，否则为0）
        """
        if not evaluation_results:
            return {name: 0.0 for name in factors_pool.keys()}
        
        scores = {}
        
        for name in factors_pool.keys():
            if name not in evaluation_results:
                scores[name] = 0.0
                continue
            
            result = evaluation_results[name]
            
            # 检查是否通过所有阈值
            passes_thresholds = self._check_thresholds(result)
            scores[name] = 1.0 if passes_thresholds else 0.0
        
        return scores
    
    def _apply_thresholds(self,
                          factors_pool: Dict[str, pd.Series],
                          evaluation_results: Dict,
                          constraints: Optional[Dict] = None) -> List[str]:
        """
        应用阈值条件选择因子
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict
            评估结果
        constraints : Dict, optional
            约束条件
            
        Returns
        -------
        List[str]
            选中的因子名称
        """
        selected = []
        rejection_details = {}
        
        for name in factors_pool.keys():
            if name not in evaluation_results:
                rejection_details[name] = "No evaluation result"
                continue
            
            result = evaluation_results[name]
            
            # 检查阈值条件
            passes_thresholds, failed_conditions = self._check_thresholds_detailed(result)
            
            if passes_thresholds:
                selected.append(name)
            else:
                rejection_details[name] = f"Failed conditions: {failed_conditions}"
        
        # 应用额外约束
        if constraints:
            selected = self._apply_constraints(selected, constraints, evaluation_results)
        
        logger.info(
            f"Threshold selection: {len(factors_pool)} -> {len(selected)} factors, "
            f"rejected {len(factors_pool) - len(selected)}"
        )
        
        if rejection_details:
            logger.debug(f"Rejection details: {rejection_details}")
        
        return selected
    
    def _check_thresholds(self, result: Any) -> bool:
        """
        检查是否通过阈值条件
        
        Parameters
        ----------
        result : Any
            评估结果对象
            
        Returns
        -------
        bool
            是否通过所有阈值
        """
        passes_conditions = []
        
        for metric_name, threshold in self.thresholds.items():
            metric_value = self._extract_metric(result, metric_name)
            
            if metric_value is not None:
                # 根据指标类型决定比较方向
                if self._is_higher_better(metric_name):
                    passes = metric_value >= threshold
                else:
                    passes = metric_value <= threshold
                
                passes_conditions.append(passes)
            else:
                # 如果无法获取指标值，视为不通过
                passes_conditions.append(False)
        
        # 根据逻辑组合结果
        if self.logic == 'AND':
            return all(passes_conditions) if passes_conditions else False
        else:  # OR
            return any(passes_conditions) if passes_conditions else False
    
    def _check_thresholds_detailed(self, result: Any) -> tuple:
        """
        详细检查阈值条件
        
        Parameters
        ----------
        result : Any
            评估结果对象
            
        Returns
        -------
        tuple
            (是否通过, 失败的条件列表)
        """
        passes_conditions = []
        failed_conditions = []
        
        for metric_name, threshold in self.thresholds.items():
            metric_value = self._extract_metric(result, metric_name)
            
            if metric_value is not None:
                if self._is_higher_better(metric_name):
                    passes = metric_value >= threshold
                    if not passes:
                        failed_conditions.append(f"{metric_name} {metric_value:.4f} < {threshold}")
                else:
                    passes = metric_value <= threshold
                    if not passes:
                        failed_conditions.append(f"{metric_name} {metric_value:.4f} > {threshold}")
                
                passes_conditions.append(passes)
            else:
                failed_conditions.append(f"{metric_name} not available")
                passes_conditions.append(False)
        
        # 根据逻辑组合结果
        if self.logic == 'AND':
            overall_pass = all(passes_conditions) if passes_conditions else False
        else:  # OR
            overall_pass = any(passes_conditions) if passes_conditions else False
        
        return overall_pass, failed_conditions
    
    def _is_higher_better(self, metric_name: str) -> bool:
        """
        判断指标是否越高越好
        
        Parameters
        ----------
        metric_name : str
            指标名称
            
        Returns
        -------
        bool
            True表示越高越好，False表示越低越好
        """
        higher_better_metrics = {
            'total_score', 'ic_mean', 'icir', 'sharpe_ratio',
            'stability_score', 'positive_periods_ratio'
        }
        
        lower_better_metrics = {
            'ic_volatility', 'max_drawdown', 'decay_rate'
        }
        
        # 处理绝对值指标
        if 'abs_' in metric_name:
            base_metric = metric_name.replace('abs_', '')
            return base_metric in higher_better_metrics
        
        if metric_name in higher_better_metrics:
            return True
        elif metric_name in lower_better_metrics:
            return False
        else:
            # 默认认为越高越好
            logger.debug(f"Unknown metric direction for {metric_name}, assuming higher is better")
            return True
    
    def _extract_metric(self, result: Any, metric_name: str) -> Optional[float]:
        """
        从评估结果中提取指标值
        
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
            
            # 尝试从stability_result获取
            if hasattr(result, 'stability_result') and result.stability_result:
                if hasattr(result.stability_result, metric_name):
                    value = getattr(result.stability_result, metric_name)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        return float(value)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract {metric_name}: {e}")
            return None
    
    def _apply_constraints(self,
                           selected_factors: List[str],
                           constraints: Dict,
                           evaluation_results: Dict) -> List[str]:
        """
        应用额外约束
        
        Parameters
        ----------
        selected_factors : List[str]
            已选中的因子
        constraints : Dict
            约束条件
        evaluation_results : Dict
            评估结果
            
        Returns
        -------
        List[str]
            约束后的因子列表
        """
        filtered = selected_factors.copy()
        
        # 最大因子数约束
        max_factors = constraints.get('max_factors')
        if max_factors is not None and len(filtered) > max_factors:
            # 按总分排序，选择最高的
            factor_scores = []
            for name in filtered:
                if name in evaluation_results:
                    result = evaluation_results[name]
                    score = self._extract_metric(result, 'total_score') or 0
                    factor_scores.append((score, name))
            
            factor_scores.sort(reverse=True)
            filtered = [name for _, name in factor_scores[:max_factors]]
        
        # 黑名单约束
        blacklist = constraints.get('blacklist', [])
        if blacklist:
            filtered = [name for name in filtered if name not in blacklist]
        
        # 白名单约束
        whitelist = constraints.get('whitelist')
        if whitelist:
            filtered = [name for name in filtered if name in whitelist]
        
        return filtered
    
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
        selection_reasons = {}
        for name in selected_names:
            if name in evaluation_results:
                result = evaluation_results[name]
                passed_conditions = []
                for metric_name, threshold in self.thresholds.items():
                    value = self._extract_metric(result, metric_name)
                    if value is not None:
                        passed_conditions.append(f"{metric_name}={value:.4f}")
                
                reason = f"Passed {self.logic} conditions: {', '.join(passed_conditions)}"
                selection_reasons[name] = reason
            else:
                selection_reasons[name] = "Selected by threshold"
        
        # 未选中的因子及原因
        rejected_factors = {}
        for name in factors_pool.keys():
            if name not in selected_names:
                if name in evaluation_results:
                    result = evaluation_results[name]
                    _, failed_conditions = self._check_thresholds_detailed(result)
                    reason = f"Failed {self.logic} conditions: {'; '.join(failed_conditions)}"
                else:
                    reason = "No evaluation result"
                rejected_factors[name] = reason
        
        result = {
            'selected_factors': selected_names,
            'factors_data': selected_factors,
            'selection_scores': selection_scores,
            'selection_reasons': selection_reasons,
            'rejected_factors': rejected_factors,
            'selection_method': 'threshold',
            'selection_params': {
                'thresholds': self.thresholds,
                'logic': self.logic
            },
            'summary': {
                'total_candidates': len(factors_pool),
                'selected_count': len(selected_names),
                'pass_rate': len(selected_names) / len(factors_pool) if factors_pool else 0,
                'thresholds_applied': len(self.thresholds)
            }
        }
        
        return result
    
    def _build_empty_result(self, factors_pool: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        构建空选择结果
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
            
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
            'rejected_factors': {name: "No evaluation results" for name in factors_pool.keys()},
            'selection_method': 'threshold',
            'selection_params': {
                'thresholds': self.thresholds,
                'logic': self.logic
            },
            'summary': {
                'total_candidates': len(factors_pool),
                'selected_count': 0,
                'pass_rate': 0,
                'thresholds_applied': len(self.thresholds)
            }
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float], logic: Optional[str] = None):
        """
        更新阈值条件
        
        Parameters
        ----------
        new_thresholds : Dict[str, float]
            新的阈值条件
        logic : str, optional
            新的逻辑组合方式
        """
        self.thresholds.update(new_thresholds)
        
        if logic is not None:
            if logic.upper() not in ['AND', 'OR']:
                raise ValueError(f"Invalid logic: {logic}. Must be 'AND' or 'OR'")
            self.logic = logic.upper()
        
        logger.info(f"Updated thresholds: {self.thresholds}, logic: {self.logic}")
    
    def add_threshold(self, metric_name: str, threshold: float):
        """
        添加阈值条件
        
        Parameters
        ----------
        metric_name : str
            指标名称
        threshold : float
            阈值
        """
        self.thresholds[metric_name] = threshold
        logger.info(f"Added threshold: {metric_name} = {threshold}")
    
    def remove_threshold(self, metric_name: str):
        """
        移除阈值条件
        
        Parameters
        ----------
        metric_name : str
            要移除的指标名称
        """
        if metric_name in self.thresholds:
            del self.thresholds[metric_name]
            logger.info(f"Removed threshold: {metric_name}")
        else:
            logger.warning(f"Threshold {metric_name} not found")