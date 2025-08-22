"""
性能筛选器

基于因子性能指标进行筛选
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class PerformanceFilter(BaseFilter):
    """
    性能筛选器
    
    基于因子的性能指标（IC、IR、总分等）进行筛选
    """
    
    def __init__(self,
                 min_score: Optional[float] = None,
                 min_ic: Optional[float] = None,
                 min_ir: Optional[float] = None,
                 min_sharpe: Optional[float] = None,
                 max_drawdown: Optional[float] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化性能筛选器
        
        Parameters
        ----------
        min_score : float, optional
            最低总分要求
        min_ic : float, optional
            最低IC要求
        min_ir : float, optional
            最低IR要求
        min_sharpe : float, optional
            最低夏普比率要求
        max_drawdown : float, optional
            最大回撤要求
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        # 从参数或配置中获取筛选条件
        self.min_score = min_score or self.config.get('min_score', 60.0)
        self.min_ic = min_ic or self.config.get('min_ic', 0.02)
        self.min_ir = min_ir or self.config.get('min_ir', 0.3)
        self.min_sharpe = min_sharpe or self.config.get('min_sharpe', 0.5)
        self.max_drawdown = max_drawdown or self.config.get('max_drawdown', 0.3)
        
        # 是否要求所有条件都满足
        self.require_all = self.config.get('require_all', False)
    
    def filter(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               **kwargs) -> Dict[str, pd.Series]:
        """
        基于性能指标筛选因子
        
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
            
            # 检查各项指标
            passed_checks = []
            
            # 检查总分
            if self.min_score is not None:
                score = self._extract_metric(result, 'total_score')
                if score is not None:
                    passed = score >= self.min_score
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"total_score {score:.2f} < {self.min_score}")
            
            # 检查IC
            if self.min_ic is not None:
                ic = self._extract_metric(result, 'ic_mean')
                if ic is not None:
                    passed = abs(ic) >= self.min_ic  # 使用绝对值
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"|ic_mean| {abs(ic):.4f} < {self.min_ic}")
            
            # 检查IR
            if self.min_ir is not None:
                ir = self._extract_metric(result, 'icir')
                if ir is not None:
                    passed = abs(ir) >= self.min_ir  # 使用绝对值
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"|icir| {abs(ir):.4f} < {self.min_ir}")
            
            # 检查夏普比率
            if self.min_sharpe is not None:
                sharpe = self._extract_metric(result, 'sharpe_ratio')
                if sharpe is not None:
                    passed = sharpe >= self.min_sharpe
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"sharpe_ratio {sharpe:.4f} < {self.min_sharpe}")
            
            # 检查最大回撤
            if self.max_drawdown is not None:
                drawdown = self._extract_metric(result, 'max_drawdown')
                if drawdown is not None:
                    passed = abs(drawdown) <= self.max_drawdown
                    passed_checks.append(passed)
                    if not passed:
                        reasons.append(f"|max_drawdown| {abs(drawdown):.4f} > {self.max_drawdown}")
            
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
            f"Performance filter: {len(factors)} -> {len(filtered)} factors "
            f"(rejected {rejected_count})"
        )
        
        if rejected_count > 0:
            logger.debug(f"Rejection reasons: {rejection_reasons}")
        
        return filtered
    
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
            # 尝试直接属性访问
            if hasattr(result, metric_name):
                value = getattr(result, metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 尝试从metrics字典获取
            if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                value = result.metrics.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 尝试字典访问
            if isinstance(result, dict):
                value = result.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract {metric_name}: {e}")
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
            'filter_type': 'performance',
            'criteria': {
                'min_score': self.min_score,
                'min_ic': self.min_ic,
                'min_ir': self.min_ir,
                'min_sharpe': self.min_sharpe,
                'max_drawdown': self.max_drawdown
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