"""
IC加权计算器
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

from .base_weight import BaseWeightCalculator

logger = logging.getLogger(__name__)


class ICWeightCalculator(BaseWeightCalculator):
    """
    IC加权计算器
    
    根据因子的IC（信息系数）计算权重
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化IC加权计算器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        self.use_abs_ic = self.config.get('use_abs_ic', True)
        self.ic_lookback = self.config.get('ic_lookback', 12)  # IC回看期数
        self.decay_factor = self.config.get('decay_factor', 1.0)  # 衰减因子
        self.min_ic = self.config.get('min_ic', 0.0)  # 最小IC阈值
    
    def calculate(self,
                 factors: Dict[str, pd.Series],
                 evaluation_results: Optional[Dict] = None,
                 **kwargs) -> Dict[str, float]:
        """
        根据IC计算权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果，包含IC信息
        **kwargs : dict
            其他参数，可包含returns用于计算IC
            
        Returns
        -------
        Dict[str, float]
            IC加权的权重字典
        """
        self.validate_inputs(factors, evaluation_results)
        
        # 获取IC值
        ic_values = self._extract_ic_values(factors, evaluation_results, **kwargs)
        
        if not ic_values:
            logger.warning("No IC values available, using equal weights")
            n_factors = len(factors)
            return {name: 1.0/n_factors for name in factors.keys()}
        
        # 计算权重
        weights = self._calculate_ic_weights(ic_values)
        
        # 应用约束
        weights = self.apply_constraints(weights)
        
        logger.info(f"Calculated IC weights for {len(weights)} factors")
        return weights
    
    def _extract_ic_values(self,
                          factors: Dict[str, pd.Series],
                          evaluation_results: Optional[Dict] = None,
                          **kwargs) -> Dict[str, float]:
        """
        提取IC值
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            因子IC值字典
        """
        ic_values = {}
        
        # 优先从评估结果中获取IC
        if evaluation_results:
            for factor_name in factors.keys():
                if factor_name in evaluation_results:
                    eval_result = evaluation_results[factor_name]
                    
                    # 尝试从不同位置获取IC
                    ic = None
                    if hasattr(eval_result, 'metrics'):
                        ic = eval_result.metrics.get('ic_mean')
                    elif hasattr(eval_result, 'ic_result'):
                        ic = eval_result.ic_result.ic_mean if eval_result.ic_result else None
                    elif isinstance(eval_result, dict):
                        ic = eval_result.get('ic_mean')
                    
                    if ic is not None:
                        ic_values[factor_name] = ic
        
        # 如果提供了收益数据，计算IC
        if 'returns' in kwargs and len(ic_values) < len(factors):
            returns = kwargs['returns']
            for factor_name, factor in factors.items():
                if factor_name not in ic_values:
                    ic = self._calculate_ic(factor, returns)
                    if ic is not None:
                        ic_values[factor_name] = ic
        
        return ic_values
    
    def _calculate_ic(self, factor: pd.Series, returns: pd.Series) -> Optional[float]:
        """
        计算因子IC
        
        Parameters
        ----------
        factor : pd.Series
            因子值
        returns : pd.Series
            收益率
            
        Returns
        -------
        float or None
            IC值
        """
        try:
            # 对齐数据
            aligned = pd.DataFrame({'factor': factor, 'return': returns}).dropna()
            if len(aligned) < 10:  # 数据太少
                return None
            
            # 按日期计算IC并取平均
            ic_series = aligned.groupby(level=0).apply(
                lambda x: x['factor'].corr(x['return'], method='spearman')
            )
            
            # 只使用最近的IC
            if self.ic_lookback > 0:
                ic_series = ic_series.tail(self.ic_lookback)
            
            # 应用衰减权重
            if self.decay_factor < 1.0:
                weights = np.power(self.decay_factor, np.arange(len(ic_series)-1, -1, -1))
                ic_mean = np.average(ic_series, weights=weights)
            else:
                ic_mean = ic_series.mean()
            
            return ic_mean
            
        except Exception as e:
            logger.warning(f"Failed to calculate IC: {e}")
            return None
    
    def _calculate_ic_weights(self, ic_values: Dict[str, float]) -> Dict[str, float]:
        """
        根据IC值计算权重
        
        Parameters
        ----------
        ic_values : Dict[str, float]
            IC值字典
            
        Returns
        -------
        Dict[str, float]
            权重字典
        """
        weights = {}
        
        # 处理IC值
        processed_ic = {}
        for name, ic in ic_values.items():
            # 使用绝对值或原值
            if self.use_abs_ic:
                ic_value = abs(ic)
            else:
                ic_value = ic
            
            # 应用最小IC阈值
            if abs(ic) < self.min_ic:
                ic_value = 0
            
            processed_ic[name] = max(0, ic_value)  # 确保非负
        
        # 计算权重
        total_ic = sum(processed_ic.values())
        
        if total_ic > 0:
            weights = {name: ic/total_ic for name, ic in processed_ic.items()}
        else:
            # 如果所有IC都是0或负数，使用等权
            n = len(processed_ic)
            weights = {name: 1.0/n for name in processed_ic.keys()}
        
        return weights
    
    def _requires_evaluation(self) -> bool:
        """
        是否需要评估结果
        
        Returns
        -------
        bool
            True，IC加权需要评估结果
        """
        return True