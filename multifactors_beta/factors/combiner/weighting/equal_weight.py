"""
等权重计算器
"""

from typing import Dict, Optional, Any
import pandas as pd
import logging

from .base_weight import BaseWeightCalculator

logger = logging.getLogger(__name__)


class EqualWeightCalculator(BaseWeightCalculator):
    """
    等权重计算器
    
    为所有因子分配相同的权重
    """
    
    def calculate(self,
                 factors: Dict[str, pd.Series],
                 evaluation_results: Optional[Dict] = None,
                 **kwargs) -> Dict[str, float]:
        """
        计算等权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果（不使用）
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            等权重字典
        """
        self.validate_inputs(factors, evaluation_results)
        
        n_factors = len(factors)
        if n_factors == 0:
            return {}
        
        # 计算等权重
        equal_weight = 1.0 / n_factors
        weights = {name: equal_weight for name in factors.keys()}
        
        # 应用约束
        weights = self.apply_constraints(weights)
        
        logger.info(f"Calculated equal weights for {n_factors} factors")
        return weights