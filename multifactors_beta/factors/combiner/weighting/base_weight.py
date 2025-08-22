"""
权重计算基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseWeightCalculator(ABC):
    """
    权重计算器基类
    
    定义了权重计算的标准接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化权重计算器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.min_weight = self.config.get('min_weight', 0.0)
        self.max_weight = self.config.get('max_weight', 1.0)
        self.normalize = self.config.get('normalize', True)
    
    @abstractmethod
    def calculate(self,
                 factors: Dict[str, pd.Series],
                 evaluation_results: Optional[Dict] = None,
                 **kwargs) -> Dict[str, float]:
        """
        计算因子权重
        
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
            因子权重
        """
        pass
    
    def apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        应用权重约束
        
        Parameters
        ----------
        weights : Dict[str, float]
            原始权重
            
        Returns
        -------
        Dict[str, float]
            约束后的权重
        """
        constrained = {}
        for name, weight in weights.items():
            constrained[name] = np.clip(weight, self.min_weight, self.max_weight)
        
        if self.normalize:
            total = sum(constrained.values())
            if total > 0:
                constrained = {k: v/total for k, v in constrained.items()}
            else:
                # 如果所有权重都是0，使用等权
                n = len(constrained)
                constrained = {k: 1.0/n for k in constrained.keys()}
        
        return constrained
    
    def validate_inputs(self,
                       factors: Dict[str, pd.Series],
                       evaluation_results: Optional[Dict] = None) -> bool:
        """
        验证输入数据
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        bool
            验证是否通过
        """
        if not factors:
            raise ValueError("Factors dictionary cannot be empty")
        
        # 如果需要评估结果但没有提供
        if self._requires_evaluation() and not evaluation_results:
            logger.warning(
                f"{self.__class__.__name__} works better with evaluation results, "
                "using default weights"
            )
        
        return True
    
    def _requires_evaluation(self) -> bool:
        """
        是否需要评估结果
        
        Returns
        -------
        bool
            是否需要评估结果
        """
        return False  # 子类可以覆盖