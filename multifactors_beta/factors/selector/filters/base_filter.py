"""
筛选器基类

定义筛选器的标准接口
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """
    筛选器基类
    
    定义了因子筛选的标准接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化筛选器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.filter_history = []
        
        logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def filter(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               **kwargs) -> Dict[str, pd.Series]:
        """
        筛选因子
        
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
        pass
    
    def validate_inputs(self,
                        factors: Dict[str, pd.Series],
                        evaluation_results: Optional[Dict] = None):
        """
        验证输入参数
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
        evaluation_results : Dict, optional
            评估结果
        """
        if not factors:
            raise ValueError("Factors dictionary cannot be empty")
        
        for name, factor in factors.items():
            if not isinstance(factor, pd.Series):
                raise TypeError(f"Factor {name} must be pd.Series")
            
            if not isinstance(factor.index, pd.MultiIndex):
                raise ValueError(f"Factor {name} must have MultiIndex")
    
    def get_filter_info(self) -> Dict[str, Any]:
        """
        获取筛选器信息
        
        Returns
        -------
        Dict[str, Any]
            筛选器配置和状态信息
        """
        return {
            'class': self.__class__.__name__,
            'config': self.config,
            'history_count': len(self.filter_history)
        }