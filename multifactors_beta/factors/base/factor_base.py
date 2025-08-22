"""
因子基类和核心功能
"""
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# 导入重构后的工具函数
from core.utils import (
    OutlierHandler,
    Normalizer,
    TechnicalIndicators,
    FactorOrthogonalizer
)
# 导入MultiIndex工具
from ..utils.multiindex_helper import (
    validate_factor_format,
    ensure_multiindex_format,
    MultiIndexHelper
)
from core.config_manager import config
from .validation import validate_inputs, ErrorHandler
from .testable_mixin import TestableMixin

logger = logging.getLogger(__name__)


class FactorBase(ABC, TestableMixin):
    """因子计算基类（集成可测试性）"""
    
    # 类级配置
    ENABLE_VALIDATION = True  # 是否启用输入验证
    ENABLE_ERROR_HANDLING = True  # 是否启用增强错误处理
    ENABLE_DATA_QUALITY_LOGGING = False  # 是否记录数据质量报告
    ENABLE_TESTABILITY = True  # 是否启用可测试性功能
    
    def __init__(self, name: str, category: str):
        """
        Parameters:
        -----------
        name : 因子名称
        category : 因子类别（fundamental, technical, risk等）
        """
        # 调用父类初始化方法（支持多重继承）
        super().__init__()
        
        self.name = name
        self.category = category
        self.description = ""
        self.parameters = {}
        
    @abstractmethod
    def calculate(self, data: pd.Series, **kwargs) -> pd.Series:
        """
        计算因子
        
        Parameters:
        -----------
        data : pd.Series
            输入数据，MultiIndex格式[TradingDates, StockCodes]
        **kwargs : 额外参数
        
        Returns:
        --------
        pd.Series : 因子值，MultiIndex格式
        """
        pass
    
    def preprocess(self, factor: pd.Series, 
                   remove_outliers: bool = True,
                   standardize: bool = True,
                   outlier_method: str = "IQR",
                   outlier_threshold: float = 3.0,
                   standardize_method: str = "zscore") -> pd.Series:
        """
        因子预处理：去极值和标准化
        """
        # 确保数据格式正确
        factor = ensure_multiindex_format(factor)
        validate_factor_format(factor)
        
        result = factor.copy()
        
        if remove_outliers:
            # 按日期分组去极值
            result = result.groupby(level='TradingDates', group_keys=False).apply(
                lambda x: pd.Series(
                    OutlierHandler.remove_outlier(
                        x.values, 
                        method=outlier_method, 
                        threshold=outlier_threshold
                    ),
                    index=x.index
                )
            )
            
        if standardize:
            # 按日期分组标准化
            result = result.groupby(level='TradingDates', group_keys=False).apply(
                lambda x: pd.Series(
                    Normalizer.normalize(
                        x.values, 
                        method=standardize_method
                    ),
                    index=x.index
                )
            )
            
        return result
    
    def save(self, factor: pd.Series, save_path: Optional[str] = None):
        """保存因子数据"""
        if save_path is None:
            save_path = config.get_path('factors') / f"{self.name}.pkl"
        
        factor.to_pickle(save_path)
        logger.info(f"Factor {self.name} saved to {save_path}")
    
    def load(self, load_path: Optional[str] = None) -> pd.Series:
        """加载因子数据"""
        if load_path is None:
            load_path = config.get_path('factors') / f"{self.name}.pkl"
        
        factor = pd.read_pickle(load_path)
        logger.info(f"Factor {self.name} loaded from {load_path}")
        return factor


class MultiFactorBase(ABC):
    """多因子组合基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.factors: Dict[str, FactorBase] = {}
        
    def add_factor(self, factor: FactorBase):
        """添加因子"""
        self.factors[factor.name] = factor
        
    def calculate_all(self, data: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """计算所有因子
        
        Parameters:
        -----------
        data : pd.Series
            输入数据，MultiIndex格式
        **kwargs : 额外参数
        
        Returns:
        --------
        Dict[str, pd.Series] : 因子名称到因子值的映射
        """
        # 确保数据格式正确
        data = ensure_multiindex_format(data)
        
        results = {}
        
        for name, factor in self.factors.items():
            try:
                logger.info(f"Calculating factor: {name}")
                results[name] = factor.calculate(data, **kwargs)
            except Exception as e:
                logger.error(f"Error calculating factor {name}: {e}")
                
        return results
    
    def orthogonalize(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """因子正交化处理
        
        Parameters:
        -----------
        factors : Dict[str, pd.Series]
            因子字典，每个因子都是MultiIndex Series
            
        Returns:
        --------
        Dict[str, pd.Series] : 正交化后的因子
        """
        # 将所有因子对齐并转换为DataFrame进行正交化
        aligned_factors = MultiIndexHelper.align_data(*factors.values())
        factor_names = list(factors.keys())
        
        # 转换为DataFrame格式（临时）
        df_dict = {}
        for name, series in zip(factor_names, aligned_factors):
            df_dict[name] = series
        
        factors_df = pd.DataFrame(df_dict)
        
        # 正交化处理
        orthogonalized_df = FactorOrthogonalizer.sequential_orthogonalize(
            factors_df,
            normalize=True,
            remove_outliers=True
        )
        
        # 转换回字典格式
        result = {}
        for col in orthogonalized_df.columns:
            result[col] = orthogonalized_df[col]
            
        return result