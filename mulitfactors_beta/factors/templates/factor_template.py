#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[因子名称] - [简要说明]

[详细描述]
- 计算逻辑
- 使用场景
- 参考文献（如果有）

Author: [Your Name]
Date: [Creation Date]
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

from ...base import FactorBase, DataProcessingMixin
from ...utils.multiindex_helper import (
    validate_factor_format,
    ensure_multiindex_format,
    MultiIndexHelper
)

logger = logging.getLogger(__name__)


class TemplateFactor(FactorBase, DataProcessingMixin):
    """
    [因子中文名] - [因子英文名]
    
    计算公式：
    [写出具体的计算公式]
    
    因子含义：
    [解释因子的经济含义和投资逻辑]
    
    Attributes
    ----------
    window : int
        计算窗口期，默认20天
    min_periods : int
        最小有效数据点数，默认为窗口期的一半
    
    Examples
    --------
    >>> from factors.generator.[category] import TemplateFactor
    >>> factor = TemplateFactor(window=30)
    >>> result = factor.calculate(price_data)
    """
    
    def __init__(self,
                 window: int = 20,
                 min_periods: Optional[int] = None,
                 **kwargs):
        """
        初始化因子
        
        Parameters
        ----------
        window : int, optional
            计算窗口期，默认20
        min_periods : int, optional
            最小有效数据点数，如果不指定则为窗口期的一半
        **kwargs
            其他参数传递给父类
        """
        # 设置默认的name和category
        kwargs.setdefault('name', 'TemplateFactor')
        kwargs.setdefault('category', 'technical')  # 修改为实际类别
        super().__init__(**kwargs)
        
        # 因子特定参数
        self.window = window
        self.min_periods = min_periods or (window // 2)
        
        # 参数验证
        self._validate_parameters()
        
    def _validate_parameters(self):
        """验证参数合法性"""
        if self.window <= 0:
            raise ValueError(f"窗口期必须为正数，当前值: {self.window}")
            
        if self.min_periods > self.window:
            raise ValueError(
                f"最小数据点数({self.min_periods})不能大于窗口期({self.window})"
            )
            
    def calculate(self, 
                 data: pd.Series,
                 **kwargs) -> pd.Series:
        """
        计算因子值
        
        Parameters
        ----------
        data : pd.Series
            输入数据，MultiIndex格式
            - 第一级索引: TradingDates（交易日期）
            - 第二级索引: StockCodes（股票代码）
        **kwargs
            额外参数
            
        Returns
        -------
        pd.Series
            因子值，MultiIndex格式
            - 第一级索引: TradingDates（与输入数据对齐）
            - 第二级索引: StockCodes（与输入数据对齐）
            
        Raises
        ------
        ValueError
            当输入数据格式不正确时
        """
        # 1. 输入验证和格式确保
        data = ensure_multiindex_format(data)
        validate_factor_format(data)
        
        if data.empty:
            raise ValueError("输入数据为空")
            
        logger.info(f"开始计算{self.name}因子，数据长度: {len(data)}")
        
        # 2. 数据预处理（可选）
        processed_data = self._preprocess_data(data)
        
        # 3. 核心计算逻辑
        try:
            # ========== 在这里实现具体的因子计算逻辑 ==========
            # 示例：按日期分组计算
            def calculate_daily(group):
                # 这里实现每日的因子计算
                return group  # 替换为实际计算
            
            factor_values = MultiIndexHelper.groupby_date(processed_data).apply(
                calculate_daily
            )
            # ===================================================
            
        except Exception as e:
            logger.error(f"计算{self.name}因子时出错: {e}")
            raise
            
        # 4. 后处理（可选）
        factor_values = self._postprocess_factor(factor_values)
        
        # 5. 数据质量检查
        self._check_output_quality(factor_values)
        
        logger.info(f"{self.name}因子计算完成，有效值比例: {factor_values.notna().mean().mean():.2%}")
        
        return factor_values
        
    def _preprocess_data(self, data: pd.Series) -> pd.Series:
        """
        数据预处理
        
        可以在这里添加：
        - 缺失值处理
        - 异常值处理
        - 数据转换
        """
        # 示例：前向填充缺失值
        processed = data.fillna(method='ffill', limit=5)
        
        return processed
        
    def _postprocess_factor(self, factor: pd.Series) -> pd.Series:
        """
        因子后处理
        
        可以在这里添加：
        - 去极值
        - 标准化
        - 中性化
        """
        # 示例：去极值和标准化（如果启用了DataProcessingMixin）
        if hasattr(self, 'winsorize'):
            factor = self.winsorize(factor, limits=(0.01, 0.99))
            
        if hasattr(self, 'standardize'):
            factor = self.standardize(factor)
            
        return factor
        
    def _check_output_quality(self, factor: pd.Series):
        """
        检查输出质量
        
        Parameters
        ----------
        factor : pd.Series
            计算得到的因子值
        """
        # 验证输出格式
        validate_factor_format(factor, raise_error=False)
        
        # 检查是否全为NaN
        if factor.isna().all():
            logger.warning(f"{self.name}因子计算结果全为NaN")
            
        # 检查缺失值比例
        missing_ratio = factor.isna().sum() / len(factor)
        if missing_ratio > 0.5:
            logger.warning(f"{self.name}因子缺失值比例过高: {missing_ratio:.2%}")
            
        # 检查是否有无穷值
        if np.isinf(factor.values).any():
            logger.warning(f"{self.name}因子包含无穷值")
            
    def get_info(self) -> Dict[str, Any]:
        """
        获取因子信息
        
        Returns
        -------
        dict
            因子元信息
        """
        return {
            'name': self.name,
            'category': self.category,
            'window': self.window,
            'min_periods': self.min_periods,
            'description': self.__doc__,
            'version': '1.0.0'
        }


# ========== 便捷函数 ==========

def calculate_template_factor(data: Union[pd.Series, pd.DataFrame],
                             window: int = 20,
                             **kwargs) -> pd.Series:
    """
    计算模板因子的便捷函数
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        输入数据（自动转换为MultiIndex Series）
    window : int
        窗口期
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series
        因子值（MultiIndex格式）
        
    Examples
    --------
    >>> factor = calculate_template_factor(price_data, window=30)
    """
    factor = TemplateFactor(window=window, **kwargs)
    return factor.calculate(data)


# ========== 导出接口 ==========

__all__ = [
    'TemplateFactor',
    'calculate_template_factor'
]