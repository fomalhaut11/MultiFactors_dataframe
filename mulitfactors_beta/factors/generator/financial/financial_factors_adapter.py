#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务因子数据格式适配器

为纯财务因子计算器提供MultiIndex Series格式的适配层
自动处理DataFrame到MultiIndex Series的转换

Author: AI Assistant
Date: 2025-08-12
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict
import logging
from functools import wraps

from ...utils.multiindex_helper import (
    ensure_multiindex_format,
    multiindex_to_dataframe,
    dataframe_to_multiindex,
    validate_factor_format,
    MultiIndexHelper
)

logger = logging.getLogger(__name__)


class FinancialFactorAdapter:
    """财务因子数据格式适配器"""
    
    @staticmethod
    def adapt_input(func):
        """
        装饰器：自动将输入转换为DataFrame格式（供内部计算使用）
        
        用于包装现有的财务因子计算方法，使其能够接受MultiIndex Series输入
        """
        @wraps(func)
        def wrapper(self, data: Union[pd.Series, pd.DataFrame], **kwargs):
            # 如果输入是MultiIndex Series，转换为DataFrame
            if isinstance(data, pd.Series):
                if isinstance(data.index, pd.MultiIndex):
                    # 转换为DataFrame格式
                    data_df = multiindex_to_dataframe(data)
                    logger.debug(f"将MultiIndex Series转换为DataFrame进行计算: {func.__name__}")
                else:
                    # 单索引Series，直接使用
                    data_df = data.to_frame()
            else:
                data_df = data
            
            # 调用原始方法
            result = func(self, data_df, **kwargs)
            
            # 如果结果是Series且不是MultiIndex，转换为MultiIndex格式
            if isinstance(result, pd.Series) and not isinstance(result.index, pd.MultiIndex):
                # 假设结果的索引结构与输入相同
                if isinstance(data, pd.Series) and isinstance(data.index, pd.MultiIndex):
                    # 尝试对齐索引
                    try:
                        result = result.reindex(data.index)
                    except:
                        logger.warning(f"无法对齐结果索引: {func.__name__}")
            
            return result
        
        return wrapper
    
    @staticmethod
    def adapt_output(func):
        """
        装饰器：自动将输出转换为MultiIndex Series格式
        
        确保所有因子计算方法返回标准的MultiIndex Series
        """
        @wraps(func)
        def wrapper(self, data: Union[pd.Series, pd.DataFrame], **kwargs):
            # 调用原始方法
            result = func(self, data, **kwargs)
            
            # 确保输出是MultiIndex Series格式
            if result is not None:
                result = ensure_multiindex_format(result)
                validate_factor_format(result, raise_error=False)
            
            return result
        
        return wrapper
    
    @staticmethod
    def batch_adapt(calculator_class):
        """
        批量适配一个计算器类的所有calculate_*方法
        
        Parameters
        ----------
        calculator_class : class
            需要适配的计算器类
            
        Returns
        -------
        class
            适配后的类
        """
        # 获取所有calculate_开头的方法
        for attr_name in dir(calculator_class):
            if attr_name.startswith('calculate_'):
                attr = getattr(calculator_class, attr_name)
                if callable(attr):
                    # 应用适配器装饰器
                    adapted_method = FinancialFactorAdapter.adapt_input(
                        FinancialFactorAdapter.adapt_output(attr)
                    )
                    setattr(calculator_class, attr_name, adapted_method)
        
        return calculator_class


class PureFinancialFactorCalculatorV2:
    """
    纯财务因子计算器V2版本
    
    支持MultiIndex Series格式的输入输出
    内部调用原始的PureFinancialFactorCalculator并进行格式转换
    """
    
    def __init__(self):
        # 导入原始计算器
        from .pure_financial_factors import PureFinancialFactorCalculator
        self.calculator = PureFinancialFactorCalculator()
        
        # 动态适配所有calculate_*方法
        self._adapt_methods()
    
    def _adapt_methods(self):
        """动态适配所有计算方法"""
        for attr_name in dir(self.calculator):
            if attr_name.startswith('calculate_'):
                # 创建一个包装方法
                original_method = getattr(self.calculator, attr_name)
                adapted_method = self._create_adapted_method(original_method)
                setattr(self, attr_name, adapted_method)
    
    def _create_adapted_method(self, original_method):
        """创建适配后的方法"""
        @wraps(original_method)
        def adapted_method(data: Union[pd.Series, pd.DataFrame], **kwargs) -> pd.Series:
            # 输入格式转换
            if isinstance(data, pd.Series) and isinstance(data.index, pd.MultiIndex):
                # MultiIndex Series转DataFrame
                data_df = multiindex_to_dataframe(data)
            elif isinstance(data, pd.DataFrame):
                data_df = data
            else:
                # 尝试转换为MultiIndex格式后再转DataFrame
                data = ensure_multiindex_format(data)
                data_df = multiindex_to_dataframe(data)
            
            # 调用原始方法
            result = original_method(data_df, **kwargs)
            
            # 输出格式转换
            if isinstance(result, pd.Series):
                # 如果结果已经是MultiIndex Series，直接返回
                if isinstance(result.index, pd.MultiIndex):
                    return result
                # 否则尝试转换
                try:
                    result = ensure_multiindex_format(result)
                except:
                    logger.warning(f"无法将结果转换为MultiIndex格式: {original_method.__name__}")
            elif isinstance(result, pd.DataFrame):
                # DataFrame转MultiIndex Series
                result = dataframe_to_multiindex(result)
            
            return result
        
        return adapted_method
    
    def calculate_multiple_factors(self,
                                  factor_names: List[str],
                                  financial_data: pd.Series,
                                  **kwargs) -> Dict[str, pd.Series]:
        """
        批量计算多个因子（MultiIndex版本）
        
        Parameters
        ----------
        factor_names : List[str]
            要计算的因子名称列表
        financial_data : pd.Series
            财务数据，MultiIndex格式
        **kwargs
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            因子名称到因子值的映射，每个因子都是MultiIndex Series
        """
        results = {}
        
        for factor_name in factor_names:
            method_name = f"calculate_{factor_name}"
            if hasattr(self, method_name):
                try:
                    logger.info(f"计算纯财务因子: {factor_name}")
                    method = getattr(self, method_name)
                    result = method(financial_data, **kwargs)
                    results[factor_name] = result
                    logger.info(f"因子 {factor_name} 计算完成")
                except Exception as e:
                    logger.error(f"计算因子 {factor_name} 失败: {e}")
            else:
                logger.warning(f"未知因子: {factor_name}")
        
        return results
    
    def get_available_factors(self) -> List[str]:
        """获取所有可用因子列表"""
        return self.calculator.get_available_factors()
    
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        return self.calculator.get_factor_categories()


# ========== 便捷函数 ==========

def create_financial_calculator(version: str = 'v2'):
    """
    创建财务因子计算器
    
    Parameters
    ----------
    version : str
        版本选择
        - 'v1': 原始版本（DataFrame格式）
        - 'v2': MultiIndex Series版本（默认）
        
    Returns
    -------
    calculator
        财务因子计算器实例
    """
    if version == 'v2':
        return PureFinancialFactorCalculatorV2()
    else:
        from .pure_financial_factors import PureFinancialFactorCalculator
        return PureFinancialFactorCalculator()


# ========== 导出接口 ==========

__all__ = [
    'FinancialFactorAdapter',
    'PureFinancialFactorCalculatorV2',
    'create_financial_calculator'
]