#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiIndex Series数据格式辅助工具

提供MultiIndex Series格式的验证、转换和处理功能
确保整个factors模块数据格式的一致性

标准格式：
- 第一级索引：TradingDates（交易日期）
- 第二级索引：StockCodes（股票代码）
- 数据类型：pd.Series

Author: AI Assistant
Date: 2025-08-12
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiIndexHelper:
    """MultiIndex Series格式辅助工具类"""
    
    # 标准索引名称
    DATE_INDEX = 'TradingDates'
    STOCK_INDEX = 'StockCodes'
    STANDARD_INDEX_NAMES = [DATE_INDEX, STOCK_INDEX]
    
    @classmethod
    def validate_format(cls, data: pd.Series, 
                       raise_error: bool = True) -> bool:
        """
        验证数据是否符合标准MultiIndex Series格式
        
        Parameters
        ----------
        data : pd.Series
            待验证的数据
        raise_error : bool
            是否在验证失败时抛出异常
            
        Returns
        -------
        bool
            验证是否通过
            
        Raises
        ------
        TypeError
            当数据不是Series类型时
        ValueError
            当数据格式不符合规范时
        """
        # 检查是否为Series
        if not isinstance(data, pd.Series):
            msg = f"数据必须是pd.Series类型，当前类型: {type(data)}"
            if raise_error:
                raise TypeError(msg)
            logger.error(msg)
            return False
        
        # 检查是否为MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            msg = "数据必须使用MultiIndex格式"
            if raise_error:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        # 检查索引级别数
        if data.index.nlevels != 2:
            msg = f"MultiIndex必须有2个级别，当前级别数: {data.index.nlevels}"
            if raise_error:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        # 检查索引名称
        if list(data.index.names) != cls.STANDARD_INDEX_NAMES:
            msg = f"索引名称必须为{cls.STANDARD_INDEX_NAMES}，当前: {data.index.names}"
            if raise_error:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        # 检查数据类型
        if not np.issubdtype(data.dtype, np.number):
            logger.warning(f"数据包含非数值类型: {data.dtype}")
        
        return True
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                      value_name: Optional[str] = None) -> pd.Series:
        """
        将DataFrame转换为标准MultiIndex Series格式
        
        Parameters
        ----------
        df : pd.DataFrame
            输入DataFrame
            - 如果index为日期，columns为股票：直接stack
            - 如果已经是MultiIndex：转换为Series
            - 如果是单列DataFrame：提取该列
        value_name : str, optional
            值的名称
            
        Returns
        -------
        pd.Series
            标准格式的MultiIndex Series
        """
        if df.empty:
            # 创建空的MultiIndex Series
            index = pd.MultiIndex.from_tuples([], names=cls.STANDARD_INDEX_NAMES)
            return pd.Series([], index=index, name=value_name)
        
        # 情况1：已经是MultiIndex的DataFrame
        if isinstance(df.index, pd.MultiIndex):
            if df.shape[1] == 1:
                # 单列DataFrame，直接转换
                series = df.iloc[:, 0]
            else:
                # 多列DataFrame，需要指定列或报错
                raise ValueError(f"MultiIndex DataFrame有{df.shape[1]}列，请指定要转换的列")
            
            # 标准化索引名称
            if series.index.nlevels == 2:
                series.index.names = cls.STANDARD_INDEX_NAMES
            
        # 情况2：普通DataFrame（日期为index，股票为columns）
        else:
            # Stack转换
            series = df.stack()
            series.index.names = cls.STANDARD_INDEX_NAMES
        
        if value_name:
            series.name = value_name
            
        return series
    
    @classmethod
    def to_dataframe(cls, series: pd.Series) -> pd.DataFrame:
        """
        将MultiIndex Series转换为DataFrame格式
        
        Parameters
        ----------
        series : pd.Series
            标准格式的MultiIndex Series
            
        Returns
        -------
        pd.DataFrame
            日期为index，股票为columns的DataFrame
        """
        # 验证格式
        cls.validate_format(series, raise_error=True)
        
        # Unstack转换
        df = series.unstack(level=cls.STOCK_INDEX)
        
        # 确保索引为日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("无法将索引转换为日期类型")
        
        return df
    
    @classmethod
    def align_data(cls, *data_series: pd.Series) -> Tuple[pd.Series, ...]:
        """
        对齐多个MultiIndex Series的索引
        
        Parameters
        ----------
        *data_series : pd.Series
            多个MultiIndex Series
            
        Returns
        -------
        tuple of pd.Series
            对齐后的Series
        """
        if not data_series:
            return tuple()
        
        # 验证所有数据格式
        for i, series in enumerate(data_series):
            cls.validate_format(series, raise_error=True)
        
        # 找到公共索引
        common_index = data_series[0].index
        for series in data_series[1:]:
            common_index = common_index.intersection(series.index)
        
        # 对齐数据
        aligned = tuple(series.reindex(common_index) for series in data_series)
        
        logger.info(f"数据对齐完成，公共索引长度: {len(common_index)}")
        
        return aligned
    
    @classmethod
    def create_empty(cls, dates: Optional[pd.DatetimeIndex] = None,
                    stocks: Optional[List[str]] = None,
                    name: Optional[str] = None) -> pd.Series:
        """
        创建空的标准格式MultiIndex Series
        
        Parameters
        ----------
        dates : pd.DatetimeIndex, optional
            日期索引
        stocks : list of str, optional
            股票代码列表
        name : str, optional
            Series名称
            
        Returns
        -------
        pd.Series
            空的MultiIndex Series（填充NaN）
        """
        if dates is None:
            dates = pd.DatetimeIndex([])
        if stocks is None:
            stocks = []
        
        # 创建MultiIndex
        index = pd.MultiIndex.from_product(
            [dates, stocks],
            names=cls.STANDARD_INDEX_NAMES
        )
        
        # 创建Series
        series = pd.Series(np.nan, index=index, name=name)
        
        return series
    
    @classmethod
    def ensure_format(cls, data: Union[pd.Series, pd.DataFrame],
                     value_name: Optional[str] = None) -> pd.Series:
        """
        确保数据为标准MultiIndex Series格式
        
        自动检测输入格式并进行必要的转换
        
        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            输入数据
        value_name : str, optional
            值的名称
            
        Returns
        -------
        pd.Series
            标准格式的MultiIndex Series
        """
        # 如果已经是正确格式，直接返回
        if isinstance(data, pd.Series):
            try:
                if cls.validate_format(data, raise_error=False):
                    return data
            except:
                pass
        
        # 如果是DataFrame，转换
        if isinstance(data, pd.DataFrame):
            return cls.from_dataframe(data, value_name)
        
        # 如果是Series但格式不对，尝试修复
        if isinstance(data, pd.Series):
            # 检查是否只是索引名称不对
            if isinstance(data.index, pd.MultiIndex) and data.index.nlevels == 2:
                data = data.copy()
                data.index.names = cls.STANDARD_INDEX_NAMES
                if value_name:
                    data.name = value_name
                return data
        
        raise ValueError(f"无法将{type(data)}转换为标准MultiIndex Series格式")
    
    @classmethod
    def groupby_date(cls, series: pd.Series):
        """
        按日期分组
        
        Parameters
        ----------
        series : pd.Series
            MultiIndex Series
            
        Returns
        -------
        SeriesGroupBy
            按日期分组的对象
        """
        cls.validate_format(series)
        return series.groupby(level=cls.DATE_INDEX)
    
    @classmethod
    def groupby_stock(cls, series: pd.Series):
        """
        按股票分组
        
        Parameters
        ----------
        series : pd.Series
            MultiIndex Series
            
        Returns
        -------
        SeriesGroupBy
            按股票分组的对象
        """
        cls.validate_format(series)
        return series.groupby(level=cls.STOCK_INDEX)
    
    @classmethod
    def get_info(cls, series: pd.Series) -> dict:
        """
        获取MultiIndex Series的信息
        
        Parameters
        ----------
        series : pd.Series
            MultiIndex Series
            
        Returns
        -------
        dict
            包含数据信息的字典
        """
        cls.validate_format(series)
        
        dates = series.index.get_level_values(cls.DATE_INDEX).unique()
        stocks = series.index.get_level_values(cls.STOCK_INDEX).unique()
        
        info = {
            'shape': len(series),
            'n_dates': len(dates),
            'n_stocks': len(stocks),
            'date_range': f"{dates.min()} to {dates.max()}" if len(dates) > 0 else "N/A",
            'missing_ratio': f"{series.isna().sum() / len(series):.2%}" if len(series) > 0 else "N/A",
            'dtype': str(series.dtype),
            'memory_usage': f"{series.memory_usage(deep=True) / 1024 / 1024:.2f} MB"
        }
        
        return info


# ========== 便捷函数 ==========

def validate_factor_format(data: pd.Series, raise_error: bool = True) -> bool:
    """验证因子数据格式"""
    return MultiIndexHelper.validate_format(data, raise_error)


def dataframe_to_multiindex(df: pd.DataFrame, 
                           value_name: Optional[str] = None) -> pd.Series:
    """DataFrame转MultiIndex Series"""
    return MultiIndexHelper.from_dataframe(df, value_name)


def multiindex_to_dataframe(series: pd.Series) -> pd.DataFrame:
    """MultiIndex Series转DataFrame"""
    return MultiIndexHelper.to_dataframe(series)


def ensure_multiindex_format(data: Union[pd.Series, pd.DataFrame],
                            value_name: Optional[str] = None) -> pd.Series:
    """确保数据为MultiIndex格式"""
    return MultiIndexHelper.ensure_format(data, value_name)


def align_factor_data(*factors: pd.Series) -> Tuple[pd.Series, ...]:
    """对齐多个因子数据"""
    return MultiIndexHelper.align_data(*factors)


# ========== 导出接口 ==========

__all__ = [
    'MultiIndexHelper',
    'validate_factor_format',
    'dataframe_to_multiindex',
    'multiindex_to_dataframe',
    'ensure_multiindex_format',
    'align_factor_data'
]