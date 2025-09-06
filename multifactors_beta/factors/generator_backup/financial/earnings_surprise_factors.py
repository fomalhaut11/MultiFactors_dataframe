#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盈余惊喜因子计算模块

包含SUE（标准化未预期盈余）等盈余惊喜相关因子
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
import warnings
import logging

# 配置logger
logger = logging.getLogger(__name__)

from ...base import FactorBase, DataProcessingMixin
from ...utils.multiindex_helper import (
    ensure_multiindex_format,
    multiindex_to_dataframe,
    dataframe_to_multiindex,
    validate_factor_format
)


class SUE(FactorBase, DataProcessingMixin):
    """
    SUE (Standardized Unexpected Earnings) - 标准化未预期盈余因子
    
    计算逻辑：
    1. 基础SUE = (当期EPS - 预期EPS) / 标准差
    2. 预期EPS可选：
       - 分析师一致预期
       - 历史平均（过去4个季度）
       - 时间序列预测
    3. 标准差：过去8个季度EPS的标准差
    
    因子含义：
    - 正值表示盈余超预期
    - 负值表示盈余低于预期
    - 绝对值越大，惊喜程度越高
    """
    
    # 定义SUE因子需要的字段映射
    REQUIRED_FIELDS = {
        'eps': 'NET_PROFIT_EXCL_MIN_INT_INC',  # 扣非净利润作为EPS代理
        'eps_basic': 'EPS_BASIC_IS',           # 基本每股收益
        'net_profit': 'NET_PROFIT_IS'          # 净利润
    }
    
    @classmethod
    def get_required_fields(cls) -> dict:
        """返回SUE因子需要的字段映射"""
        return cls.REQUIRED_FIELDS
    
    @classmethod
    def validate_data_format(cls, data) -> str:
        """
        验证并返回数据应该使用的字段
        
        Returns
        -------
        str
            推荐使用的字段名
        """
        if isinstance(data, pd.Series):
            return data.name if data.name else 'unknown_field'
        elif isinstance(data, pd.DataFrame):
            # 如果是DataFrame，尝试自动选择合适的字段
            available_fields = data.columns.tolist()
            
            # 优先级：扣非净利润 > 基本EPS > 净利润
            for field_key, field_name in cls.REQUIRED_FIELDS.items():
                if field_name in available_fields:
                    logger.info(f"SUE因子自动选择字段: {field_name}")
                    return field_name
            
            # 如果没找到推荐字段，抛出明确错误
            eps_like_fields = [col for col in available_fields if 
                             any(keyword in col.upper() for keyword in ['EPS', 'PROFIT', 'EARN'])]
            
            raise ValueError(
                f"SUE因子需要单列EPS数据，但收到了{len(available_fields)}列的DataFrame。\n"
                f"建议使用的字段: {list(cls.REQUIRED_FIELDS.values())}\n"
                f"可用的EPS相关字段: {eps_like_fields[:5]}...\n"
                f"使用方式: sue.calculate(financial_data['{cls.REQUIRED_FIELDS['eps']}'])"
            )
        else:
            raise TypeError(f"SUE因子需要Series或DataFrame，但收到了{type(data)}")
    
    def __init__(self, 
                 method: str = 'historical',
                 lookback_quarters: int = 4,
                 std_quarters: int = 8,
                 min_quarters: int = 4,
                 **kwargs):
        """
        初始化SUE因子
        
        Parameters
        ----------
        method : str
            预期计算方法：
            - 'historical': 使用历史平均
            - 'analyst': 使用分析师预期（需要额外数据）
            - 'timeseries': 时间序列预测
        lookback_quarters : int
            历史平均的回看季度数
        std_quarters : int
            计算标准差的季度数
        min_quarters : int
            最少需要的季度数
        """
        # 设置默认的name和category
        kwargs.setdefault('name', 'SUE')
        kwargs.setdefault('category', 'earnings_surprise')
        super().__init__(**kwargs)
        self.method = method
        self.lookback_quarters = lookback_quarters
        self.std_quarters = std_quarters
        self.min_quarters = min_quarters
        
    def calculate(self, 
                  eps_data: Union[pd.Series, pd.DataFrame],
                  analyst_data: Optional[Union[pd.Series, pd.DataFrame]] = None,
                  **kwargs) -> pd.Series:
        """
        计算SUE因子
        
        Parameters
        ----------
        eps_data : pd.Series or pd.DataFrame
            EPS数据，优先接受MultiIndex Series格式
        analyst_data : pd.Series or pd.DataFrame, optional
            分析师预期数据（如果method='analyst'）
            
        Returns
        -------
        pd.Series
            SUE因子值，MultiIndex格式
        """
        # 首先验证和处理输入数据格式
        if isinstance(eps_data, pd.DataFrame):
            # 如果输入是DataFrame，自动选择合适的字段
            field_name = self.validate_data_format(eps_data)
            eps_data = eps_data[field_name]
            logger.info(f"SUE因子自动提取字段: {field_name}")
        
        # 确保输入格式
        eps_data = ensure_multiindex_format(eps_data)
        if analyst_data is not None:
            analyst_data = ensure_multiindex_format(analyst_data)
        
        # 转换为DataFrame进行内部计算
        eps_df = multiindex_to_dataframe(eps_data)
        analyst_df = multiindex_to_dataframe(analyst_data) if analyst_data is not None else None
        
        if self.method == 'analyst' and analyst_df is None:
            warnings.warn("分析师预期数据缺失，使用历史平均方法")
            self.method = 'historical'
            
        if self.method == 'historical':
            sue_df = self._calculate_historical_sue(eps_df)
        elif self.method == 'analyst':
            sue_df = self._calculate_analyst_sue(eps_df, analyst_df)
        elif self.method == 'timeseries':
            sue_df = self._calculate_timeseries_sue(eps_df)
        else:
            raise ValueError(f"不支持的方法: {self.method}")
        
        # 转换回MultiIndex Series格式
        sue = dataframe_to_multiindex(sue_df, value_name='SUE')
        validate_factor_format(sue, raise_error=False)
        
        return sue
            
    def _calculate_historical_sue(self, eps_data: pd.DataFrame) -> pd.DataFrame:
        """
        使用历史平均计算SUE
        """
        # 计算历史平均EPS作为预期
        expected_eps = eps_data.rolling(
            window=self.lookback_quarters,
            min_periods=self.min_quarters
        ).mean().shift(1)
        
        # 计算EPS标准差
        eps_std = eps_data.rolling(
            window=self.std_quarters,
            min_periods=self.min_quarters
        ).std().shift(1)
        
        # 计算未预期盈余
        unexpected_earnings = eps_data - expected_eps
        
        # 标准化（避免除零）
        sue = unexpected_earnings / eps_std.replace(0, np.nan)
        
        # 处理异常值（限制在[-5, 5]范围内）
        sue = sue.clip(lower=-5, upper=5)
        
        return sue
        
    def _calculate_analyst_sue(self, 
                               eps_data: pd.DataFrame,
                               analyst_data: pd.DataFrame) -> pd.DataFrame:
        """
        使用分析师预期计算SUE
        """
        # 确保数据对齐
        common_index = eps_data.index.intersection(analyst_data.index)
        common_columns = eps_data.columns.intersection(analyst_data.columns)
        
        eps_aligned = eps_data.loc[common_index, common_columns]
        analyst_aligned = analyst_data.loc[common_index, common_columns]
        
        # 计算EPS标准差
        eps_std = eps_aligned.rolling(
            window=self.std_quarters,
            min_periods=self.min_quarters
        ).std()
        
        # 计算未预期盈余
        unexpected_earnings = eps_aligned - analyst_aligned
        
        # 标准化
        sue = unexpected_earnings / eps_std.replace(0, np.nan)
        
        # 处理异常值
        sue = sue.clip(lower=-5, upper=5)
        
        # 扩展到原始数据的大小
        result = pd.DataFrame(np.nan, index=eps_data.index, columns=eps_data.columns)
        result.loc[common_index, common_columns] = sue
        
        return result
        
    def _calculate_timeseries_sue(self, eps_data: pd.DataFrame) -> pd.DataFrame:
        """
        使用时间序列模型计算SUE
        简单实现：使用线性趋势预测
        """
        sue_values = pd.DataFrame(index=eps_data.index, columns=eps_data.columns)
        
        for col in eps_data.columns:
            series = eps_data[col].dropna()
            
            if len(series) < self.min_quarters:
                continue
                
            # 使用滚动窗口进行线性回归预测
            for i in range(self.lookback_quarters, len(series)):
                window_data = series.iloc[i-self.lookback_quarters:i]
                
                # 简单线性趋势
                x = np.arange(len(window_data))
                y = window_data.values
                
                # 计算趋势
                coef = np.polyfit(x, y, 1)
                expected = np.polyval(coef, len(window_data))
                
                # 实际值
                actual = series.iloc[i]
                
                # 计算历史标准差
                if i >= self.std_quarters:
                    std = series.iloc[i-self.std_quarters:i].std()
                    if std > 0:
                        sue_values.loc[series.index[i], col] = (actual - expected) / std
                        
        # 处理异常值
        sue_values = sue_values.clip(lower=-5, upper=5)
        
        return sue_values


class EarningsRevision(FactorBase, DataProcessingMixin):
    """
    盈余修正因子
    
    衡量分析师对公司盈余预期的调整幅度
    """
    
    def __init__(self, 
                 revision_period: int = 30,
                 **kwargs):
        """
        初始化盈余修正因子
        
        Parameters
        ----------
        revision_period : int
            修正期间（天数）
        """
        kwargs.setdefault('name', 'EarningsRevision')
        kwargs.setdefault('category', 'earnings_surprise')
        super().__init__(**kwargs)
        self.revision_period = revision_period
        
    def calculate(self,
                  analyst_current: pd.DataFrame,
                  analyst_previous: pd.DataFrame,
                  **kwargs) -> pd.DataFrame:
        """
        计算盈余修正因子
        
        Parameters
        ----------
        analyst_current : pd.DataFrame
            当前分析师预期
        analyst_previous : pd.DataFrame
            之前的分析师预期
            
        Returns
        -------
        pd.DataFrame
            盈余修正因子值
        """
        # 计算预期变化率
        revision = (analyst_current - analyst_previous) / analyst_previous.abs()
        
        # 处理异常值
        revision = revision.clip(lower=-1, upper=1)
        
        return revision


class EarningsMomentum(FactorBase, DataProcessingMixin):
    """
    盈余动量因子
    
    基于连续几个季度的盈余惊喜计算动量
    """
    
    def __init__(self,
                 momentum_quarters: int = 4,
                 **kwargs):
        """
        初始化盈余动量因子
        
        Parameters
        ----------
        momentum_quarters : int
            计算动量的季度数
        """
        kwargs.setdefault('name', 'EarningsMomentum')
        kwargs.setdefault('category', 'earnings_surprise')
        super().__init__(**kwargs)
        self.momentum_quarters = momentum_quarters
        
    def calculate(self,
                  sue_data: pd.DataFrame,
                  **kwargs) -> pd.DataFrame:
        """
        计算盈余动量因子
        
        Parameters
        ----------
        sue_data : pd.DataFrame
            SUE数据
            
        Returns
        -------
        pd.DataFrame
            盈余动量因子值
        """
        # 计算SUE的移动平均作为动量
        momentum = sue_data.rolling(
            window=self.momentum_quarters,
            min_periods=2
        ).mean()
        
        # 考虑方向性：连续正/负惊喜的加权
        direction_weight = (sue_data > 0).rolling(
            window=self.momentum_quarters,
            min_periods=2
        ).mean()
        
        # 方向调整后的动量
        adjusted_momentum = momentum * (2 * direction_weight - 1)
        
        return adjusted_momentum


# 导出因子类
__all__ = [
    'SUE',
    'EarningsRevision',
    'EarningsMomentum'
]