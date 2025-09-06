#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EP_ttm 因子 (Earnings to Price, TTM)
使用现有工具的简洁实现

计算公式：EP_ttm = TTM净利润 / 总市值
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from ...base.factor_base import FactorBase
from .financial_report_processor import FinancialReportProcessor
from ...base.data_processing_mixin import DataProcessingMixin
from ...utils.data_loader import FactorDataLoader
from ...utils.multiindex_helper import MultiIndexHelper

logger = logging.getLogger(__name__)


class EP_ttm_Factor(FactorBase, DataProcessingMixin):
    """EP_ttm因子：使用现有工具的简洁实现"""
    
    def __init__(self):
        super().__init__(name='EP_ttm', category='valuation')
        self.description = "Earnings-to-Price ratio (TTM), inverse of P/E ratio"
    
    def calculate(self, data: pd.Series = None, **kwargs) -> pd.Series:
        """
        计算EP_ttm因子
        
        Parameters:
        -----------
        data : pd.Series, optional
            输入数据（这里不使用，自动加载）
        **kwargs : 其他参数
            
        Returns:
        --------
        pd.Series
            EP_ttm因子值
        """
        logger.info("开始计算 EP_ttm 因子")
        
        # 1. 使用现有接口加载数据
        financial_data = self._load_financial_data()
        market_cap_data = self._load_market_cap_data()
        
        # 2. 使用现有工具计算TTM净利润
        ttm_data = FinancialReportProcessor.calculate_ttm(financial_data)
        ttm_earnings = ttm_data['net_income']
        
        # 3. 使用现有数据处理混入进行数据扩展和对齐
        # 将TTM净利润转为DataFrame格式（_expand_and_align_data需要）
        ttm_earnings_df = ttm_earnings.to_frame('net_income_ttm')
        
        aligned_earnings, aligned_market_cap = self._expand_and_align_data(
            factor_data=ttm_earnings_df,
            market_cap=market_cap_data
        )
        
        # 4. 计算EP比率
        ep_ratio = self._calculate_ep_ratio(aligned_earnings, aligned_market_cap)
        
        # 5. 使用基类的预处理功能
        ep_ratio = self.preprocess(ep_ratio)
        
        logger.info(f"EP_ttm 因子计算完成，数据点数：{len(ep_ratio)}")
        return ep_ratio
    
    def _load_financial_data(self) -> pd.DataFrame:
        """使用现有接口加载财务数据"""
        try:
            data_path = FactorDataLoader._get_data_path('FinancialData_unified.pkl')
            financial_data = pd.read_pickle(data_path)
            
            # 确保有net_income列和d_quarter列（TTM计算需要）
            if 'net_income' not in financial_data.columns:
                raise ValueError("财务数据缺少 net_income 列")
            if 'd_quarter' not in financial_data.columns:
                raise ValueError("财务数据缺少 d_quarter 列")
                
            return financial_data
            
        except Exception as e:
            logger.error(f"加载财务数据失败: {e}")
            raise RuntimeError(f"无法加载财务数据: {e}")
    
    def _load_market_cap_data(self) -> pd.Series:
        """使用现有接口加载市值数据"""
        try:
            return FactorDataLoader.load_market_cap()
        except Exception as e:
            logger.error(f"加载市值数据失败: {e}")
            raise RuntimeError(f"无法加载市值数据: {e}")
    
    def _calculate_ep_ratio(self, ttm_earnings: pd.Series, market_cap: pd.Series) -> pd.Series:
        """计算EP比率"""
        logger.debug("计算EP比率")
        
        # 使用现有工具对齐数据
        aligned_earnings, aligned_market_cap = MultiIndexHelper.align_data(ttm_earnings, market_cap)
        
        # 计算EP比率，处理除零情况
        with np.errstate(divide='ignore', invalid='ignore'):
            ep_ratio = aligned_earnings / aligned_market_cap
        
        # 处理异常值
        ep_ratio = ep_ratio.replace([np.inf, -np.inf], np.nan)
        
        return ep_ratio.dropna()


def calculate_ep_ttm(**kwargs) -> pd.Series:
    """
    计算EP_ttm因子的便捷函数
    
    Returns:
    --------
    pd.Series
        EP_ttm因子
        
    Examples:
    --------
    >>> from factors.generator.financial.ep_ttm_factor import calculate_ep_ttm
    >>> ep_factor = calculate_ep_ttm()
    >>> print(f"EP_ttm因子: {ep_factor.shape}")
    """
    factor = EP_ttm_Factor()
    return factor.calculate(**kwargs)


if __name__ == "__main__":
    # 测试代码
    print("EP_ttm因子计算模块测试")
    
    try:
        ep_factor = calculate_ep_ttm()
        
        print(f"EP_ttm因子计算成功:")
        print(f"  数据形状: {ep_factor.shape}")
        print(f"  数据范围: {ep_factor.min():.6f} ~ {ep_factor.max():.6f}")
        print(f"  均值: {ep_factor.mean():.6f}")
        print(f"  标准差: {ep_factor.std():.6f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()