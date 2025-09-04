#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价值因子实现
包含EP、BP、SP等经典价值因子（需要结合市值数据）
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

from ...base.factor_base import FactorBase
from ...base.data_processing_mixin import DataProcessingMixin
from ...base.flexible_data_adapter import ColumnMapperMixin
from .financial_report_processor import FinancialReportProcessor

logger = logging.getLogger(__name__)


class EPRatioFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    EP比率因子（Earnings to Price）
    计算公式：EP = 净利润TTM / 市值
    这是PE倒数，数值越高表示估值越低（价值股特征）
    """
    
    def __init__(self):
        super().__init__(name='EP_Ratio', category='value')
        self.description = "盈利收益率 - 净利润TTM/市值"
        
    def calculate(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算EP比率
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含净利润和季度信息
        market_cap : pd.Series
            市值数据，MultiIndex[TradingDates, StockCodes]
        
        Returns:
        --------
        pd.Series : EP因子值
        """
        try:
            # 使用实际的字段名
            earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
            
            # 检查必要字段是否存在
            required_cols = [earnings_col, 'd_year', 'd_quarter']
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 使用新的FinancialReportProcessor接口计算TTM
            ttm_results = FinancialReportProcessor.calculate_ttm(financial_data)
            earnings_series = ttm_results[f'{earnings_col}_ttm']
            
            # 扩展财务数据到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                earnings_df = earnings_series.to_frame(name='EarningsTTM')
                earnings_expanded = self._expand_and_align_data(
                    earnings_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                earnings_series = earnings_expanded['EarningsTTM']
            
            # 对齐数据
            earnings_aligned, market_cap_aligned = earnings_series.align(market_cap, join='inner')
            
            # 计算EP比率
            ep_ratio = self._safe_division(earnings_aligned, market_cap_aligned)
            
            # 预处理
            ep_ratio = self.preprocess(ep_ratio)
            
            return ep_ratio
            
        except Exception as e:
            logger.error(f"计算EP比率失败: {e}")
            return pd.Series(dtype=float)


class BPRatioFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    BP比率因子（Book to Price）
    计算公式：BP = 净资产 / 市值
    这是PB倒数，数值越高表示估值越低（价值股特征）
    """
    
    def __init__(self):
        super().__init__(name='BP_Ratio', category='value')
        self.description = "净资产市值比 - 净资产/市值"
        
    def calculate(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算BP比率
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含净资产信息
        market_cap : pd.Series
            市值数据，MultiIndex[TradingDates, StockCodes]
        
        Returns:
        --------
        pd.Series : BP因子值
        """
        try:
            # 使用实际的字段名
            equity_col = 'EQY_BELONGTO_PARCOMSH'  # 归属母公司股东权益
            
            # 检查必要字段是否存在
            if equity_col not in financial_data.columns:
                raise ValueError(f"Missing required column: {equity_col}")
            
            book_value = financial_data[equity_col]
            
            # 扩展财务数据到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                book_value_df = book_value.to_frame(name='BookValue')
                book_value_expanded = self._expand_and_align_data(
                    book_value_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                book_value = book_value_expanded['BookValue']
            
            # 对齐数据
            book_value_aligned, market_cap_aligned = book_value.align(market_cap, join='inner')
            
            # 计算BP比率
            bp_ratio = self._safe_division(book_value_aligned, market_cap_aligned)
            
            # 预处理
            bp_ratio = self.preprocess(bp_ratio)
            
            return bp_ratio
            
        except Exception as e:
            logger.error(f"计算BP比率失败: {e}")
            return pd.Series(dtype=float)


class SPRatioFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    SP比率因子（Sales to Price）
    计算公式：SP = 营业收入TTM / 市值
    这是PS倒数，数值越高表示估值越低（价值股特征）
    """
    
    def __init__(self):
        super().__init__(name='SP_Ratio', category='value')
        self.description = "销售收益率 - 营业收入TTM/市值"
        
    def calculate(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算SP比率
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含营业收入和季度信息
        market_cap : pd.Series
            市值数据，MultiIndex[TradingDates, StockCodes]
        
        Returns:
        --------
        pd.Series : SP因子值
        """
        try:
            # 使用实际的字段名
            revenue_col = 'TOT_OPER_REV'  # 营业收入
            
            # 检查必要字段是否存在
            required_cols = [revenue_col, 'd_year', 'd_quarter']
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 使用新的FinancialReportProcessor接口计算TTM
            ttm_results = FinancialReportProcessor.calculate_ttm(financial_data)
            revenue_series = ttm_results[f'{revenue_col}_ttm']
            
            # 扩展财务数据到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                revenue_df = revenue_series.to_frame(name='RevenueTTM')
                revenue_expanded = self._expand_and_align_data(
                    revenue_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                revenue_series = revenue_expanded['RevenueTTM']
            
            # 对齐数据
            revenue_aligned, market_cap_aligned = revenue_series.align(market_cap, join='inner')
            
            # 计算SP比率
            sp_ratio = self._safe_division(revenue_aligned, market_cap_aligned)
            
            # 预处理
            sp_ratio = self.preprocess(sp_ratio)
            
            return sp_ratio
            
        except Exception as e:
            logger.error(f"计算SP比率失败: {e}")
            return pd.Series(dtype=float)


class CFPRatioFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    CFP比率因子（Cash Flow to Price）
    计算公式：CFP = 经营现金流TTM / 市值
    这是PCF倒数，数值越高表示估值越低（价值股特征）
    """
    
    def __init__(self):
        super().__init__(name='CFP_Ratio', category='value')
        self.description = "现金流收益率 - 经营现金流TTM/市值"
        
    def calculate(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算CFP比率
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含经营现金流和季度信息
        market_cap : pd.Series
            市值数据，MultiIndex[TradingDates, StockCodes]
        
        Returns:
        --------
        pd.Series : CFP因子值
        """
        try:
            if not self.validate_data_requirements(financial_data, ['operating_cashflow', 'quarter']):
                raise ValueError("Required data not available for CFP calculation")
                
            extracted_data = self.extract_required_data(
                financial_data, required_columns=['operating_cashflow', 'quarter']
            )
            
            # 计算经营现金流TTM
            cashflow_data = extracted_data[['operating_cashflow', 'quarter']].copy()
            cashflow_data = cashflow_data.rename(columns={
                'operating_cashflow': 'NET_CASH_FLOWS_OPER_ACT', 
                'quarter': 'd_quarter'
            })
            
            cashflow_ttm = FinancialReportProcessor.calculate_ttm(cashflow_data)
            cashflow_series = cashflow_ttm.iloc[:, 0] if cashflow_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 扩展财务数据到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                cashflow_df = cashflow_series.to_frame(name='CashflowTTM')
                cashflow_expanded = self._expand_and_align_data(
                    cashflow_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                cashflow_series = cashflow_expanded['CashflowTTM']
            
            # 对齐数据
            cashflow_aligned, market_cap_aligned = cashflow_series.align(market_cap, join='inner')
            
            # 计算CFP比率
            cfp_ratio = self._safe_division(cashflow_aligned, market_cap_aligned)
            
            # 预处理
            cfp_ratio = self.preprocess(cfp_ratio)
            
            return cfp_ratio
            
        except Exception as e:
            logger.error(f"计算CFP比率失败: {e}")
            return pd.Series(dtype=float)


class EarningsYieldFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    盈利收益率因子（EBIT/Enterprise Value）
    更精确的价值衡量指标，考虑了债务的影响
    """
    
    def __init__(self):
        super().__init__(name='EarningsYield', category='value')
        self.description = "盈利收益率 - EBIT/企业价值"
        
    def calculate(self, financial_data: pd.DataFrame, market_cap: pd.Series, 
                  total_debt: Optional[pd.Series] = None, cash: Optional[pd.Series] = None,
                  **kwargs) -> pd.Series:
        """
        计算盈利收益率
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含营业利润信息
        market_cap : pd.Series
            市值数据
        total_debt : pd.Series, optional
            总债务数据，如果提供则计算企业价值
        cash : pd.Series, optional
            现金数据，如果提供则计算企业价值
        
        Returns:
        --------
        pd.Series : 盈利收益率因子值
        """
        try:
            if not self.validate_data_requirements(financial_data, ['operating_profit', 'quarter']):
                # 如果没有营业利润，用净利润代替EBIT
                if not self.validate_data_requirements(financial_data, ['earnings', 'quarter']):
                    raise ValueError("Required data not available for EarningsYield calculation")
                profit_field = 'earnings'
                profit_rename = 'DEDUCTEDPROFIT'
            else:
                profit_field = 'operating_profit'
                profit_rename = 'OPER_PROFIT'
                
            extracted_data = self.extract_required_data(
                financial_data, required_columns=[profit_field, 'quarter']
            )
            
            # 计算利润TTM
            profit_data = extracted_data[[profit_field, 'quarter']].copy()
            profit_data = profit_data.rename(columns={
                profit_field: profit_rename, 
                'quarter': 'd_quarter'
            })
            
            profit_ttm = FinancialReportProcessor.calculate_ttm(profit_data)
            profit_series = profit_ttm.iloc[:, 0] if profit_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 扩展财务数据到日频
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                profit_df = profit_series.to_frame(name='ProfitTTM')
                profit_expanded = self._expand_and_align_data(
                    profit_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                profit_series = profit_expanded['ProfitTTM']
            
            # 计算企业价值 = 市值 + 净债务
            enterprise_value = market_cap.copy()
            if total_debt is not None:
                enterprise_value = enterprise_value + total_debt
            if cash is not None:
                enterprise_value = enterprise_value - cash
            
            # 对齐数据
            profit_aligned, ev_aligned = profit_series.align(enterprise_value, join='inner')
            
            # 计算盈利收益率
            earnings_yield = self._safe_division(profit_aligned, ev_aligned)
            
            # 预处理
            earnings_yield = self.preprocess(earnings_yield)
            
            return earnings_yield
            
        except Exception as e:
            logger.error(f"计算盈利收益率失败: {e}")
            return pd.Series(dtype=float)