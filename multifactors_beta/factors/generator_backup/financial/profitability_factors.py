#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盈利能力因子实现
包含ROE、ROA、利润率等盈利能力相关因子
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from ...base.factor_base import FactorBase
from ...base.data_processing_mixin import DataProcessingMixin
from ...base.flexible_data_adapter import ColumnMapperMixin
from .financial_report_processor import FinancialReportProcessor

logger = logging.getLogger(__name__)


class ROE_ttm_Factor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """ROE_ttm因子：TTM净资产收益率"""
    
    def __init__(self):
        super().__init__(name='ROE_ttm', category='profitability')
        self.description = "TTM净资产收益率 - 扣非净利润TTM/股东权益均值"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROE_ttm（TTM净资产收益率）"""
        try:
            # 使用实际的字段名
            earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
            equity_col = 'EQY_BELONGTO_PARCOMSH'  # 归属母公司股东权益
            
            # 检查必要字段是否存在
            required_cols = [earnings_col, equity_col, 'd_year', 'd_quarter']
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 使用新的FinancialReportProcessor接口计算TTM
            ttm_results = FinancialReportProcessor.calculate_ttm(financial_data)
            earnings_ttm = ttm_results[f'{earnings_col}_ttm']
            
            # 股东权益取最新值（季末值）
            equity_data = financial_data[equity_col]
            
            # 对齐数据并计算ROE
            earnings_aligned, equity_aligned = earnings_ttm.align(equity_data, join='inner')
            
            # ROE = 净利润TTM / 股东权益
            roe = self._safe_division(earnings_aligned, equity_aligned)
            
            # 过滤异常值
            roe = roe.replace([np.inf, -np.inf], np.nan)
            roe = self.preprocess(roe)
            
            return roe
            
        except Exception as e:
            logger.error(f"计算ROE_ttm失败: {e}")
            return pd.Series(dtype=float)


class ROA_ttm_Factor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """ROA_ttm因子：TTM总资产收益率"""
    
    def __init__(self):
        super().__init__(name='ROA_ttm', category='profitability')
        self.description = "TTM总资产收益率 - 扣非净利润TTM/总资产均值"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROA_ttm（TTM总资产收益率）"""
        try:
            # 使用实际的字段名
            earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
            assets_col = 'TOT_ASSETS'  # 总资产
            
            # 检查必要字段是否存在
            required_cols = [earnings_col, assets_col, 'd_year', 'd_quarter']
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 使用新的FinancialReportProcessor接口计算TTM
            ttm_results = FinancialReportProcessor.calculate_ttm(financial_data)
            earnings_ttm = ttm_results[f'{earnings_col}_ttm']
            
            # 总资产取最新值（季末值）
            assets_data = financial_data[assets_col]
            
            # 对齐数据并计算ROA
            earnings_aligned, assets_aligned = earnings_ttm.align(assets_data, join='inner')
            
            # ROA = 净利润TTM / 总资产
            roa = self._safe_division(earnings_aligned, assets_aligned)
            
            # 过滤异常值
            roa = roa.replace([np.inf, -np.inf], np.nan)
            roa = self.preprocess(roa)
            
            return roa
            
        except Exception as e:
            logger.error(f"计算ROA_ttm失败: {e}")
            return pd.Series(dtype=float)


class GrossProfitMargin_ttm_Factor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """毛利率TTM因子"""
    
    def __init__(self):
        super().__init__(name='GrossProfitMargin_ttm', category='profitability')
        self.description = "毛利率TTM - (营业收入-营业成本)TTM/营业收入TTM"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算毛利率_ttm"""
        if not self.validate_data_requirements(financial_data, ['revenue', 'cost', 'quarter']):
            raise ValueError("Required data not available for GrossProfitMargin_ttm calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['revenue', 'cost', 'quarter']
        )
        
        revenue = extracted_data[['revenue', 'quarter']].copy()
        cost = extracted_data[['cost', 'quarter']].copy()
        
        revenue = revenue.rename(columns={'revenue': 'TOT_OPER_REV', 'quarter': 'd_quarter'})
        cost = cost.rename(columns={'cost': 'TOT_OPER_COST', 'quarter': 'd_quarter'})
        
        # TTM处理
        revenue_ttm = FinancialReportProcessor.calculate_ttm(revenue)
        cost_ttm = FinancialReportProcessor.calculate_ttm(cost)
        
        # 计算毛利率
        revenue_series = revenue_ttm.iloc[:, 0]
        cost_series = cost_ttm.iloc[:, 0]
        
        gross_profit_margin = self._safe_division(revenue_series - cost_series, revenue_series)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            gross_profit_margin = self._expand_to_daily_if_needed(
                gross_profit_margin, 'GrossProfitMargin_ttm', **kwargs)
            
        return gross_profit_margin


class ProfitCost_ttm_Factor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """利润成本比率TTM因子"""
    
    def __init__(self):
        super().__init__(name='ProfitCost_ttm', category='profitability')
        self.description = "利润成本比率TTM - TTM扣非净利润/(TTM财务费用+TTM所得税)"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ProfitCost_ttm（扣非净利润相对成本比率）
        
        计算公式：TTM扣非净利润 / (TTM财务费用 + TTM所得税)
        反映企业扣非净利润相对于财务成本和税收成本的效率
        """
        if not self.validate_data_requirements(
            financial_data, ['earnings', 'financial_expense', 'income_tax', 'quarter']
        ):
            raise ValueError("Required data not available for ProfitCost_ttm calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, 
            required_columns=['earnings', 'financial_expense', 'income_tax', 'quarter']
        )
        
        # 准备TTM计算所需数据
        # 扣非净利润数据
        earnings_data = extracted_data[['earnings', 'quarter']].copy()
        earnings_data = earnings_data.rename(columns={
            'earnings': 'DEDUCTEDPROFIT', 
            'quarter': 'd_quarter'
        })
        
        # 财务费用数据
        fin_exp_data = extracted_data[['financial_expense', 'quarter']].copy()
        fin_exp_data = fin_exp_data.rename(columns={
            'financial_expense': 'FIN_EXP_IS',
            'quarter': 'd_quarter'
        })
        
        # 所得税数据
        tax_data = extracted_data[['income_tax', 'quarter']].copy()
        tax_data = tax_data.rename(columns={
            'income_tax': 'TAX',
            'quarter': 'd_quarter'
        })
        
        # 计算TTM值
        earnings_ttm = FinancialReportProcessor.calculate_ttm(earnings_data)
        fin_exp_ttm = FinancialReportProcessor.calculate_ttm(fin_exp_data)
        tax_ttm = FinancialReportProcessor.calculate_ttm(tax_data)
        
        # 提取数值序列
        earnings_series = earnings_ttm.iloc[:, 0] if earnings_ttm.shape[1] > 0 else pd.Series(dtype=float)
        fin_exp_series = fin_exp_ttm.iloc[:, 0] if fin_exp_ttm.shape[1] > 0 else pd.Series(dtype=float)
        tax_series = tax_ttm.iloc[:, 0] if tax_ttm.shape[1] > 0 else pd.Series(dtype=float)
        
        # 数据对齐
        earnings_aligned, fin_exp_aligned = earnings_series.align(fin_exp_series, join='inner')
        earnings_aligned, tax_aligned = earnings_aligned.align(tax_series, join='inner')
        fin_exp_aligned, tax_aligned = fin_exp_aligned.align(tax_aligned, join='inner')
        
        # 计算分母：财务费用 + 所得税
        total_cost = fin_exp_aligned + tax_aligned
        
        # 计算ProfitCost：扣非净利润 / (财务费用 + 所得税)
        profitcost = self._safe_division(earnings_aligned, total_cost)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            profitcost_df = profitcost.to_frame(name='ProfitCost')
            profitcost_expanded = self._expand_and_align_data(
                profitcost_df,
                kwargs['release_dates'],
                kwargs['trading_dates']
            )
            return profitcost_expanded['ProfitCost']
        
        return profitcost