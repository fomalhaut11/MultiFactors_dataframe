#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
估值因子计算模块
需要财务数据和市值数据结合的混合因子

估值因子类别：
===========
1. BP (Book-to-Price): 净资产市值比
2. EP_ttm (Earnings-to-Price TTM): 净利润市值比
3. SP_ttm (Sales-to-Price TTM): 营收市值比  
4. CFP_ttm (CashFlow-to-Price TTM): 现金流市值比

使用示例：
=========
from factors.generator.mixed import ValuationFactorCalculator

# 创建计算器
calculator = ValuationFactorCalculator()

# 计算单个因子
bp = calculator.calculate_BP(financial_data, market_cap)

# 批量计算因子
factors = calculator.calculate_multiple_factors(
    ['BP', 'EP_ttm'], financial_data, market_cap
)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any
import logging

from ...base.factor_base import FactorBase
from ..financial.financial_report_processor import FinancialReportProcessor
from ...base.data_processing_mixin import DataProcessingMixin
from ...base.flexible_data_adapter import ColumnMapperMixin
# from ..financial.pure_financial_factors import PureFinancialFactorCalculator  # 暂时注释

logger = logging.getLogger(__name__)


class ValuationFactorCalculator(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    估值因子计算器
    
    计算需要财务数据和市值数据结合的估值因子
    """
    
    def __init__(self):
        super().__init__(name='ValuationFactorCalculator', category='mixed')
        self.description = "Calculator for valuation factors requiring financial and market data"
        
        # 初始化纯财务因子计算器，用于获取基础财务数据
        self.financial_calculator = PureFinancialFactorCalculator()
        
        # 因子分类
        self.factor_categories = {
            'valuation': [
                'BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm'
            ]
        }
        
        # 注册所有因子方法
        self.factor_methods = self._register_all_factors()
        
    def _register_all_factors(self) -> Dict[str, callable]:
        """注册所有因子计算方法"""
        methods = {
            'BP': self.calculate_BP,
            'EP_ttm': self.calculate_EP_ttm,
            'SP_ttm': self.calculate_SP_ttm,
            'CFP_ttm': self.calculate_CFP_ttm,
        }
        return methods
    
    def get_available_factors(self) -> List[str]:
        """获取所有可用因子列表"""
        return list(self.factor_methods.keys())
    
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        return self.factor_categories.copy()

    # =====================================================
    # 估值因子计算方法
    # =====================================================
    
    def calculate_BP(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算BP因子（净资产市值比）
        
        计算公式：BP = 净资产 / 市值
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，MultiIndex格式
        market_cap : pd.Series  
            市值数据，MultiIndex格式
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        pd.Series
            BP因子值，MultiIndex格式
        """
        try:
            logger.info("计算BP因子（净资产市值比）")
            
            # 先计算净资产
            book_value = self._calculate_book_value(financial_data, **kwargs)
            
            if book_value.empty:
                logger.warning("净资产数据为空，无法计算BP因子")
                return pd.Series(dtype=float, name='BP')
            
            # 数据对齐
            book_value_aligned, market_cap_aligned = book_value.align(market_cap, join='inner')
            
            if book_value_aligned.empty or market_cap_aligned.empty:
                logger.warning("净资产和市值数据对齐后为空")
                return pd.Series(dtype=float, name='BP')
            
            # 计算BP = 净资产 / 市值
            bp = self._safe_division(book_value_aligned, market_cap_aligned)
            bp.name = 'BP'
            
            # 预处理
            bp = self.preprocess(bp)
            
            logger.info(f"BP因子计算完成，数据点数: {len(bp)}")
            return bp
            
        except Exception as e:
            logger.error(f"BP因子计算失败: {e}")
            return pd.Series(dtype=float, name='BP')
    
    def calculate_EP_ttm(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算EP_ttm因子（净利润TTM市值比）
        
        计算公式：EP_ttm = 净利润TTM / 市值
        """
        try:
            logger.info("计算EP_ttm因子（净利润TTM市值比）")
            
            # 计算净利润TTM
            earnings_ttm = self._calculate_earnings_ttm(financial_data, **kwargs)
            
            if earnings_ttm.empty:
                logger.warning("净利润TTM数据为空，无法计算EP_ttm因子")
                return pd.Series(dtype=float, name='EP_ttm')
            
            # 数据对齐
            earnings_aligned, market_cap_aligned = earnings_ttm.align(market_cap, join='inner')
            
            if earnings_aligned.empty or market_cap_aligned.empty:
                logger.warning("净利润TTM和市值数据对齐后为空")
                return pd.Series(dtype=float, name='EP_ttm')
            
            # 计算EP_ttm = 净利润TTM / 市值
            ep_ttm = self._safe_division(earnings_aligned, market_cap_aligned)
            ep_ttm.name = 'EP_ttm'
            
            # 预处理
            ep_ttm = self.preprocess(ep_ttm)
            
            logger.info(f"EP_ttm因子计算完成，数据点数: {len(ep_ttm)}")
            return ep_ttm
            
        except Exception as e:
            logger.error(f"EP_ttm因子计算失败: {e}")
            return pd.Series(dtype=float, name='EP_ttm')
    
    def calculate_SP_ttm(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算SP_ttm因子（营业收入TTM市值比）
        
        计算公式：SP_ttm = 营业收入TTM / 市值
        """
        try:
            logger.info("计算SP_ttm因子（营业收入TTM市值比）")
            
            # 计算营业收入TTM
            revenue_ttm = self._calculate_revenue_ttm(financial_data, **kwargs)
            
            if revenue_ttm.empty:
                logger.warning("营业收入TTM数据为空，无法计算SP_ttm因子")
                return pd.Series(dtype=float, name='SP_ttm')
            
            # 数据对齐
            revenue_aligned, market_cap_aligned = revenue_ttm.align(market_cap, join='inner')
            
            if revenue_aligned.empty or market_cap_aligned.empty:
                logger.warning("营业收入TTM和市值数据对齐后为空")
                return pd.Series(dtype=float, name='SP_ttm')
            
            # 计算SP_ttm = 营业收入TTM / 市值
            sp_ttm = self._safe_division(revenue_aligned, market_cap_aligned)
            sp_ttm.name = 'SP_ttm'
            
            # 预处理
            sp_ttm = self.preprocess(sp_ttm)
            
            logger.info(f"SP_ttm因子计算完成，数据点数: {len(sp_ttm)}")
            return sp_ttm
            
        except Exception as e:
            logger.error(f"SP_ttm因子计算失败: {e}")
            return pd.Series(dtype=float, name='SP_ttm')
    
    def calculate_CFP_ttm(self, financial_data: pd.DataFrame, market_cap: pd.Series, **kwargs) -> pd.Series:
        """
        计算CFP_ttm因子（经营现金流TTM市值比）
        
        计算公式：CFP_ttm = 经营现金流TTM / 市值
        """
        try:
            logger.info("计算CFP_ttm因子（经营现金流TTM市值比）")
            
            # 计算经营现金流TTM
            cashflow_ttm = self._calculate_operating_cashflow_ttm(financial_data, **kwargs)
            
            if cashflow_ttm.empty:
                logger.warning("经营现金流TTM数据为空，无法计算CFP_ttm因子")
                return pd.Series(dtype=float, name='CFP_ttm')
            
            # 数据对齐
            cashflow_aligned, market_cap_aligned = cashflow_ttm.align(market_cap, join='inner')
            
            if cashflow_aligned.empty or market_cap_aligned.empty:
                logger.warning("经营现金流TTM和市值数据对齐后为空")
                return pd.Series(dtype=float, name='CFP_ttm')
            
            # 计算CFP_ttm = 经营现金流TTM / 市值
            cfp_ttm = self._safe_division(cashflow_aligned, market_cap_aligned)
            cfp_ttm.name = 'CFP_ttm'
            
            # 预处理
            cfp_ttm = self.preprocess(cfp_ttm)
            
            logger.info(f"CFP_ttm因子计算完成，数据点数: {len(cfp_ttm)}")
            return cfp_ttm
            
        except Exception as e:
            logger.error(f"CFP_ttm因子计算失败: {e}")
            return pd.Series(dtype=float, name='CFP_ttm')

    # =====================================================
    # 基础财务数据计算方法（内部使用）
    # =====================================================
    
    def _calculate_book_value(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算净资产（内部方法）"""
        try:
            # 使用PureFinancialFactorCalculator获取净资产数据
            if not self.validate_data_requirements(financial_data, ['equity']):
                raise ValueError("Required data not available for book value calculation")
                
            extracted_data = self.extract_required_data(
                financial_data, required_columns=['equity']
            )
            
            book_value = extracted_data['equity']
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                book_value_df = book_value.to_frame(name='BookValue')
                book_value_expanded = self._expand_and_align_data(
                    book_value_df,
                    kwargs['release_dates'],
                    kwargs['trading_dates']
                )
                return book_value_expanded['BookValue']
                
            return book_value
            
        except Exception as e:
            logger.error(f"净资产计算失败: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_earnings_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算净利润TTM（内部方法）"""
        try:
            # 调用PureFinancialFactorCalculator的ROE_ttm计算中的逻辑
            if not self.validate_data_requirements(financial_data, ['earnings', 'quarter']):
                raise ValueError("Required data not available for earnings TTM calculation")
                
            extracted_data = self.extract_required_data(
                financial_data, required_columns=['earnings', 'quarter']
            )
            
            # 净利润数据
            earnings = extracted_data[['earnings', 'quarter']].copy()
            earnings = earnings.rename(columns={'earnings': 'DEDUCTEDPROFIT', 'quarter': 'd_quarter'})
            earnings_ttm = TimeSeriesProcessor.calculate_ttm(earnings)
            
            earnings_series = earnings_ttm.iloc[:, 0] if earnings_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                earnings_df = earnings_series.to_frame(name='EarningsTTM')
                earnings_expanded = self._expand_and_align_data(
                    earnings_df,
                    kwargs['release_dates'],
                    kwargs['trading_dates']
                )
                return earnings_expanded['EarningsTTM']
                
            return earnings_series
            
        except Exception as e:
            logger.error(f"净利润TTM计算失败: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_revenue_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算营业收入TTM（内部方法）"""
        try:
            if not self.validate_data_requirements(financial_data, ['revenue', 'quarter']):
                raise ValueError("Required data not available for revenue TTM calculation")
                
            extracted_data = self.extract_required_data(
                financial_data, required_columns=['revenue', 'quarter']
            )
            
            revenue = extracted_data[['revenue', 'quarter']].copy()
            revenue = revenue.rename(columns={'revenue': 'TOT_OPER_REV', 'quarter': 'd_quarter'})
            
            # TTM处理
            revenue_ttm = TimeSeriesProcessor.calculate_ttm(revenue)
            revenue_series = revenue_ttm.iloc[:, 0] if revenue_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                revenue_df = revenue_series.to_frame(name='RevenueTTM')
                revenue_expanded = self._expand_and_align_data(
                    revenue_df,
                    kwargs['release_dates'],
                    kwargs['trading_dates']
                )
                return revenue_expanded['RevenueTTM']
                
            return revenue_series
            
        except Exception as e:
            logger.error(f"营业收入TTM计算失败: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_operating_cashflow_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算经营现金流TTM（内部方法）"""
        try:
            if not self.validate_data_requirements(financial_data, ['operating_cashflow', 'quarter']):
                raise ValueError("Required data not available for operating cashflow TTM calculation")
                
            extracted_data = self.extract_required_data(
                financial_data, required_columns=['operating_cashflow', 'quarter']
            )
            
            cashflow = extracted_data[['operating_cashflow', 'quarter']].copy()
            cashflow = cashflow.rename(columns={'operating_cashflow': 'NET_CASH_FLOWS_OPER_ACT', 'quarter': 'd_quarter'})
            
            # TTM处理
            cashflow_ttm = TimeSeriesProcessor.calculate_ttm(cashflow)
            cashflow_series = cashflow_ttm.iloc[:, 0] if cashflow_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                cashflow_df = cashflow_series.to_frame(name='CashflowTTM')
                cashflow_expanded = self._expand_and_align_data(
                    cashflow_df,
                    kwargs['release_dates'],
                    kwargs['trading_dates']
                )
                return cashflow_expanded['CashflowTTM']
                
            return cashflow_series
            
        except Exception as e:
            logger.error(f"经营现金流TTM计算失败: {e}")
            return pd.Series(dtype=float)

    # =====================================================
    # 批量计算接口
    # =====================================================

    def calculate_multiple_factors(self,
                                 factor_names: List[str],
                                 financial_data: pd.DataFrame,
                                 market_cap: pd.Series,
                                 **kwargs) -> pd.DataFrame:
        """
        批量计算多个估值因子
        
        Parameters:
        -----------
        factor_names : 要计算的因子名称列表
        financial_data : 财务数据
        market_cap : 市值数据
        **kwargs : 其他参数
        
        Returns:
        --------
        DataFrame: 包含所有因子的数据框
        """
        results = {}
        
        for factor_name in factor_names:
            if factor_name not in self.factor_methods:
                logger.warning(f"未知因子: {factor_name}")
                continue
                
            try:
                logger.info(f"计算估值因子: {factor_name}")
                method = self.factor_methods[factor_name]
                result = method(financial_data, market_cap, **kwargs)
                results[factor_name] = result
                logger.info(f"因子 {factor_name} 计算完成，数据点: {len(result)}")
                
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 失败: {e}")
                
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()

    def calculate(self, 
                 financial_data: pd.DataFrame,
                 market_cap: pd.Series,
                 factor_names: Union[str, List[str]] = None,
                 **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        主计算接口
        
        Parameters:
        -----------
        financial_data : 财务数据
        market_cap : 市值数据
        factor_names : 要计算的因子列表，None表示计算所有因子
        **kwargs : 其他参数
        
        Returns:
        --------
        单个因子返回Series，多个因子返回DataFrame
        """
        if factor_names is None:
            factor_names = self.get_available_factors()
        elif isinstance(factor_names, str):
            factor_names = [factor_names]
            
        if len(factor_names) == 1:
            method = self.factor_methods.get(factor_names[0])
            if method is None:
                raise ValueError(f"Unknown factor: {factor_names[0]}")
            return method(financial_data, market_cap, **kwargs)
        else:
            return self.calculate_multiple_factors(factor_names, financial_data, market_cap, **kwargs)

    def get_factor_info(self, factor_name: str = None) -> Dict[str, str]:
        """获取因子信息和说明"""
        info_map = {
            'BP': 'BP - 净资产/市值，反映账面价值相对市值的比率',
            'EP_ttm': 'EP_ttm - 净利润TTM/市值，反映盈利能力相对市值的比率',
            'SP_ttm': 'SP_ttm - 营业收入TTM/市值，反映营收能力相对市值的比率',
            'CFP_ttm': 'CFP_ttm - 经营现金流TTM/市值，反映现金流生成能力相对市值的比率',
        }
        
        if factor_name:
            return {factor_name: info_map.get(factor_name, '暂无描述')}
        return info_map


# =====================================================
# 便捷函数接口
# =====================================================

def calculate_valuation_factor(factor_name: str, 
                             financial_data: pd.DataFrame,
                             market_cap: pd.Series,
                             **kwargs) -> pd.Series:
    """计算单个估值因子的便捷函数"""
    calculator = ValuationFactorCalculator()
    return calculator.calculate(financial_data, market_cap, factor_names=factor_name, **kwargs)


def calculate_multiple_valuation_factors(factor_names: List[str],
                                        financial_data: pd.DataFrame,
                                        market_cap: pd.Series,
                                        **kwargs) -> pd.DataFrame:
    """计算多个估值因子的便捷函数"""
    calculator = ValuationFactorCalculator()
    return calculator.calculate_multiple_factors(factor_names, financial_data, market_cap, **kwargs)


def get_all_valuation_factors() -> List[str]:
    """获取所有可用估值因子列表"""
    calculator = ValuationFactorCalculator()
    return calculator.get_available_factors()