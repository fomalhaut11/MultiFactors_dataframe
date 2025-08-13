#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯财务因子计算模块
包含所有仅依赖财务报表数据的因子，不依赖市值、价格等市场数据

因子分类：
=========
1. 盈利能力因子 (13个)
   - ROE_ttm, ROE_lyr: 净资产收益率
   - ROA_ttm, ROA_lyr: 总资产收益率  
   - ROIC_ttm: 投入资本收益率
   - GrossProfitMargin_ttm: 毛利率
   - NetProfitMargin_ttm: 净利率
   - OperatingMargin_ttm: 营业利润率
   - EBITDAMargin_ttm: EBITDA利润率
   - InterestMargin_ttm: 净息差（银行业）
   - CostIncomeRatio_ttm: 成本收入比

2. 偿债能力因子 (8个)
   - CurrentRatio: 流动比率
   - QuickRatio: 速动比率
   - CashRatio: 现金比率
   - DebtToAssets: 资产负债率
   - DebtToEquity: 产权比率
   - EquityMultiplier: 权益乘数
   - InterestCoverage_ttm: 利息保障倍数
   - DebtServiceCoverage_ttm: 债务偿付比率

3. 营运效率因子 (9个)
   - AssetTurnover_ttm: 总资产周转率
   - EquityTurnover_ttm: 净资产周转率
   - InventoryTurnover_ttm: 存货周转率
   - AccountsReceivableTurnover_ttm: 应收账款周转率
   - AccountsPayableTurnover_ttm: 应付账款周转率
   - CashCycle_ttm: 现金转换周期
   - WorkingCapitalTurnover_ttm: 营运资本周转率
   - FixedAssetTurnover_ttm: 固定资产周转率

4. 成长能力因子 (10个)
   - RevenueGrowth_yoy: 营收同比增长率
   - NetIncomeGrowth_yoy: 净利润同比增长率
   - TotalAssetsGrowth_yoy: 总资产同比增长率
   - EquityGrowth_yoy: 净资产同比增长率
   - ROEGrowth_yoy: ROE同比增长率
   - OperatingCashFlowGrowth_yoy: 经营现金流同比增长率
   - RevenueGrowth_3y: 营收3年复合增长率
   - NetIncomeGrowth_3y: 净利润3年复合增长率

5. 现金流因子 (7个)
   - OperatingCashFlowRatio_ttm: 经营现金流比率
   - FreeCashFlowMargin_ttm: 自由现金流利润率
   - CashFlowToDebt_ttm: 现金流负债比
   - OperatingCashFlowToRevenue_ttm: 经营现金流收入比
   - CapexToRevenue_ttm: 资本开支收入比
   - CashFlowCoverage_ttm: 现金流覆盖率

6. 资产质量因子 (8个)
   - AssetQuality: 资产质量综合评分
   - TangibleAssetRatio: 有形资产比率
   - GoodwillRatio: 商誉资产比
   - AccrualsRatio_ttm: 应计项目比率
   - WorkingCapitalRatio: 营运资本比率
   - NonCurrentAssetRatio: 非流动资产比率

7. 盈利质量因子 (6个)
   - EarningsQuality_ttm: 盈利质量（现金流/净利润）
   - AccrualQuality_ttm: 应计质量
   - EarningsStability_5y: 盈利稳定性（5年ROE标准差）
   - EarningsPersistence: 盈利持续性
   - OperatingLeverage: 经营杠杆系数

使用示例：
=========
from factors.financial.pure_financial_factors import PureFinancialFactorCalculator

# 创建计算器
calculator = PureFinancialFactorCalculator()

# 查看所有可用因子
print(calculator.get_available_factors())

# 计算单个因子
roe = calculator.calculate_ROE_ttm(financial_data, **kwargs)

# 批量计算因子
factors = calculator.calculate_multiple_factors(
    ['ROE_ttm', 'CurrentRatio', 'AssetTurnover_ttm'], 
    financial_data
)

# 按类别计算因子
profitability_factors = calculator.calculate_by_category('profitability', financial_data)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any, Tuple
import logging
from datetime import datetime, timedelta

from ...base.factor_base import FactorBase
from ...base.time_series_processor import TimeSeriesProcessor
from ...base.data_processing_mixin import DataProcessingMixin
from ...base.flexible_data_adapter import ColumnMapperMixin

logger = logging.getLogger(__name__)


class PureFinancialFactorCalculator(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    纯财务因子计算器
    
    专门计算仅依赖财务报表数据的因子
    不依赖市值、价格、成交量等市场数据
    """
    
    def __init__(self):
        super().__init__(name='PureFinancialFactorCalculator', category='financial')
        self.description = "Calculator for financial statement based factors only"
        
        # 因子分类
        self.factor_categories = {
            'profitability': [
                'ROE_ttm', 'ROE_lyr', 'ROA_ttm', 'ROA_lyr', 'ROIC_ttm',
                'GrossProfitMargin_ttm', 'NetProfitMargin_ttm', 'OperatingMargin_ttm',
                'EBITDAMargin_ttm', 'InterestMargin_ttm', 'CostIncomeRatio_ttm'
            ],
            'solvency': [
                'CurrentRatio', 'QuickRatio', 'CashRatio', 'DebtToAssets', 
                'DebtToEquity', 'EquityMultiplier', 'InterestCoverage_ttm', 
                'DebtServiceCoverage_ttm'
            ],
            'efficiency': [
                'AssetTurnover_ttm', 'EquityTurnover_ttm', 'InventoryTurnover_ttm',
                'AccountsReceivableTurnover_ttm', 'AccountsPayableTurnover_ttm',
                'CashCycle_ttm', 'WorkingCapitalTurnover_ttm', 'FixedAssetTurnover_ttm'
            ],
            'growth': [
                'RevenueGrowth_yoy', 'NetIncomeGrowth_yoy', 'TotalAssetsGrowth_yoy',
                'EquityGrowth_yoy', 'ROEGrowth_yoy', 'OperatingCashFlowGrowth_yoy',
                'RevenueGrowth_3y', 'NetIncomeGrowth_3y'
            ],
            'cashflow': [
                'OperatingCashFlowRatio_ttm', 'FreeCashFlowMargin_ttm', 'CashFlowToDebt_ttm',
                'OperatingCashFlowToRevenue_ttm', 'CapexToRevenue_ttm', 'CashFlowCoverage_ttm'
            ],
            'quality': [
                'AssetQuality', 'TangibleAssetRatio', 'GoodwillRatio', 'AccrualsRatio_ttm',
                'WorkingCapitalRatio', 'NonCurrentAssetRatio', 'EarningsQuality_ttm',
                'EarningsStability_5y'
            ]
        }
        
        # 注册所有因子方法
        self.factor_methods = self._register_all_factors()
        
    def _register_all_factors(self) -> Dict[str, callable]:
        """注册所有因子计算方法"""
        methods = {}
        
        # 盈利能力因子
        methods.update({
            'ROE_ttm': self.calculate_ROE_ttm,
            'ROE_lyr': self.calculate_ROE_lyr,
            'ROA_ttm': self.calculate_ROA_ttm,
            'ROA_lyr': self.calculate_ROA_lyr,
            'ROIC_ttm': self.calculate_ROIC_ttm,
            'GrossProfitMargin_ttm': self.calculate_GrossProfitMargin_ttm,
            'NetProfitMargin_ttm': self.calculate_NetProfitMargin_ttm,
            'OperatingMargin_ttm': self.calculate_OperatingMargin_ttm,
            'EBITDAMargin_ttm': self.calculate_EBITDAMargin_ttm,
            'InterestMargin_ttm': self.calculate_InterestMargin_ttm,
            'CostIncomeRatio_ttm': self.calculate_CostIncomeRatio_ttm,
        })
        
        # 偿债能力因子
        methods.update({
            'CurrentRatio': self.calculate_CurrentRatio,
            'QuickRatio': self.calculate_QuickRatio,
            'CashRatio': self.calculate_CashRatio,
            'DebtToAssets': self.calculate_DebtToAssets,
            'DebtToEquity': self.calculate_DebtToEquity,
            'EquityMultiplier': self.calculate_EquityMultiplier,
            'InterestCoverage_ttm': self.calculate_InterestCoverage_ttm,
            'DebtServiceCoverage_ttm': self.calculate_DebtServiceCoverage_ttm,
        })
        
        # 营运效率因子
        methods.update({
            'AssetTurnover_ttm': self.calculate_AssetTurnover_ttm,
            'EquityTurnover_ttm': self.calculate_EquityTurnover_ttm,
            'InventoryTurnover_ttm': self.calculate_InventoryTurnover_ttm,
            'AccountsReceivableTurnover_ttm': self.calculate_AccountsReceivableTurnover_ttm,
            'AccountsPayableTurnover_ttm': self.calculate_AccountsPayableTurnover_ttm,
            'CashCycle_ttm': self.calculate_CashCycle_ttm,
            'WorkingCapitalTurnover_ttm': self.calculate_WorkingCapitalTurnover_ttm,
            'FixedAssetTurnover_ttm': self.calculate_FixedAssetTurnover_ttm,
        })
        
        # 成长能力因子
        methods.update({
            'RevenueGrowth_yoy': self.calculate_RevenueGrowth_yoy,
            'NetIncomeGrowth_yoy': self.calculate_NetIncomeGrowth_yoy,
            'TotalAssetsGrowth_yoy': self.calculate_TotalAssetsGrowth_yoy,
            'EquityGrowth_yoy': self.calculate_EquityGrowth_yoy,
            'ROEGrowth_yoy': self.calculate_ROEGrowth_yoy,
            'OperatingCashFlowGrowth_yoy': self.calculate_OperatingCashFlowGrowth_yoy,
            'RevenueGrowth_3y': self.calculate_RevenueGrowth_3y,
            'NetIncomeGrowth_3y': self.calculate_NetIncomeGrowth_3y,
        })
        
        # 现金流因子
        methods.update({
            'OperatingCashFlowRatio_ttm': self.calculate_OperatingCashFlowRatio_ttm,
            'FreeCashFlowMargin_ttm': self.calculate_FreeCashFlowMargin_ttm,
            'CashFlowToDebt_ttm': self.calculate_CashFlowToDebt_ttm,
            'OperatingCashFlowToRevenue_ttm': self.calculate_OperatingCashFlowToRevenue_ttm,
            'CapexToRevenue_ttm': self.calculate_CapexToRevenue_ttm,
            'CashFlowCoverage_ttm': self.calculate_CashFlowCoverage_ttm,
        })
        
        # 质量因子
        methods.update({
            'AssetQuality': self.calculate_AssetQuality,
            'TangibleAssetRatio': self.calculate_TangibleAssetRatio,
            'GoodwillRatio': self.calculate_GoodwillRatio,
            'AccrualsRatio_ttm': self.calculate_AccrualsRatio_ttm,
            'WorkingCapitalRatio': self.calculate_WorkingCapitalRatio,
            'NonCurrentAssetRatio': self.calculate_NonCurrentAssetRatio,
            'EarningsQuality_ttm': self.calculate_EarningsQuality_ttm,
            'EarningsStability_5y': self.calculate_EarningsStability_5y,
        })
        
        return methods
    
    def get_available_factors(self) -> List[str]:
        """获取所有可用因子列表"""
        return list(self.factor_methods.keys())
    
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        return self.factor_categories.copy()
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """根据分类获取因子列表"""
        return self.factor_categories.get(category, [])

    def calculate_multiple_factors(self,
                                 factor_names: List[str],
                                 financial_data: pd.DataFrame,
                                 **kwargs) -> pd.DataFrame:
        """
        批量计算多个因子
        
        Parameters:
        -----------
        factor_names : 要计算的因子名称列表
        financial_data : 财务数据
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
                logger.info(f"计算纯财务因子: {factor_name}")
                method = self.factor_methods[factor_name]
                result = method(financial_data, **kwargs)
                results[factor_name] = result
                logger.info(f"因子 {factor_name} 计算完成，数据点: {len(result)}")
                
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 失败: {e}")
                
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()

    def calculate_by_category(self,
                            category: str,
                            financial_data: pd.DataFrame,
                            **kwargs) -> pd.DataFrame:
        """
        按分类批量计算因子
        
        Parameters:
        -----------
        category : 因子分类 ('profitability', 'solvency', 'efficiency', 'growth', 'cashflow', 'quality')
        financial_data : 财务数据
        **kwargs : 其他参数
        
        Returns:
        --------
        DataFrame: 该分类下所有因子的数据框
        """
        if category not in self.factor_categories:
            raise ValueError(f"未知分类: {category}. 可选: {list(self.factor_categories.keys())}")
        
        factor_names = self.factor_categories[category]
        return self.calculate_multiple_factors(factor_names, financial_data, **kwargs)

    # =====================================================
    # 1. 盈利能力因子 (Profitability Factors)
    # =====================================================
    
    def calculate_ROE_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROE_ttm（TTM净资产收益率）- 已实现"""
        if not self.validate_data_requirements(financial_data, ['earnings', 'equity', 'quarter']):
            raise ValueError("Required data not available for ROE_ttm calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['earnings', 'equity', 'quarter']
        )
        
        # 利润数据
        earnings = extracted_data[['earnings', 'quarter']].copy()
        earnings = earnings.rename(columns={'earnings': 'DEDUCTEDPROFIT', 'quarter': 'd_quarter'})
        earnings_ttm = TimeSeriesProcessor.calculate_ttm(earnings)
        
        # 权益数据
        equity = extracted_data[['equity']].copy()
        equity = equity.rename(columns={'equity': 'EQY_BELONGTO_PARCOMSH'})
        equity_avg = TimeSeriesProcessor.calculate_avg(equity)
        
        # 计算ROE
        earnings_series = earnings_ttm.iloc[:, 0] if earnings_ttm.shape[1] > 0 else pd.Series(dtype=float)
        equity_series = equity_avg.iloc[:, 0] if equity_avg.shape[1] > 0 else pd.Series(dtype=float)
        
        earnings_aligned, equity_aligned = earnings_series.align(equity_series, join='inner')
        roe = self._safe_division(earnings_aligned, equity_aligned)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            roe_df = roe.to_frame(name='ROE')
            roe_expanded = self._expand_and_align_data(
                roe_df,
                release_dates=kwargs['release_dates'],
                trading_dates=kwargs['trading_dates']
            )
            if isinstance(roe_expanded, pd.DataFrame):
                roe = roe_expanded.iloc[:, 0]
            else:
                roe = roe_expanded
            roe = self.preprocess(roe)
        else:
            roe = roe.replace([np.inf, -np.inf], np.nan).dropna()
            
        return roe
    
    def calculate_ROE_lyr(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROE_lyr（年报净资产收益率）"""
        # 获取年报数据（Q4）
        annual_data = financial_data[
            financial_data.index.get_level_values('ReportDates').to_series().dt.month == 12
        ]
        
        if not self.validate_data_requirements(annual_data, ['earnings', 'equity']):
            raise ValueError("Required annual data not available for ROE_lyr calculation")
            
        extracted_data = self.extract_required_data(
            annual_data, required_columns=['earnings', 'equity']
        )
        
        earnings = extracted_data[['earnings']].copy()
        equity = extracted_data[['equity']].copy()
        
        # 计算年报ROE
        earnings_series = earnings.iloc[:, 0]
        equity_avg = TimeSeriesProcessor.calculate_avg(equity).iloc[:, 0]
        
        earnings_aligned, equity_aligned = earnings_series.align(equity_avg, join='inner')
        roe_lyr = self._safe_division(earnings_aligned, equity_aligned)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            roe_lyr = self._expand_to_daily_if_needed(roe_lyr, 'ROE_lyr', **kwargs)
            
        return roe_lyr
    
    def calculate_ROA_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROA_ttm（TTM总资产收益率）"""
        if not self.validate_data_requirements(financial_data, ['earnings', 'total_assets', 'quarter']):
            raise ValueError("Required data not available for ROA_ttm calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['earnings', 'total_assets', 'quarter']
        )
        
        # 利润数据TTM
        earnings = extracted_data[['earnings', 'quarter']].copy()
        earnings = earnings.rename(columns={'earnings': 'DEDUCTEDPROFIT', 'quarter': 'd_quarter'})
        earnings_ttm = TimeSeriesProcessor.calculate_ttm(earnings)
        
        # 总资产平均
        assets = extracted_data[['total_assets']].copy()
        assets = assets.rename(columns={'total_assets': 'TOT_ASSETS'})
        assets_avg = TimeSeriesProcessor.calculate_avg(assets)
        
        # 计算ROA
        earnings_series = earnings_ttm.iloc[:, 0]
        assets_series = assets_avg.iloc[:, 0]
        
        earnings_aligned, assets_aligned = earnings_series.align(assets_series, join='inner')
        roa = self._safe_division(earnings_aligned, assets_aligned)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            roa = self._expand_to_daily_if_needed(roa, 'ROA_ttm', **kwargs)
            
        return roa
    
    def calculate_ROA_lyr(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROA_lyr（年报总资产收益率）"""
        # 实现年报ROA
        pass  # TODO: 实现
    
    def calculate_ROIC_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROIC_ttm（TTM投入资本收益率）"""
        # ROIC = NOPAT / 投入资本
        # NOPAT = 营业利润 * (1 - 税率)
        # 投入资本 = 股东权益 + 有息负债
        pass  # TODO: 实现
    
    def calculate_GrossProfitMargin_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
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
        revenue_ttm = TimeSeriesProcessor.calculate_ttm(revenue)
        cost_ttm = TimeSeriesProcessor.calculate_ttm(cost)
        
        # 计算毛利率
        revenue_series = revenue_ttm.iloc[:, 0]
        cost_series = cost_ttm.iloc[:, 0]
        
        gross_profit_margin = self._safe_division(revenue_series - cost_series, revenue_series)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            gross_profit_margin = self._expand_to_daily_if_needed(
                gross_profit_margin, 'GrossProfitMargin_ttm', **kwargs)
            
        return gross_profit_margin
    
    def calculate_NetProfitMargin_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算净利率_ttm"""
        # 净利率 = 净利润TTM / 营业收入TTM
        pass  # TODO: 实现
    
    def calculate_OperatingMargin_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算营业利润率_ttm"""
        # 营业利润率 = 营业利润TTM / 营业收入TTM
        pass  # TODO: 实现
    
    def calculate_EBITDAMargin_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算EBITDA利润率_ttm"""
        # EBITDA利润率 = EBITDA TTM / 营业收入TTM
        pass  # TODO: 实现
    
    def calculate_InterestMargin_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算净息差_ttm（银行业专用）"""
        # 净息差 = (利息收入 - 利息支出) / 生息资产平均余额
        pass  # TODO: 实现
    
    def calculate_CostIncomeRatio_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算成本收入比_ttm"""
        # 成本收入比 = 营业成本TTM / 营业收入TTM
        pass  # TODO: 实现

    # =====================================================
    # 2. 偿债能力因子 (Solvency Factors)
    # =====================================================
    
    def calculate_CurrentRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算流动比率"""
        if not self.validate_data_requirements(financial_data, ['current_assets', 'current_liabilities']):
            raise ValueError("Required data not available for CurrentRatio calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['current_assets', 'current_liabilities']
        )
        
        current_assets = extracted_data['current_assets']
        current_liabilities = extracted_data['current_liabilities']
        
        current_ratio = self._safe_division(current_assets, current_liabilities)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            current_ratio = self._expand_to_daily_if_needed(current_ratio, 'CurrentRatio', **kwargs)
            
        return current_ratio
    
    def calculate_QuickRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算速动比率"""
        # 速动比率 = (流动资产 - 存货) / 流动负债
        pass  # TODO: 实现
    
    def calculate_CashRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算现金比率"""
        # 现金比率 = (货币资金 + 短期投资) / 流动负债
        pass  # TODO: 实现
    
    def calculate_DebtToAssets(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算资产负债率"""
        if not self.validate_data_requirements(financial_data, ['total_liabilities', 'total_assets']):
            raise ValueError("Required data not available for DebtToAssets calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['total_liabilities', 'total_assets']
        )
        
        total_liabilities = extracted_data['total_liabilities']
        total_assets = extracted_data['total_assets']
        
        debt_to_assets = self._safe_division(total_liabilities, total_assets)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            debt_to_assets = self._expand_to_daily_if_needed(debt_to_assets, 'DebtToAssets', **kwargs)
            
        return debt_to_assets
    
    def calculate_DebtToEquity(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算产权比率"""
        # 产权比率 = 负债总额 / 股东权益
        pass  # TODO: 实现
    
    def calculate_EquityMultiplier(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算权益乘数"""
        # 权益乘数 = 总资产 / 股东权益
        pass  # TODO: 实现
    
    def calculate_InterestCoverage_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算利息保障倍数_ttm"""
        # 利息保障倍数 = EBIT TTM / 利息费用TTM
        pass  # TODO: 实现
    
    def calculate_DebtServiceCoverage_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算债务偿付比率_ttm"""
        # 债务偿付比率 = 经营现金流TTM / (本期偿还债务 + 利息费用)
        pass  # TODO: 实现

    # =====================================================
    # 3-6. 其他因子分类的占位方法
    # =====================================================
    
    # 3. 营运效率因子
    def calculate_AssetTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """总资产周转率_ttm"""
        pass  # TODO: 实现
        
    def calculate_EquityTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """净资产周转率_ttm"""
        pass  # TODO: 实现
        
    def calculate_InventoryTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """存货周转率_ttm"""
        pass  # TODO: 实现
        
    def calculate_AccountsReceivableTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """应收账款周转率_ttm"""
        pass  # TODO: 实现
        
    def calculate_AccountsPayableTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """应付账款周转率_ttm"""
        pass  # TODO: 实现
        
    def calculate_CashCycle_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """现金转换周期_ttm"""
        pass  # TODO: 实现
        
    def calculate_WorkingCapitalTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """营运资本周转率_ttm"""
        pass  # TODO: 实现
        
    def calculate_FixedAssetTurnover_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """固定资产周转率_ttm"""
        pass  # TODO: 实现
    
    # 4. 成长能力因子
    def calculate_RevenueGrowth_yoy(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """营收同比增长率"""
        pass  # TODO: 实现
        
    def calculate_NetIncomeGrowth_yoy(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """净利润同比增长率"""
        pass  # TODO: 实现
        
    def calculate_TotalAssetsGrowth_yoy(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """总资产同比增长率"""
        pass  # TODO: 实现
        
    def calculate_EquityGrowth_yoy(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """净资产同比增长率"""
        pass  # TODO: 实现
        
    def calculate_ROEGrowth_yoy(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """ROE同比增长率"""
        pass  # TODO: 实现
        
    def calculate_OperatingCashFlowGrowth_yoy(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """经营现金流同比增长率"""
        pass  # TODO: 实现
        
    def calculate_RevenueGrowth_3y(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """营收3年复合增长率"""
        pass  # TODO: 实现
        
    def calculate_NetIncomeGrowth_3y(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """净利润3年复合增长率"""
        pass  # TODO: 实现
    
    # 5. 现金流因子
    def calculate_OperatingCashFlowRatio_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """经营现金流比率_ttm"""
        pass  # TODO: 实现
        
    def calculate_FreeCashFlowMargin_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """自由现金流利润率_ttm"""
        pass  # TODO: 实现
        
    def calculate_CashFlowToDebt_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """现金流负债比_ttm"""
        pass  # TODO: 实现
        
    def calculate_OperatingCashFlowToRevenue_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """经营现金流收入比_ttm"""
        pass  # TODO: 实现
        
    def calculate_CapexToRevenue_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """资本开支收入比_ttm"""
        pass  # TODO: 实现
        
    def calculate_CashFlowCoverage_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """现金流覆盖率_ttm"""
        pass  # TODO: 实现
    
    # 6. 质量因子
    def calculate_AssetQuality(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """资产质量综合评分"""
        pass  # TODO: 实现
        
    def calculate_TangibleAssetRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """有形资产比率"""
        pass  # TODO: 实现
        
    def calculate_GoodwillRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """商誉资产比"""
        pass  # TODO: 实现
        
    def calculate_AccrualsRatio_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """应计项目比率_ttm"""
        pass  # TODO: 实现
        
    def calculate_WorkingCapitalRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """营运资本比率"""
        pass  # TODO: 实现
        
    def calculate_NonCurrentAssetRatio(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """非流动资产比率"""
        pass  # TODO: 实现
        
    def calculate_EarningsQuality_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """盈利质量_ttm"""
        pass  # TODO: 实现
        
    def calculate_EarningsStability_5y(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """盈利稳定性_5y"""
        pass  # TODO: 实现

    # =====================================================
    # 工具方法
    # =====================================================
    
    def _expand_to_daily_if_needed(self, factor_data: pd.Series, factor_name: str, **kwargs) -> pd.Series:
        """如果需要，扩展因子到日频"""
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            factor_df = factor_data.to_frame(name=factor_name)
            expanded = self._expand_and_align_data(
                factor_df,
                release_dates=kwargs['release_dates'],
                trading_dates=kwargs['trading_dates']
            )
            if isinstance(expanded, pd.DataFrame):
                result = expanded.iloc[:, 0]
            else:
                result = expanded
            result = self.preprocess(result)
        else:
            result = factor_data.replace([np.inf, -np.inf], np.nan).dropna()
            
        return result
    
    def get_factor_info(self, factor_name: str = None) -> Dict[str, str]:
        """获取因子信息和说明"""
        info_map = {
            # 盈利能力因子
            'ROE_ttm': 'ROE(TTM) - 扣非净利润TTM/股东权益均值',
            'ROE_lyr': 'ROE(年报) - 年报扣非净利润/股东权益均值',
            'ROA_ttm': 'ROA(TTM) - 扣非净利润TTM/总资产均值',
            'GrossProfitMargin_ttm': '毛利率(TTM) - (营业收入-营业成本)TTM/营业收入TTM',
            
            # 偿债能力因子
            'CurrentRatio': '流动比率 - 流动资产/流动负债',
            'DebtToAssets': '资产负债率 - 负债总额/资产总额',
            
            # 其他因子...
        }
        
        if factor_name:
            return {factor_name: info_map.get(factor_name, '暂无描述')}
        return info_map
    
    def calculate(self, 
                 financial_data: pd.DataFrame, 
                 factor_names: Union[str, List[str]] = None,
                 category: str = None,
                 **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        主计算接口
        
        Parameters:
        -----------
        financial_data : 财务数据
        factor_names : 要计算的因子列表，None表示计算所有因子
        category : 因子分类，如果指定则计算该分类下所有因子
        **kwargs : 其他参数
        
        Returns:
        --------
        单个因子返回Series，多个因子返回DataFrame
        """
        if category:
            return self.calculate_by_category(category, financial_data, **kwargs)
        
        if factor_names is None:
            factor_names = self.get_available_factors()
        elif isinstance(factor_names, str):
            factor_names = [factor_names]
            
        if len(factor_names) == 1:
            method = self.factor_methods.get(factor_names[0])
            if method is None:
                raise ValueError(f"Unknown factor: {factor_names[0]}")
            return method(financial_data, **kwargs)
        else:
            return self.calculate_multiple_factors(factor_names, financial_data, **kwargs)


# =====================================================
# 便捷函数接口
# =====================================================

def calculate_pure_financial_factor(factor_name: str, 
                                   financial_data: pd.DataFrame,
                                   **kwargs) -> pd.Series:
    """计算单个纯财务因子的便捷函数"""
    calculator = PureFinancialFactorCalculator()
    return calculator.calculate(financial_data, factor_names=factor_name, **kwargs)


def calculate_multiple_pure_financial_factors(factor_names: List[str],
                                            financial_data: pd.DataFrame,
                                            **kwargs) -> pd.DataFrame:
    """计算多个纯财务因子的便捷函数"""
    calculator = PureFinancialFactorCalculator()
    return calculator.calculate_multiple_factors(factor_names, financial_data, **kwargs)


def calculate_financial_factors_by_category(category: str,
                                          financial_data: pd.DataFrame,
                                          **kwargs) -> pd.DataFrame:
    """按分类计算纯财务因子的便捷函数"""
    calculator = PureFinancialFactorCalculator()
    return calculator.calculate_by_category(category, financial_data, **kwargs)


def get_all_pure_financial_factors() -> List[str]:
    """获取所有可用纯财务因子列表"""
    calculator = PureFinancialFactorCalculator()
    return calculator.get_available_factors()


def get_pure_financial_factor_categories() -> Dict[str, List[str]]:
    """获取纯财务因子分类"""
    calculator = PureFinancialFactorCalculator()
    return calculator.get_factor_categories()