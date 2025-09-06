#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量因子实现
包含盈利质量、财务稳定性等质量相关因子
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


class ROEQualityFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    ROE质量因子
    衡量ROE的稳定性和持续性
    """
    
    def __init__(self, stability_periods: int = 12):
        """
        Parameters:
        -----------
        stability_periods : int
            计算稳定性的期数，默认12个季度
        """
        super().__init__(name=f'ROE_Quality_{stability_periods}', category='quality')
        self.stability_periods = stability_periods
        self.description = f"ROE质量因子，基于{stability_periods}期稳定性"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ROE质量因子
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含净利润、净资产和季度信息
        
        Returns:
        --------
        pd.Series : ROE质量因子值
        """
        try:
            # 使用实际的字段名
            earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
            equity_col = 'EQY_BELONGTO_PARCOMSH'  # 归属母公司股东权益
            
            # 检查必要字段是否存在
            required_cols = [earnings_col, equity_col, 'd_year', 'd_quarter']
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 计算季度ROE
            earnings = financial_data[earnings_col]
            equity = financial_data[equity_col]
            
            # 计算净资产的期初期末平均值
            equity_avg = equity.groupby(level='StockCodes').rolling(
                window=2, min_periods=1
            ).mean().droplevel(0)
            
            quarterly_roe = self._safe_division(earnings, equity_avg)
            
            # 计算ROE的稳定性（标准差的倒数）
            roe_stability = quarterly_roe.groupby(level='StockCodes').rolling(
                window=self.stability_periods, min_periods=self.stability_periods//2
            ).std().droplevel(0)
            
            # ROE质量 = 1 / ROE波动率（波动率越小，质量越高）
            roe_quality = self._safe_division(1.0, roe_stability)
            
            # 加入ROE水平的考虑（高ROE且稳定的更好）
            roe_mean = quarterly_roe.groupby(level='StockCodes').rolling(
                window=self.stability_periods, min_periods=self.stability_periods//2
            ).mean().droplevel(0)
            
            # 综合质量分数 = 稳定性 * 平均ROE水平
            combined_quality = roe_quality * np.maximum(roe_mean, 0)  # 负ROE给0权重
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                quality_df = combined_quality.to_frame(name='ROE_Quality')
                quality_expanded = self._expand_and_align_data(
                    quality_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                combined_quality = quality_expanded['ROE_Quality']
                
            # 预处理
            combined_quality = self.preprocess(combined_quality)
            
            return combined_quality
            
        except Exception as e:
            logger.error(f"计算ROE质量因子失败: {e}")
            return pd.Series(dtype=float)


class EarningsQualityFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    盈利质量因子
    比较净利润与经营现金流的一致性
    """
    
    def __init__(self):
        super().__init__(name='EarningsQuality', category='quality')
        self.description = "盈利质量 - 经营现金流TTM/净利润TTM"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算盈利质量因子
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含净利润、经营现金流和季度信息
        
        Returns:
        --------
        pd.Series : 盈利质量因子值
        """
        try:
            # 使用实际的字段名
            earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
            cashflow_col = 'NET_CASH_FLOWS_OPER_ACT'  # 经营现金流
            
            # 检查必要字段是否存在
            required_cols = [earnings_col, cashflow_col, 'd_year', 'd_quarter']
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 使用新的FinancialReportProcessor接口计算TTM
            ttm_results = FinancialReportProcessor.calculate_ttm(financial_data)
            
            # 提取TTM序列
            earnings_series = ttm_results[f'{earnings_col}_ttm']
            cashflow_series = ttm_results[f'{cashflow_col}_ttm']
            
            # 对齐数据
            earnings_aligned, cashflow_aligned = earnings_series.align(cashflow_series, join='inner')
            
            # 计算盈利质量：经营现金流/净利润
            earnings_quality = self._safe_division(cashflow_aligned, earnings_aligned)
            
            # 处理异常情况：负利润但正现金流的情况
            negative_earnings = earnings_aligned < 0
            positive_cashflow = cashflow_aligned > 0
            special_case = negative_earnings & positive_cashflow
            
            # 对于负利润但正现金流的情况，给予高质量分数
            earnings_quality[special_case] = 2.0
            
            # 限制极值
            earnings_quality = earnings_quality.clip(lower=-5, upper=5)
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                quality_df = earnings_quality.to_frame(name='EarningsQuality')
                quality_expanded = self._expand_and_align_data(
                    quality_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                earnings_quality = quality_expanded['EarningsQuality']
            
            # 预处理
            earnings_quality = self.preprocess(earnings_quality)
            
            return earnings_quality
            
        except Exception as e:
            logger.error(f"计算盈利质量因子失败: {e}")
            return pd.Series(dtype=float)


class DebtQualityFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    债务质量因子
    衡量公司的财务杠杆健康程度
    """
    
    def __init__(self):
        super().__init__(name='DebtQuality', category='quality')
        self.description = "债务质量 - 综合负债比率和利息保障倍数"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算债务质量因子
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含相关负债和盈利信息
        
        Returns:
        --------
        pd.Series : 债务质量因子值
        """
        try:
            # 使用实际的字段名
            liabilities_col = 'TOT_LIAB'  # 总负债
            assets_col = 'TOT_ASSETS'  # 总资产
            operating_profit_col = 'OPER_PROFIT'  # 营业利润
            financial_expense_col = 'FIN_EXP_IS'  # 财务费用
            
            # 检查资产负债率相关数据
            has_debt_ratio = all(col in financial_data.columns for col in [liabilities_col, assets_col])
            
            # 检查利息保障相关数据
            has_interest_coverage = all(col in financial_data.columns for col in [operating_profit_col, financial_expense_col, 'd_year', 'd_quarter'])
            
            if not (has_debt_ratio or has_interest_coverage):
                raise ValueError("Required data not available for Debt Quality calculation")
            
            quality_components = []
            
            # 1. 资产负债率质量（越低越好）
            if has_debt_ratio:
                debt_ratio = self._safe_division(
                    financial_data[liabilities_col],
                    financial_data[assets_col]
                )
                
                # 债务质量 = 1 - 资产负债率（资产负债率越低，质量越好）
                debt_ratio_quality = 1 - debt_ratio
                quality_components.append(debt_ratio_quality)
            
            # 2. 利息保障倍数质量（越高越好）
            if has_interest_coverage:
                # 使用新的FinancialReportProcessor接口计算TTM
                ttm_results = FinancialReportProcessor.calculate_ttm(financial_data)
                
                # 提取TTM结果
                profit_ttm = ttm_results[f'{operating_profit_col}_ttm']
                expense_ttm = ttm_results[f'{financial_expense_col}_ttm']
                
                # 对齐数据并计算利息保障倍数
                profit_aligned, expense_aligned = profit_ttm.align(expense_ttm, join='inner')
                interest_coverage = self._safe_division(profit_aligned, expense_aligned.abs())
                
                # 转换为质量分数（使用对数变换避免极值）
                interest_quality = np.log1p(np.maximum(interest_coverage, 0))
                quality_components.append(interest_quality)
            
            # 组合质量分数
            if len(quality_components) == 1:
                debt_quality = quality_components[0]
            else:
                # 等权重平均
                debt_quality = sum(quality_components) / len(quality_components)
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                quality_df = debt_quality.to_frame(name='DebtQuality')
                quality_expanded = self._expand_and_align_data(
                    quality_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                debt_quality = quality_expanded['DebtQuality']
            
            # 预处理
            debt_quality = self.preprocess(debt_quality)
            
            return debt_quality
            
        except Exception as e:
            logger.error(f"计算债务质量因子失败: {e}")
            return pd.Series(dtype=float)


class ProfitabilityStabilityFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    盈利稳定性因子
    衡量公司盈利能力的稳定程度
    """
    
    def __init__(self, lookback_periods: int = 20):
        """
        Parameters:
        -----------
        lookback_periods : int
            计算稳定性的期数，默认20个季度（5年）
        """
        super().__init__(name=f'ProfitabilityStability_{lookback_periods}', category='quality')
        self.lookback_periods = lookback_periods
        self.description = f"盈利稳定性因子，基于{lookback_periods}期数据"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算盈利稳定性因子
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含净利润、营业收入等信息
        
        Returns:
        --------
        pd.Series : 盈利稳定性因子值
        """
        try:
            # 使用实际的字段名
            earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
            revenue_col = 'TOT_OPER_REV'  # 营业收入
            
            # 检查必要字段是否存在
            required_cols = [earnings_col, revenue_col]
            missing_cols = [col for col in required_cols if col not in financial_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 计算净利率
            net_margin = self._safe_division(
                financial_data[earnings_col],
                financial_data[revenue_col]
            )
            
            # 计算净利率的稳定性（变异系数的倒数）
            net_margin_mean = net_margin.groupby(level='StockCodes').rolling(
                window=self.lookback_periods, min_periods=self.lookback_periods//2
            ).mean().droplevel(0)
            
            net_margin_std = net_margin.groupby(level='StockCodes').rolling(
                window=self.lookback_periods, min_periods=self.lookback_periods//2
            ).std().droplevel(0)
            
            # 变异系数 = 标准差 / 均值
            coefficient_of_variation = self._safe_division(net_margin_std, net_margin_mean.abs())
            
            # 稳定性 = 1 / (1 + 变异系数)，变异系数越小稳定性越高
            profitability_stability = self._safe_division(1.0, 1 + coefficient_of_variation)
            
            # 考虑盈利能力的水平：只有盈利且稳定的才给高分
            positive_margin = net_margin_mean > 0
            profitability_stability = profitability_stability * positive_margin
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                stability_df = profitability_stability.to_frame(name='ProfitabilityStability')
                stability_expanded = self._expand_and_align_data(
                    stability_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                profitability_stability = stability_expanded['ProfitabilityStability']
            
            # 预处理
            profitability_stability = self.preprocess(profitability_stability)
            
            return profitability_stability
            
        except Exception as e:
            logger.error(f"计算盈利稳定性因子失败: {e}")
            return pd.Series(dtype=float)


class AssetQualityFactor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """
    资产质量因子
    评估公司资产的质量和效率
    """
    
    def __init__(self):
        super().__init__(name='AssetQuality', category='quality')
        self.description = "资产质量 - 资产周转率和有形资产比例"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算资产质量因子
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含相关资产信息
        
        Returns:
        --------
        pd.Series : 资产质量因子值
        """
        try:
            # 检查数据可用性
            has_asset_turnover = self.validate_data_requirements(
                financial_data, ['revenue', 'total_assets', 'quarter']
            )
            has_tangible_ratio = self.validate_data_requirements(
                financial_data, ['total_assets', 'intangible_assets']
            )
            
            if not (has_asset_turnover or has_tangible_ratio):
                raise ValueError("Required data not available for Asset Quality calculation")
            
            quality_components = []
            
            # 1. 资产周转率（效率指标）
            if has_asset_turnover:
                extracted_data = self.extract_required_data(
                    financial_data, required_columns=['revenue', 'total_assets', 'quarter']
                )
                
                # 计算营业收入TTM
                revenue_data = extracted_data[['revenue', 'quarter']].copy()
                revenue_data = revenue_data.rename(columns={
                    'revenue': 'TOT_OPER_REV', 
                    'quarter': 'd_quarter'
                })
                revenue_ttm = FinancialReportProcessor.calculate_ttm(revenue_data)
                
                # 计算资产平均值
                assets = extracted_data[['total_assets']].copy()
                assets = assets.rename(columns={'total_assets': 'TOT_ASSETS'})
                assets_avg = FinancialReportProcessor.calculate_avg(assets)
                
                # 计算资产周转率
                revenue_series = revenue_ttm.iloc[:, 0] if revenue_ttm.shape[1] > 0 else pd.Series(dtype=float)
                assets_series = assets_avg.iloc[:, 0] if assets_avg.shape[1] > 0 else pd.Series(dtype=float)
                
                asset_turnover = self._safe_division(revenue_series, assets_series)
                quality_components.append(asset_turnover)
            
            # 2. 有形资产比例（资产"真实性"指标）
            if has_tangible_ratio:
                extracted_data = self.extract_required_data(
                    financial_data, required_columns=['total_assets', 'intangible_assets']
                )
                
                # 计算有形资产比例
                tangible_assets = extracted_data['total_assets'] - extracted_data['intangible_assets']
                tangible_ratio = self._safe_division(tangible_assets, extracted_data['total_assets'])
                quality_components.append(tangible_ratio)
            
            # 组合资产质量分数
            if len(quality_components) == 1:
                asset_quality = quality_components[0]
            else:
                # 等权重平均
                asset_quality = sum(quality_components) / len(quality_components)
            
            # 扩展到日频（如果需要）
            if 'release_dates' in kwargs and 'trading_dates' in kwargs:
                quality_df = asset_quality.to_frame(name='AssetQuality')
                quality_expanded = self._expand_and_align_data(
                    quality_df,
                    release_dates=kwargs['release_dates'],
                    trading_dates=kwargs['trading_dates']
                )
                asset_quality = quality_expanded['AssetQuality']
            
            # 预处理
            asset_quality = self.preprocess(asset_quality)
            
            return asset_quality
            
        except Exception as e:
            logger.error(f"计算资产质量因子失败: {e}")
            return pd.Series(dtype=float)