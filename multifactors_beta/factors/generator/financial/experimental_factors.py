#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验性因子模块 - 用于测试和验证新因子想法

设计原则：
==========
1. 继承PureFinancialFactorCalculator，复用所有基础功能
2. 实验性因子不注册到正式系统，保持灵活性
3. 使用EXPERIMENTAL_前缀标识实验性因子
4. 提供快速验证和测试工具
5. 经过验证的因子可以轻松迁移到正式模块

工作流程：
==========
1. 在这里实现新因子想法 -> 2. 快速测试验证 -> 3. 通过验证后迁移到正式模块

使用示例：
==========
from factors.generator.financial.experimental_factors import ExperimentalFactorCalculator

calculator = ExperimentalFactorCalculator()
# 直接调用测试
new_factor = calculator.calculate_EXPERIMENTAL_YourIdea_ttm(financial_data)
# 快速验证
result = calculator.quick_validate_factor(new_factor, 'YourIdea')
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any, Tuple
import logging
from datetime import datetime
import warnings
from scipy import stats
from collections import defaultdict

from .pure_financial_factors import PureFinancialFactorCalculator
from ...base.time_series_processor import TimeSeriesProcessor
from ...config.field_mapper import get_field_mapper

logger = logging.getLogger(__name__)


class ExperimentalFactorCalculator(PureFinancialFactorCalculator):
    """
    实验性因子计算器
    
    继承PureFinancialFactorCalculator的所有功能，专门用于测试新因子想法
    """
    
    def __init__(self):
        super().__init__()
        self.description = "Experimental Factor Calculator - For testing new factor ideas"
        
        # 实验性因子不会注册到正式系统
        self.experimental_factors = []
        self.validation_results = {}
        
        # 加载字段映射器
        self.field_mapper = get_field_mapper()
        
        logger.info("🧪 实验性因子计算器已初始化")
    
    # =====================================================
    # 实验性因子模板和示例
    # =====================================================
    
    def calculate_EXPERIMENTAL_Template_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        实验性因子模板 - 复制这个模板开始新因子开发
        
        计算公式：[在这里描述你的因子计算公式]
        经济含义：[解释这个因子的经济学意义]
        假设验证：[说明你想验证的假设]
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        pd.Series
            因子值
        """
        # 步骤1：验证数据需求
        required_cols = ['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'TAX', 'd_quarter']  # 修改为你需要的列
        if not self.validate_data_requirements(financial_data, required_cols):
            raise ValueError(f"Required data not available for Template calculation")
        
        # 步骤2：提取数据
        extracted_data = self.extract_required_data(financial_data, required_cols)
         
        # 步骤3：实现你的计算逻辑
        # TODO: 在这里实现具体的因子计算
        result = pd.Series(index=financial_data.index, dtype=float)
        
        # 步骤4：返回结果
        logger.info("✨ 实验性因子Template计算完成")
        return result
    
    def calculate_EXPERIMENTAL_ProfitGrowthQuality_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        实验性因子：盈利增长质量 - 示例实现
        
        计算公式：(TTM净利润增长率 × 经营现金流/净利润) / ROE波动率
        经济含义：衡量企业盈利增长的质量和可持续性
        假设验证：高质量的盈利增长应该伴随现金流增长且ROE稳定
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            财务数据，需包含净利润、经营现金流等字段
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        pd.Series
            盈利增长质量因子值
        """
        try:
            # 验证数据需求
            required_cols = ['earnings', 'operating_cash_flow', 'equity', 'quarter']
            if not self.validate_data_requirements(financial_data, required_cols):
                raise ValueError("Required data not available for ProfitGrowthQuality calculation")
            
            extracted_data = self.extract_required_data(financial_data, required_cols)
            
            # 1. 计算TTM净利润
            earnings_data = extracted_data[['earnings', 'quarter']].copy()
            earnings_data = earnings_data.rename(columns={
                'earnings': 'DEDUCTEDPROFIT', 
                'quarter': 'd_quarter'
            })
            earnings_ttm = TimeSeriesProcessor.calculate_ttm(earnings_data)
            
            # 2. 计算TTM经营现金流
            cf_data = extracted_data[['operating_cash_flow', 'quarter']].copy()
            cf_data = cf_data.rename(columns={
                'operating_cash_flow': 'NETCASH_OPER', 
                'quarter': 'd_quarter'
            })
            cf_ttm = TimeSeriesProcessor.calculate_ttm(cf_data)
            
            # 3. 计算ROE
            roe_data = extracted_data[['earnings', 'equity', 'quarter']].copy()
            roe_data = roe_data.rename(columns={
                'earnings': 'DEDUCTEDPROFIT',
                'equity': 'EQY_BELONGTO_PARCOMSH',
                'quarter': 'd_quarter'
            })
            roe_ttm_data = TimeSeriesProcessor.calculate_ttm(roe_data[['DEDUCTEDPROFIT', 'd_quarter']])
            equity_avg = TimeSeriesProcessor.calculate_avg(roe_data[['EQY_BELONGTO_PARCOMSH']])
            
            # 4. 计算各个组件
            earnings_series = earnings_ttm.iloc[:, 0] if earnings_ttm.shape[1] > 0 else pd.Series(dtype=float)
            cf_series = cf_ttm.iloc[:, 0] if cf_ttm.shape[1] > 0 else pd.Series(dtype=float)
            roe_ttm_series = roe_ttm_data.iloc[:, 0] if roe_ttm_data.shape[1] > 0 else pd.Series(dtype=float)
            equity_series = equity_avg.iloc[:, 0] if equity_avg.shape[1] > 0 else pd.Series(dtype=float)
            
            # 对齐数据
            earnings_aligned, cf_aligned = earnings_series.align(cf_series, join='inner')
            roe_values = self._safe_division(roe_ttm_series, equity_series)
            
            # 5. 计算净利润同比增长率
            earnings_yoy = TimeSeriesProcessor.calculate_yoy(earnings_ttm)
            earnings_growth = earnings_yoy.iloc[:, 0] if earnings_yoy.shape[1] > 0 else pd.Series(dtype=float)
            
            # 6. 计算现金流质量 (经营现金流/净利润)
            cf_quality = self._safe_division(cf_aligned, earnings_aligned)
            
            # 7. 计算ROE波动率 (过去8个季度的标准差)
            roe_volatility = roe_values.groupby(level='StockCodes').rolling(window=8, min_periods=4).std()
            if isinstance(roe_volatility.index, pd.MultiIndex):
                roe_volatility.index = roe_volatility.index.droplevel(0)
            
            # 8. 综合计算盈利增长质量
            # (净利润增长率 × 现金流质量) / ROE波动率
            growth_quality = self._safe_division(
                earnings_growth * cf_quality,
                roe_volatility
            )
            
            # 清理异常值
            growth_quality = growth_quality.replace([np.inf, -np.inf], np.nan)
            
            logger.info("✨ 实验性因子ProfitGrowthQuality计算完成")
            return growth_quality
            
        except Exception as e:
            logger.error(f"ProfitGrowthQuality计算失败: {e}")
            return pd.Series(index=financial_data.index, dtype=float)
    
    def calculate_EXPERIMENTAL_DebtServiceAbility_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        实验性因子：债务偿付能力 - 示例实现
        
        计算公式：(经营现金流TTM + 货币资金) / (短期借款 + 一年内到期的长期负债 + 利息费用TTM)
        经济含义：衡量企业短期偿债能力，考虑现金流和现有资金
        假设验证：更好的债务偿付能力应该降低违约风险，提升估值
        """
        try:
            required_cols = ['operating_cash_flow', 'cash_equivalents', 'short_term_debt', 'financial_expense', 'quarter']
            if not self.validate_data_requirements(financial_data, required_cols):
                logger.warning("Data not sufficient for DebtServiceAbility calculation")
                return pd.Series(index=financial_data.index, dtype=float)
            
            extracted_data = self.extract_required_data(financial_data, required_cols)
            
            # 经营现金流TTM
            cf_data = extracted_data[['operating_cash_flow', 'quarter']].copy()
            cf_data = cf_data.rename(columns={'operating_cash_flow': 'NETCASH_OPER', 'quarter': 'd_quarter'})
            cf_ttm = TimeSeriesProcessor.calculate_ttm(cf_data)
            
            # 利息费用TTM
            interest_data = extracted_data[['financial_expense', 'quarter']].copy()
            interest_data = interest_data.rename(columns={'financial_expense': 'FIN_EXP_IS', 'quarter': 'd_quarter'})
            interest_ttm = TimeSeriesProcessor.calculate_ttm(interest_data)
            
            # 获取数值
            cf_values = cf_ttm.iloc[:, 0] if cf_ttm.shape[1] > 0 else pd.Series(dtype=float)
            interest_values = interest_ttm.iloc[:, 0] if interest_ttm.shape[1] > 0 else pd.Series(dtype=float)
            cash_values = extracted_data['cash_equivalents']
            debt_values = extracted_data['short_term_debt']
            
            # 计算偿债能力 = (现金流 + 现金) / (短期债务 + 利息)
            numerator = cf_values + cash_values
            denominator = debt_values + interest_values.abs()  # 利息费用取绝对值
            
            debt_service_ability = self._safe_division(numerator, denominator)
            
            logger.info("✨ 实验性因子DebtServiceAbility计算完成")
            return debt_service_ability
            
        except Exception as e:
            logger.error(f"DebtServiceAbility计算失败: {e}")
            return pd.Series(index=financial_data.index, dtype=float)
    
    # =====================================================
    # 单因子检验模块
    # =====================================================
    
    def single_factor_test(self,
                          factor_data: pd.Series,
                          return_data: pd.Series,
                          factor_name: str,
                          periods: List[int] = [1, 5, 10, 20],
                          quantiles: int = 5,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        完整的单因子检验
        
        Parameters:
        -----------
        factor_data : 因子数据 (MultiIndex: TradingDates, StockCodes)
        return_data : 收益率数据 (MultiIndex: TradingDates, StockCodes)
        factor_name : 因子名称
        periods : 测试的持有期列表
        quantiles : 分组数量
        save_results : 是否保存结果
        
        Returns:
        --------
        完整检验结果字典
        """
        logger.info(f"🔬 开始单因子检验: {factor_name}")
        
        test_results = {
            'factor_name': factor_name,
            'test_time': datetime.now(),
            'ic_analysis': {},
            'group_analysis': {},
            'monotonicity_test': {},
            'significance_test': {},
            'decay_analysis': {},
            'summary': {}
        }
        
        try:
            # 1. IC分析
            logger.info("   📊 执行IC分析...")
            test_results['ic_analysis'] = self._calculate_ic_analysis(
                factor_data, return_data, periods
            )
            
            # 2. 分组分析
            logger.info("   📈 执行分组分析...")
            test_results['group_analysis'] = self._calculate_group_analysis(
                factor_data, return_data, periods, quantiles
            )
            
            # 3. 单调性检验
            logger.info("   📉 执行单调性检验...")
            test_results['monotonicity_test'] = self._test_monotonicity(
                test_results['group_analysis']
            )
            
            # 4. 统计显著性检验
            logger.info("   🎯 执行显著性检验...")
            test_results['significance_test'] = self._test_significance(
                test_results['ic_analysis'], test_results['group_analysis']
            )
            
            # 5. 因子衰减分析
            logger.info("   ⏰ 执行衰减分析...")
            test_results['decay_analysis'] = self._analyze_factor_decay(
                factor_data, return_data, periods=[1, 5, 10, 20, 60]
            )
            
            # 6. 生成综合评价
            logger.info("   📋 生成综合评价...")
            test_results['summary'] = self._generate_test_summary(test_results)
            
            # 保存结果
            if save_results:
                self.validation_results[f"{factor_name}_single_test"] = test_results
            
            # 打印报告
            self._print_single_factor_report(test_results)
            
        except Exception as e:
            logger.error(f"单因子检验失败: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def _calculate_ic_analysis(self, 
                              factor_data: pd.Series,
                              return_data: pd.Series,
                              periods: List[int]) -> Dict[str, Any]:
        """计算IC分析"""
        ic_results = {}
        
        # 确保数据对齐
        factor_aligned, return_aligned = factor_data.align(return_data, join='inner')
        
        if len(factor_aligned) == 0:
            return {'error': 'No aligned data for IC calculation'}
        
        for period in periods:
            # 计算前瞻收益
            future_returns = self._calculate_forward_returns(return_aligned, period)
            
            # 对齐因子值和前瞻收益
            factor_for_ic, returns_for_ic = factor_aligned.align(future_returns, join='inner')
            
            if len(factor_for_ic) == 0:
                ic_results[f'period_{period}'] = {'error': 'No data for this period'}
                continue
            
            # 按日期分组计算IC
            dates = factor_for_ic.index.get_level_values(0).unique()
            daily_ics = []
            
            for date in dates:
                date_factor = factor_for_ic[factor_for_ic.index.get_level_values(0) == date]
                date_return = returns_for_ic[returns_for_ic.index.get_level_values(0) == date]
                
                if len(date_factor) > 10:  # 至少10只股票
                    # 计算Spearman相关系数（Rank IC）
                    ic_value = stats.spearmanr(date_factor, date_return)[0]
                    if not np.isnan(ic_value):
                        daily_ics.append(ic_value)
            
            if len(daily_ics) > 0:
                daily_ics = np.array(daily_ics)
                
                ic_results[f'period_{period}'] = {
                    'mean_ic': np.mean(daily_ics),
                    'std_ic': np.std(daily_ics),
                    'ic_ir': np.mean(daily_ics) / np.std(daily_ics) if np.std(daily_ics) > 0 else 0,
                    'win_rate': np.sum(daily_ics > 0) / len(daily_ics),
                    'daily_ics': daily_ics.tolist(),
                    'ic_t_stat': stats.ttest_1samp(daily_ics, 0)[0],
                    'ic_p_value': stats.ttest_1samp(daily_ics, 0)[1]
                }
            else:
                ic_results[f'period_{period}'] = {'error': 'Insufficient data for IC calculation'}
        
        return ic_results
    
    def _calculate_forward_returns(self, 
                                  return_data: pd.Series, 
                                  period: int) -> pd.Series:
        """
        计算前瞻收益
        
        Parameters:
        -----------
        return_data : pd.Series
            日收益率数据（通常是对数收益率）
        period : int
            持有期天数
            
        Returns:
        --------
        pd.Series
            前瞻收益率
            - period=1: 直接使用下一期收益率
            - period>1: 累积对数收益率（适用于日收益率是对数收益率的情况）
        """
        def calc_forward_return(stock_returns):
            if period == 1:
                # 单期：直接使用下一期收益率
                return stock_returns.shift(-1)
            else:
                # 多期：累积对数收益率
                # 使用rolling窗口向前计算累积收益
                forward_cumulative = stock_returns.rolling(
                    window=period, 
                    min_periods=period
                ).sum().shift(-period)
                return forward_cumulative
        
        forward_returns = return_data.groupby(level='StockCodes').apply(calc_forward_return)
        
        # 重新整理索引
        if isinstance(forward_returns.index, pd.MultiIndex):
            forward_returns.index = forward_returns.index.droplevel(0)
        
        return forward_returns
    
    def _calculate_group_analysis(self,
                                 factor_data: pd.Series,
                                 return_data: pd.Series,
                                 periods: List[int],
                                 quantiles: int) -> Dict[str, Any]:
        """计算分组分析"""
        group_results = {}
        
        # 确保数据对齐
        factor_aligned, return_aligned = factor_data.align(return_data, join='inner')
        
        for period in periods:
            # 计算前瞻收益
            future_returns = self._calculate_forward_returns(return_aligned, period)
            
            # 对齐数据
            factor_for_group, returns_for_group = factor_aligned.align(future_returns, join='inner')
            
            if len(factor_for_group) == 0:
                group_results[f'period_{period}'] = {'error': 'No data for grouping'}
                continue
            
            # 按日期分组分析
            dates = factor_for_group.index.get_level_values(0).unique()
            daily_group_returns = []
            
            for date in dates:
                date_factor = factor_for_group[factor_for_group.index.get_level_values(0) == date]
                date_return = returns_for_group[returns_for_group.index.get_level_values(0) == date]
                
                if len(date_factor) >= quantiles * 5:  # 确保每组至少5只股票
                    # 按因子值分组
                    factor_ranks = date_factor.rank(pct=True)
                    
                    group_returns = []
                    for q in range(quantiles):
                        q_min = q / quantiles
                        q_max = (q + 1) / quantiles
                        
                        group_mask = (factor_ranks >= q_min) & (factor_ranks < q_max)
                        if q == quantiles - 1:  # 最后一组包含等于上界的值
                            group_mask = (factor_ranks >= q_min) & (factor_ranks <= q_max)
                        
                        group_stocks = date_factor[group_mask]
                        if len(group_stocks) > 0:
                            group_return = date_return[group_stocks.index].mean()
                            group_returns.append(group_return)
                        else:
                            group_returns.append(np.nan)
                    
                    if not all(np.isnan(group_returns)):
                        daily_group_returns.append(group_returns)
            
            if len(daily_group_returns) > 0:
                # 计算各组平均收益
                daily_group_returns = np.array(daily_group_returns)
                
                group_stats = {}
                for q in range(quantiles):
                    group_q_returns = daily_group_returns[:, q]
                    valid_returns = group_q_returns[~np.isnan(group_q_returns)]
                    
                    if len(valid_returns) > 0:
                        group_stats[f'group_{q+1}'] = {
                            'mean_return': np.mean(valid_returns),
                            'std_return': np.std(valid_returns),
                            'sharpe_ratio': np.mean(valid_returns) / np.std(valid_returns) if np.std(valid_returns) > 0 else 0,
                            'win_rate': np.sum(valid_returns > 0) / len(valid_returns),
                            'daily_returns': valid_returns.tolist()
                        }
                
                # 计算多空收益 (最高组 - 最低组)
                if f'group_{quantiles}' in group_stats and 'group_1' in group_stats:
                    high_returns = np.array(group_stats[f'group_{quantiles}']['daily_returns'])
                    low_returns = np.array(group_stats['group_1']['daily_returns'])
                    
                    # 确保长度一致
                    min_len = min(len(high_returns), len(low_returns))
                    if min_len > 0:
                        long_short = high_returns[:min_len] - low_returns[:min_len]
                        
                        group_stats['long_short'] = {
                            'mean_return': np.mean(long_short),
                            'std_return': np.std(long_short),
                            'sharpe_ratio': np.mean(long_short) / np.std(long_short) if np.std(long_short) > 0 else 0,
                            'win_rate': np.sum(long_short > 0) / len(long_short),
                            't_stat': stats.ttest_1samp(long_short, 0)[0],
                            'p_value': stats.ttest_1samp(long_short, 0)[1]
                        }
                
                group_results[f'period_{period}'] = group_stats
            else:
                group_results[f'period_{period}'] = {'error': 'No valid group data'}
        
        return group_results
    
    def _test_monotonicity(self, group_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """检验因子单调性"""
        monotonicity_results = {}
        
        for period, period_data in group_analysis.items():
            if 'error' in period_data:
                monotonicity_results[period] = {'error': period_data['error']}
                continue
            
            # 提取各组收益率
            group_returns = []
            for key in sorted(period_data.keys()):
                if key.startswith('group_') and not key == 'long_short':
                    group_returns.append(period_data[key]['mean_return'])
            
            if len(group_returns) >= 3:
                # 计算单调性指标
                # 1. Spearman rank correlation
                group_ranks = list(range(1, len(group_returns) + 1))
                monotonicity_corr = stats.spearmanr(group_ranks, group_returns)[0]
                
                # 2. 单调递增的组数
                increasing_pairs = 0
                total_pairs = 0
                for i in range(len(group_returns)):
                    for j in range(i + 1, len(group_returns)):
                        total_pairs += 1
                        if group_returns[j] > group_returns[i]:
                            increasing_pairs += 1
                
                monotonicity_rate = increasing_pairs / total_pairs if total_pairs > 0 else 0
                
                monotonicity_results[period] = {
                    'monotonicity_corr': monotonicity_corr,
                    'monotonicity_rate': monotonicity_rate,
                    'group_returns': group_returns,
                    'is_monotonic': monotonicity_rate > 0.7  # 70%以上的组对是递增的
                }
            else:
                monotonicity_results[period] = {'error': 'Insufficient groups for monotonicity test'}
        
        return monotonicity_results
    
    def _test_significance(self, 
                          ic_analysis: Dict[str, Any],
                          group_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """统计显著性检验"""
        significance_results = {}
        
        for period in ic_analysis.keys():
            if period in group_analysis:
                sig_results = {}
                
                # IC显著性
                ic_data = ic_analysis[period]
                if 'ic_p_value' in ic_data:
                    sig_results['ic_significant'] = ic_data['ic_p_value'] < 0.05
                    sig_results['ic_p_value'] = ic_data['ic_p_value']
                    sig_results['ic_t_stat'] = ic_data['ic_t_stat']
                
                # 多空收益显著性
                group_data = group_analysis[period]
                if 'long_short' in group_data:
                    ls_data = group_data['long_short']
                    sig_results['long_short_significant'] = ls_data['p_value'] < 0.05
                    sig_results['long_short_p_value'] = ls_data['p_value']
                    sig_results['long_short_t_stat'] = ls_data['t_stat']
                
                significance_results[period] = sig_results
        
        return significance_results
    
    def _analyze_factor_decay(self,
                             factor_data: pd.Series,
                             return_data: pd.Series,
                             periods: List[int]) -> Dict[str, Any]:
        """因子衰减分析"""
        decay_results = {}
        
        try:
            # 计算不同期间的IC
            ic_by_period = {}
            for period in periods:
                ic_result = self._calculate_ic_analysis(factor_data, return_data, [period])
                if f'period_{period}' in ic_result and 'mean_ic' in ic_result[f'period_{period}']:
                    ic_by_period[period] = ic_result[f'period_{period}']['mean_ic']
            
            if len(ic_by_period) > 0:
                # 计算衰减率
                periods_sorted = sorted(ic_by_period.keys())
                ic_values = [ic_by_period[p] for p in periods_sorted]
                
                # 计算相对于第一期的衰减
                if len(ic_values) > 1 and abs(ic_values[0]) > 0.001:
                    decay_rates = [(ic_values[i] / ic_values[0] - 1) for i in range(1, len(ic_values))]
                    
                    decay_results = {
                        'periods': periods_sorted,
                        'ic_values': ic_values,
                        'decay_rates': decay_rates,
                        'half_life_period': None
                    }
                    
                    # 估算半衰期
                    half_ic = ic_values[0] * 0.5
                    for i, ic in enumerate(ic_values[1:], 1):
                        if abs(ic) <= abs(half_ic):
                            decay_results['half_life_period'] = periods_sorted[i]
                            break
                
        except Exception as e:
            decay_results = {'error': str(e)}
        
        return decay_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合评价"""
        summary = {
            'overall_score': 0,
            'grade': 'F',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        try:
            score = 0
            total_checks = 0
            
            # 1. IC分析评分 (30分)
            ic_scores = []
            for period, ic_data in test_results['ic_analysis'].items():
                if 'mean_ic' in ic_data:
                    abs_ic = abs(ic_data['mean_ic'])
                    ic_ir = ic_data.get('ic_ir', 0)
                    
                    if abs_ic > 0.05:
                        ic_scores.append(15)
                    elif abs_ic > 0.03:
                        ic_scores.append(10)
                    elif abs_ic > 0.01:
                        ic_scores.append(5)
                    else:
                        ic_scores.append(0)
                    
                    if abs(ic_ir) > 1.0:
                        ic_scores[-1] += 15
                    elif abs(ic_ir) > 0.5:
                        ic_scores[-1] += 10
                    else:
                        ic_scores[-1] += 5
            
            if ic_scores:
                score += max(ic_scores)
                total_checks += 30
                if max(ic_scores) > 20:
                    summary['strengths'].append('IC质量优秀')
                elif max(ic_scores) < 10:
                    summary['weaknesses'].append('IC较弱')
            
            # 2. 分组分析评分 (25分)
            group_scores = []
            for period, group_data in test_results['group_analysis'].items():
                if 'long_short' in group_data:
                    ls_return = group_data['long_short']['mean_return']
                    ls_sharpe = group_data['long_short']['sharpe_ratio']
                    
                    if abs(ls_return) > 0.01:
                        group_scores.append(15)
                    elif abs(ls_return) > 0.005:
                        group_scores.append(10)
                    else:
                        group_scores.append(5)
                    
                    if abs(ls_sharpe) > 1.0:
                        group_scores[-1] += 10
                    elif abs(ls_sharpe) > 0.5:
                        group_scores[-1] += 5
            
            if group_scores:
                score += max(group_scores)
                total_checks += 25
                if max(group_scores) > 20:
                    summary['strengths'].append('分组效果显著')
            
            # 3. 单调性评分 (20分)
            mono_scores = []
            for period, mono_data in test_results['monotonicity_test'].items():
                if 'monotonicity_rate' in mono_data:
                    mono_rate = mono_data['monotonicity_rate']
                    if mono_rate > 0.8:
                        mono_scores.append(20)
                    elif mono_rate > 0.6:
                        mono_scores.append(15)
                    elif mono_rate > 0.4:
                        mono_scores.append(10)
                    else:
                        mono_scores.append(5)
            
            if mono_scores:
                score += max(mono_scores)
                total_checks += 20
                if max(mono_scores) > 15:
                    summary['strengths'].append('单调性良好')
                else:
                    summary['weaknesses'].append('单调性较差')
            
            # 4. 显著性评分 (15分)
            sig_count = 0
            sig_total = 0
            for period, sig_data in test_results['significance_test'].items():
                if 'ic_significant' in sig_data:
                    sig_total += 1
                    if sig_data['ic_significant']:
                        sig_count += 1
                if 'long_short_significant' in sig_data:
                    sig_total += 1
                    if sig_data['long_short_significant']:
                        sig_count += 1
            
            if sig_total > 0:
                sig_score = (sig_count / sig_total) * 15
                score += sig_score
                total_checks += 15
                if sig_score > 10:
                    summary['strengths'].append('统计显著性强')
            
            # 5. 衰减分析评分 (10分)
            if 'half_life_period' in test_results['decay_analysis']:
                half_life = test_results['decay_analysis']['half_life_period']
                if half_life and half_life > 10:
                    score += 10
                    summary['strengths'].append('因子持续性好')
                elif half_life and half_life > 5:
                    score += 7
                else:
                    score += 3
                    summary['weaknesses'].append('因子衰减较快')
                total_checks += 10
            
            # 计算最终评分
            if total_checks > 0:
                summary['overall_score'] = (score / total_checks) * 100
            
            # 评级
            if summary['overall_score'] >= 80:
                summary['grade'] = 'A'
                summary['recommendations'].append('🎉 优秀因子，强烈推荐进入生产环境')
            elif summary['overall_score'] >= 70:
                summary['grade'] = 'B'
                summary['recommendations'].append('🌟 良好因子，建议进一步优化后使用')
            elif summary['overall_score'] >= 60:
                summary['grade'] = 'C'
                summary['recommendations'].append('⚡ 一般因子，需要显著改进')
            elif summary['overall_score'] >= 40:
                summary['grade'] = 'D'
                summary['recommendations'].append('⚠️ 较差因子，建议重新设计')
            else:
                summary['grade'] = 'F'
                summary['recommendations'].append('❌ 无效因子，不建议使用')
        
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _print_single_factor_report(self, test_results: Dict[str, Any]):
        """打印单因子检验报告"""
        print("\n" + "=" * 80)
        print(f"🔬 单因子检验报告: {test_results['factor_name']}")
        print("=" * 80)
        
        # IC分析结果
        print(f"\n📊 IC分析:")
        for period, ic_data in test_results['ic_analysis'].items():
            if 'mean_ic' in ic_data:
                period_num = period.split('_')[1]
                print(f"   {period_num}期 - IC: {ic_data['mean_ic']:.4f}, "
                      f"IR: {ic_data['ic_ir']:.4f}, "
                      f"胜率: {ic_data['win_rate']:.2%}, "
                      f"t值: {ic_data['ic_t_stat']:.2f}")
        
        # 分组分析结果
        print(f"\n📈 分组分析:")
        for period, group_data in test_results['group_analysis'].items():
            if 'long_short' in group_data:
                period_num = period.split('_')[1]
                ls_data = group_data['long_short']
                print(f"   {period_num}期多空 - 收益: {ls_data['mean_return']:.4f}, "
                      f"夏普: {ls_data['sharpe_ratio']:.4f}, "
                      f"胜率: {ls_data['win_rate']:.2%}, "
                      f"t值: {ls_data['t_stat']:.2f}")
        
        # 单调性检验
        print(f"\n📉 单调性检验:")
        for period, mono_data in test_results['monotonicity_test'].items():
            if 'monotonicity_rate' in mono_data:
                period_num = period.split('_')[1]
                print(f"   {period_num}期 - 单调性: {mono_data['monotonicity_rate']:.2%}, "
                      f"相关性: {mono_data['monotonicity_corr']:.4f}")
        
        # 综合评价
        summary = test_results['summary']
        print(f"\n⭐ 综合评价:")
        print(f"   评分: {summary['overall_score']:.1f}/100 (等级: {summary['grade']})")
        
        if summary['strengths']:
            print(f"   优势: {', '.join(summary['strengths'])}")
        if summary['weaknesses']:
            print(f"   劣势: {', '.join(summary['weaknesses'])}")
        if summary['recommendations']:
            print(f"   建议: {summary['recommendations'][0]}")
        
        print("=" * 80)
    
    # =====================================================
    # 快速验证和测试工具
    # =====================================================
    
    def quick_validate_factor(self, 
                             factor_data: pd.Series,
                             factor_name: str,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        快速验证实验性因子的基本特征
        
        Parameters:
        -----------
        factor_data : 因子数据
        factor_name : 因子名称
        save_results : 是否保存验证结果
        
        Returns:
        --------
        验证结果字典
        """
        logger.info(f"🔍 快速验证实验性因子: {factor_name}")
        
        validation = {
            'factor_name': factor_name,
            'validation_time': datetime.now(),
            'basic_stats': {},
            'data_quality': {},
            'recommendations': []
        }
        
        try:
            # 1. 基础统计
            validation['basic_stats'] = {
                'count': len(factor_data),
                'valid_count': factor_data.count(),
                'missing_rate': (len(factor_data) - factor_data.count()) / len(factor_data),
                'mean': factor_data.mean(),
                'std': factor_data.std(),
                'min': factor_data.min(),
                'max': factor_data.max(),
                'q25': factor_data.quantile(0.25),
                'median': factor_data.median(),
                'q75': factor_data.quantile(0.75),
                'skewness': factor_data.skew(),
                'kurtosis': factor_data.kurtosis()
            }
            
            # 2. 数据质量检查
            stats = validation['basic_stats']
            
            # 检查缺失率
            if stats['missing_rate'] > 0.5:
                validation['data_quality']['missing_rate'] = 'HIGH'
                validation['recommendations'].append('⚠️  缺失率过高，检查数据来源')
            elif stats['missing_rate'] > 0.2:
                validation['data_quality']['missing_rate'] = 'MEDIUM'
                validation['recommendations'].append('⚡ 缺失率中等，考虑数据插补')
            else:
                validation['data_quality']['missing_rate'] = 'LOW'
            
            # 检查极值
            if abs(stats['skewness']) > 5:
                validation['data_quality']['skewness'] = 'HIGH'
                validation['recommendations'].append('📊 偏度过大，考虑去极值处理')
            
            if abs(stats['kurtosis']) > 10:
                validation['data_quality']['kurtosis'] = 'HIGH'
                validation['recommendations'].append('📈 峰度过大，存在极端值')
            
            # 检查数值范围
            if np.isinf(stats['max']) or np.isinf(stats['min']):
                validation['data_quality']['infinite_values'] = True
                validation['recommendations'].append('🚫 存在无穷大值，检查计算逻辑')
            
            # 3. 分布特征
            valid_data = factor_data.dropna()
            if len(valid_data) > 100:
                # 简单的正态性检查
                normal_test_stat = abs(stats['skewness']) + abs(stats['kurtosis'] - 3)
                if normal_test_stat < 2:
                    validation['data_quality']['distribution'] = 'NORMAL_LIKE'
                    validation['recommendations'].append('✅ 分布接近正态')
                else:
                    validation['data_quality']['distribution'] = 'NON_NORMAL'
                    validation['recommendations'].append('📊 非正态分布，可能需要变换')
            
            # 4. 时间稳定性（如果有时间索引）
            if isinstance(factor_data.index, pd.MultiIndex):
                dates = factor_data.index.get_level_values(0).unique()
                if len(dates) > 4:
                    # 按时间计算统计量的变化
                    monthly_stats = []
                    for date in dates[-12:]:  # 最近12期
                        date_data = factor_data[factor_data.index.get_level_values(0) == date]
                        monthly_stats.append(date_data.mean())
                    
                    if len(monthly_stats) > 1:
                        stability_cv = np.std(monthly_stats) / np.mean(monthly_stats) if np.mean(monthly_stats) != 0 else np.inf
                        
                        if stability_cv < 0.1:
                            validation['data_quality']['time_stability'] = 'STABLE'
                            validation['recommendations'].append('⭐ 时间稳定性良好')
                        elif stability_cv < 0.3:
                            validation['data_quality']['time_stability'] = 'MODERATE'
                            validation['recommendations'].append('⚡ 时间稳定性中等')
                        else:
                            validation['data_quality']['time_stability'] = 'UNSTABLE'
                            validation['recommendations'].append('⚠️  时间不稳定，检查计算逻辑')
            
            # 5. 综合评分
            score = 100
            if validation['data_quality'].get('missing_rate') == 'HIGH':
                score -= 30
            elif validation['data_quality'].get('missing_rate') == 'MEDIUM':
                score -= 15
            
            if validation['data_quality'].get('infinite_values'):
                score -= 25
            
            if validation['data_quality'].get('skewness') == 'HIGH':
                score -= 10
            
            if validation['data_quality'].get('time_stability') == 'UNSTABLE':
                score -= 20
            elif validation['data_quality'].get('time_stability') == 'STABLE':
                score += 5
            
            validation['overall_score'] = max(0, score)
            
            # 6. 最终建议
            if score >= 80:
                validation['recommendation'] = '🎉 质量良好，建议深度测试'
            elif score >= 60:
                validation['recommendation'] = '⚡ 质量中等，需要优化后测试'
            elif score >= 40:
                validation['recommendation'] = '⚠️  存在问题，需要重大修改'
            else:
                validation['recommendation'] = '❌ 质量较差，建议重新设计'
            
            # 保存结果
            if save_results:
                self.validation_results[factor_name] = validation
            
            # 输出报告
            self._print_validation_report(validation)
            
        except Exception as e:
            logger.error(f"因子验证失败: {e}")
            validation['error'] = str(e)
        
        return validation
    
    def _print_validation_report(self, validation: Dict[str, Any]):
        """打印验证报告"""
        print("=" * 60)
        print(f"🧪 实验性因子验证报告: {validation['factor_name']}")
        print("=" * 60)
        
        stats = validation['basic_stats']
        print(f"📊 基础统计:")
        print(f"   数据点数: {stats['count']:,}")
        print(f"   有效数据: {stats['valid_count']:,} ({(1-validation['basic_stats']['missing_rate'])*100:.1f}%)")
        print(f"   均值: {stats['mean']:.6f}")
        print(f"   标准差: {stats['std']:.6f}")
        print(f"   分位数: [{stats['q25']:.4f}, {stats['median']:.4f}, {stats['q75']:.4f}]")
        
        print(f"\n🔍 数据质量:")
        for key, value in validation['data_quality'].items():
            print(f"   {key}: {value}")
        
        print(f"\n⭐ 综合评分: {validation['overall_score']}/100")
        print(f"📋 总体建议: {validation['recommendation']}")
        
        if validation['recommendations']:
            print(f"\n💡 具体建议:")
            for rec in validation['recommendations']:
                print(f"   {rec}")
        
        print("=" * 60)
    
    def run_experimental_batch(self, 
                              financial_data: pd.DataFrame,
                              factor_list: List[str] = None) -> Dict[str, pd.Series]:
        """
        批量运行实验性因子
        
        Parameters:
        -----------
        financial_data : 财务数据
        factor_list : 要测试的因子列表，None表示所有实验性因子
        
        Returns:
        --------
        因子结果字典
        """
        if factor_list is None:
            # 自动发现所有EXPERIMENTAL_开头的方法
            factor_list = [method for method in dir(self) 
                          if method.startswith('calculate_EXPERIMENTAL_')]
        
        results = {}
        logger.info(f"🚀 开始批量测试 {len(factor_list)} 个实验性因子")
        
        for method_name in factor_list:
            factor_name = method_name.replace('calculate_EXPERIMENTAL_', '').replace('_ttm', '')
            
            try:
                method = getattr(self, method_name)
                logger.info(f"   计算因子: {factor_name}")
                
                factor_data = method(financial_data)
                results[factor_name] = factor_data
                
                # 快速验证
                self.quick_validate_factor(factor_data, factor_name, save_results=True)
                
            except Exception as e:
                logger.error(f"   ❌ {factor_name} 计算失败: {e}")
                results[factor_name] = None
        
        logger.info(f"✅ 批量测试完成，成功计算 {sum(1 for v in results.values() if v is not None)} 个因子")
        return results
    
    def export_to_production(self, 
                            experimental_method_name: str,
                            production_name: str,
                            category: str = 'profitability') -> str:
        """
        生成将实验性因子迁移到生产环境的代码模板
        
        Parameters:
        -----------
        experimental_method_name : 实验性方法名
        production_name : 生产环境因子名
        category : 因子分类
        
        Returns:
        --------
        迁移代码模板
        """
        template = f'''
# 将 {experimental_method_name} 迁移到 pure_financial_factors.py

# 1. 在 PureFinancialFactorCalculator 类中添加方法：
def calculate_{production_name}_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    {production_name}因子（经验证有效）
    
    迁移自实验性因子: {experimental_method_name}
    """
    # 从实验性模块复制实现代码
    pass

# 2. 在 _register_all_factors 方法中注册：
methods.update({{
    '{production_name}_ttm': self.calculate_{production_name}_ttm,
}})

# 3. 在 factor_categories 中添加到分类：
self.factor_categories['{category}'].append('{production_name}_ttm')

# 4. 在 get_factor_info 中添加说明：
'{production_name}_ttm': '{production_name} - [添加因子说明]',

# 5. 删除实验性版本：
# 从 experimental_factors.py 中删除 {experimental_method_name} 方法
        '''
        
        print(template)
        return template
    
    # =====================================================
    # 字段验证和说明工具
    # =====================================================
    
    def validate_and_explain_fields(self, field_names: List[str]) -> Dict:
        """
        验证字段并提供中文说明
        
        Parameters:
        -----------
        field_names : list
            字段名列表
            
        Returns:
        --------
        dict
            验证结果和字段说明
        """
        results = {
            'validation': {},
            'explanations': {},
            'missing_fields': [],
            'available_fields': []
        }
        
        for field_name in field_names:
            field_info = self.field_mapper.get_field_info(field_name)
            
            if field_info:
                results['validation'][field_name] = True
                results['explanations'][field_name] = {
                    'chinese_name': field_info['chinese_name'],
                    'table': field_info['table_chinese'] or field_info['table'],
                    'table_en': field_info['table']
                }
                results['available_fields'].append(field_name)
            else:
                results['validation'][field_name] = False
                results['missing_fields'].append(field_name)
        
        return results
    
    def search_similar_fields(self, keyword: str, max_results: int = 10) -> List[Dict]:
        """
        搜索相似字段
        
        Parameters:
        -----------
        keyword : str
            搜索关键词
        max_results : int
            最大结果数
            
        Returns:
        --------
        list
            相似字段列表
        """
        return self.field_mapper.search_fields(keyword)[:max_results]
    
    def print_field_usage_report(self, field_names: List[str]):
        """
        打印字段使用报告
        
        Parameters:
        -----------
        field_names : list
            使用的字段列表
        """
        results = self.validate_and_explain_fields(field_names)
        
        print("=" * 60)
        print("字段使用报告")
        print("=" * 60)
        
        if results['available_fields']:
            print("✅ 可用字段:")
            for field_name in results['available_fields']:
                info = results['explanations'][field_name]
                print(f"   {field_name} -> {info['chinese_name']} ({info['table']})")
        
        if results['missing_fields']:
            print("❌ 未找到字段:")
            for field_name in results['missing_fields']:
                print(f"   {field_name}")
                # 尝试搜索相似字段
                similar = self.search_similar_fields(field_name, 3)
                if similar:
                    print(f"     建议使用: {', '.join([s['field_name'] for s in similar])}")
        
        print("=" * 60)
    
    # =====================================================
    # 你的实验性因子从这里开始添加
    # =====================================================
    
    # TODO: 在这里添加你的实验性因子
    # 复制上面的模板，修改方法名和计算逻辑
    
    pass


# =====================================================
# 便捷函数
# =====================================================

def create_experimental_factor_template(factor_name: str, 
                                       formula_description: str,
                                       economic_meaning: str,
                                       hypothesis: str) -> str:
    """
    生成实验性因子代码模板
    
    Parameters:
    -----------
    factor_name : 因子名称
    formula_description : 计算公式描述
    economic_meaning : 经济含义
    hypothesis : 验证假设
    
    Returns:
    --------
    代码模板字符串
    """
    template = f'''
def calculate_EXPERIMENTAL_{factor_name}_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    实验性因子：{factor_name}
    
    计算公式：{formula_description}
    经济含义：{economic_meaning}
    假设验证：{hypothesis}
    
    Parameters:
    -----------
    financial_data : pd.DataFrame
        财务数据
    **kwargs : dict
        其他参数
        
    Returns:
    --------
    pd.Series
        因子值
    """
    try:
        # 步骤1：验证数据需求
        required_cols = ['earnings', 'revenue']  # 修改为你需要的列
        if not self.validate_data_requirements(financial_data, required_cols):
            raise ValueError(f"Required data not available for {factor_name} calculation")
        
        # 步骤2：提取数据
        extracted_data = self.extract_required_data(financial_data, required_cols)
        
        # 步骤3：实现计算逻辑
        # TODO: 在这里实现你的因子计算
        result = pd.Series(index=financial_data.index, dtype=float)
        
        # 步骤4：返回结果
        logger.info("✨ 实验性因子{factor_name}计算完成")
        return result
        
    except Exception as e:
        logger.error(f"{factor_name}计算失败: {{e}}")
        return pd.Series(index=financial_data.index, dtype=float)
'''
    
    return template


def quick_factor_test(factor_method, financial_data: pd.DataFrame, factor_name: str = "TestFactor"):
    """
    快速测试因子函数
    
    Parameters:
    -----------
    factor_method : 因子计算函数
    financial_data : 财务数据
    factor_name : 因子名称
    """
    calculator = ExperimentalFactorCalculator()
    
    print(f"🧪 快速测试因子: {factor_name}")
    
    try:
        result = factor_method(financial_data)
        validation = calculator.quick_validate_factor(result, factor_name, save_results=False)
        return result, validation
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None, None


def main():
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "auxiliary" / "FinancialData_unified.pkl"
    data_path1 =  r"E:\Documents\PythonProject\StockProject\StockData\LogReturn_daily_o2o.pkl"       
    if not data_path.exists():
        logger.error(f"财务数据文件不存在: {data_path}")
        return None
        
    financial_data = pd.read_pickle(data_path)
    log_return_data = pd.read_pickle(data_path1)
    experimentfactor = ExperimentalFactorCalculator()


# 使用示例
if __name__ == "__main__":
    print("🧪 实验性因子模块")
    print("使用方法:")
    print("1. 复制模板创建新因子")
    print("2. 使用 quick_validate_factor 验证")
    print("3. 使用 export_to_production 迁移到正式环境")