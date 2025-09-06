"""
盈利质量因子计算工具

提供盈利质量、现金流质量等相关的纯计算函数
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
import sys
import os

# 添加路径以导入generators模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from generators.financial import calculate_ttm

logger = logging.getLogger(__name__)


def calculate_accrual_ratio_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算应计项目比率TTM
    
    公式: (TTM净利润 - TTM经营现金流) / TTM净利润
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含NET_PROFIT_INCL_MIN_INT_INC和NET_CASH_FLOWS_OPER_ACT列
        
    Returns
    -------
    pd.Series
        应计项目比率TTM计算结果
    """
    try:
        profit_col = 'NET_PROFIT_INCL_MIN_INT_INC'  # 净利润
        cashflow_col = 'NET_CASH_FLOWS_OPER_ACT'    # 经营现金流
        
        # 检查必要字段
        required_cols = [profit_col, cashflow_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        profit_ttm = ttm_results[f'{profit_col}_ttm']
        cashflow_ttm = ttm_results[f'{cashflow_col}_ttm']
        
        # 对齐数据
        profit_aligned, cf_aligned = profit_ttm.align(cashflow_ttm, join='inner')
        
        # 计算应计项目比率
        accrual_ratio = _safe_division(
            profit_aligned - cf_aligned, 
            profit_aligned.abs()
        )
        
        # 过滤异常值
        accrual_ratio = accrual_ratio.replace([np.inf, -np.inf], np.nan)
        
        return accrual_ratio
        
    except Exception as e:
        logger.error(f"计算应计项目比率TTM失败: {e}")
        return pd.Series(dtype=float)


def calculate_cash_flow_coverage_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算现金流覆盖率TTM
    
    公式: TTM经营现金流 / TTM净利润
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含相关现金流和利润字段
        
    Returns
    -------
    pd.Series
        现金流覆盖率TTM计算结果
    """
    try:
        cashflow_col = 'NET_CASH_FLOWS_OPER_ACT'    # 经营现金流
        profit_col = 'NET_PROFIT_INCL_MIN_INT_INC'   # 净利润
        
        # 检查必要字段
        required_cols = [cashflow_col, profit_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        cashflow_ttm = ttm_results[f'{cashflow_col}_ttm']
        profit_ttm = ttm_results[f'{profit_col}_ttm']
        
        # 对齐数据并计算覆盖率
        cf_aligned, profit_aligned = cashflow_ttm.align(profit_ttm, join='inner')
        
        coverage = _safe_division(cf_aligned, profit_aligned)
        
        # 过滤异常值
        coverage = coverage.replace([np.inf, -np.inf], np.nan)
        
        return coverage
        
    except Exception as e:
        logger.error(f"计算现金流覆盖率TTM失败: {e}")
        return pd.Series(dtype=float)


def calculate_earning_persistence_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算盈利持续性TTM
    
    基于营业利润的稳定性衡量盈利质量
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含OPER_PROFIT列
        
    Returns
    -------
    pd.Series
        盈利持续性TTM计算结果
    """
    try:
        profit_col = 'OPER_PROFIT'  # 营业利润
        
        # 检查必要字段
        required_cols = [profit_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 按股票分组计算营业利润的变异系数
        result = pd.Series(index=financial_data.index, dtype=float)
        
        for stock in financial_data.index.get_level_values('StockCodes').unique():
            stock_data = financial_data.xs(stock, level='StockCodes')[profit_col]
            
            # 计算过去4个季度的变异系数（如果有足够数据）
            if len(stock_data) >= 4:
                rolling_cv = stock_data.rolling(4).apply(
                    lambda x: x.std() / (abs(x.mean()) + 1e-8)
                )
                # 盈利持续性 = 1 / (变异系数 + 1)，变异系数越小，持续性越高
                persistence = 1 / (rolling_cv + 1)
                result.loc[(slice(None), stock)] = persistence.values
        
        return result
        
    except Exception as e:
        logger.error(f"计算盈利持续性TTM失败: {e}")
        return pd.Series(dtype=float)


def calculate_asset_turnover_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算资产周转率TTM
    
    公式: TTM营业收入 / 平均总资产
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含TOT_OPER_REV和TOT_ASSETS列
        
    Returns
    -------
    pd.Series
        资产周转率TTM计算结果
    """
    try:
        revenue_col = 'TOT_OPER_REV'  # 营业收入
        assets_col = 'TOT_ASSETS'     # 总资产
        
        # 检查必要字段
        required_cols = [revenue_col, assets_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM营业收入
        ttm_results = calculate_ttm(financial_data)
        revenue_ttm = ttm_results[f'{revenue_col}_ttm']
        
        # 计算平均总资产（当期和前期的平均）
        assets_current = financial_data[assets_col]
        assets_prev = financial_data.groupby('StockCodes')[assets_col].shift(1)
        assets_avg = (assets_current + assets_prev) / 2
        
        # 对齐数据并计算周转率
        revenue_aligned, assets_aligned = revenue_ttm.align(assets_avg, join='inner')
        
        turnover = _safe_division(revenue_aligned, assets_aligned)
        
        # 过滤异常值
        turnover = turnover.replace([np.inf, -np.inf], np.nan)
        
        return turnover
        
    except Exception as e:
        logger.error(f"计算资产周转率TTM失败: {e}")
        return pd.Series(dtype=float)


def calculate_receivables_turnover_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算应收账款周转率TTM
    
    公式: TTM营业收入 / 平均应收账款
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含TOT_OPER_REV和ACCT_RCV列
        
    Returns
    -------
    pd.Series
        应收账款周转率TTM计算结果
    """
    try:
        revenue_col = 'TOT_OPER_REV'  # 营业收入
        receivables_col = 'ACCT_RCV'  # 应收账款
        
        # 检查必要字段
        required_cols = [revenue_col, receivables_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM营业收入
        ttm_results = calculate_ttm(financial_data)
        revenue_ttm = ttm_results[f'{revenue_col}_ttm']
        
        # 计算平均应收账款
        receivables_current = financial_data[receivables_col]
        receivables_prev = financial_data.groupby('StockCodes')[receivables_col].shift(1)
        receivables_avg = (receivables_current + receivables_prev) / 2
        
        # 对齐数据并计算周转率
        revenue_aligned, receivables_aligned = revenue_ttm.align(receivables_avg, join='inner')
        
        turnover = _safe_division(revenue_aligned, receivables_aligned)
        
        # 过滤异常值
        turnover = turnover.replace([np.inf, -np.inf], np.nan)
        
        return turnover
        
    except Exception as e:
        logger.error(f"计算应收账款周转率TTM失败: {e}")
        return pd.Series(dtype=float)


def _safe_division(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    安全的除法运算，处理除零和无穷值
    
    Parameters
    ----------
    numerator : pd.Series
        分子
    denominator : pd.Series
        分母
        
    Returns
    -------
    pd.Series
        除法结果，异常值设为NaN
    """
    # 避免除零
    result = numerator / denominator.replace(0, np.nan)
    
    # 处理无穷值
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result


# 质量工具函数注册表
QUALITY_TOOLS = {
    'AccrualRatio_ttm': calculate_accrual_ratio_ttm,
    'CashFlowCoverage_ttm': calculate_cash_flow_coverage_ttm,
    'EarningPersistence_ttm': calculate_earning_persistence_ttm,
    'AssetTurnover_ttm': calculate_asset_turnover_ttm,
    'ReceivablesTurnover_ttm': calculate_receivables_turnover_ttm,
}