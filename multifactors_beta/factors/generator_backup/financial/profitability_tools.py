"""
盈利能力因子计算工具

提供ROE、ROA、毛利率等盈利能力相关的纯计算函数
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


def calculate_roe_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算ROE_ttm（TTM净资产收益率）
    
    公式: TTM扣非净利润 / 股东权益
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含DEDUCTEDPROFIT和EQY_BELONGTO_PARCOMSH列
        
    Returns
    -------
    pd.Series
        ROE_ttm计算结果
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
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        
        # 股东权益取最新值（季末值）
        equity_data = financial_data[equity_col]
        
        # 对齐数据并计算ROE
        earnings_aligned, equity_aligned = earnings_ttm.align(equity_data, join='inner')
        
        # ROE = 净利润TTM / 股东权益
        roe = _safe_division(earnings_aligned, equity_aligned)
        
        # 过滤异常值
        roe = roe.replace([np.inf, -np.inf], np.nan)
        
        return roe
        
    except Exception as e:
        logger.error(f"计算ROE_ttm失败: {e}")
        return pd.Series(dtype=float)


def calculate_roa_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算ROA_ttm（TTM总资产收益率）
    
    公式: TTM扣非净利润 / 总资产
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含DEDUCTEDPROFIT和TOT_ASSETS列
        
    Returns
    -------
    pd.Series
        ROA_ttm计算结果
    """
    try:
        # 使用实际的字段名
        earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
        assets_col = 'TOT_ASSETS'  # 总资产
        
        # 检查必要字段是否存在
        required_cols = [earnings_col, assets_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        
        # 总资产取最新值（季末值）
        assets_data = financial_data[assets_col]
        
        # 对齐数据并计算ROA
        earnings_aligned, assets_aligned = earnings_ttm.align(assets_data, join='inner')
        
        # ROA = 净利润TTM / 总资产
        roa = _safe_division(earnings_aligned, assets_aligned)
        
        # 过滤异常值
        roa = roa.replace([np.inf, -np.inf], np.nan)
        
        return roa
        
    except Exception as e:
        logger.error(f"计算ROA_ttm失败: {e}")
        return pd.Series(dtype=float)


def calculate_gross_profit_margin_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算毛利率TTM
    
    公式: (营业收入-营业成本)TTM / 营业收入TTM
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含TOT_OPER_REV和TOT_OPER_COST列
        
    Returns
    -------
    pd.Series
        毛利率TTM计算结果
    """
    try:
        revenue_col = 'TOT_OPER_REV'  # 营业收入
        cost_col = 'TOT_OPER_COST'    # 营业成本
        
        # 检查必要字段
        required_cols = [revenue_col, cost_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        revenue_ttm = ttm_results[f'{revenue_col}_ttm']
        cost_ttm = ttm_results[f'{cost_col}_ttm']
        
        # 对齐数据
        revenue_aligned, cost_aligned = revenue_ttm.align(cost_ttm, join='inner')
        
        # 计算毛利率
        gross_profit_margin = _safe_division(
            revenue_aligned - cost_aligned, 
            revenue_aligned
        )
        
        # 过滤异常值
        gross_profit_margin = gross_profit_margin.replace([np.inf, -np.inf], np.nan)
        
        return gross_profit_margin
        
    except Exception as e:
        logger.error(f"计算毛利率TTM失败: {e}")
        return pd.Series(dtype=float)


def calculate_profit_cost_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算利润成本比率TTM
    
    公式: TTM扣非净利润 / (TTM财务费用 + TTM所得税)
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含DEDUCTEDPROFIT、FIN_EXP_IS和TAX列
        
    Returns
    -------
    pd.Series
        利润成本比率TTM计算结果
    """
    try:
        earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
        fin_exp_col = 'FIN_EXP_IS'       # 财务费用
        tax_col = 'TAX'                  # 所得税
        
        # 检查必要字段
        required_cols = [earnings_col, fin_exp_col, tax_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        fin_exp_ttm = ttm_results[f'{fin_exp_col}_ttm']
        tax_ttm = ttm_results[f'{tax_col}_ttm']
        
        # 对齐数据
        earnings_aligned = earnings_ttm
        fin_exp_aligned = fin_exp_ttm.reindex(earnings_aligned.index)
        tax_aligned = tax_ttm.reindex(earnings_aligned.index)
        
        # 计算分母：财务费用 + 所得税
        total_cost = fin_exp_aligned + tax_aligned
        
        # 计算利润成本比率
        profitcost = _safe_division(earnings_aligned, total_cost)
        
        # 过滤异常值
        profitcost = profitcost.replace([np.inf, -np.inf], np.nan)
        
        return profitcost
        
    except Exception as e:
        logger.error(f"计算利润成本比率TTM失败: {e}")
        return pd.Series(dtype=float)


def calculate_operating_profit_margin_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算营业利润率TTM
    
    公式: TTM营业利润 / TTM营业收入
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含OPER_PROFIT和TOT_OPER_REV列
        
    Returns
    -------
    pd.Series
        营业利润率TTM计算结果
    """
    try:
        profit_col = 'OPER_PROFIT'    # 营业利润
        revenue_col = 'TOT_OPER_REV'  # 营业收入
        
        # 检查必要字段
        required_cols = [profit_col, revenue_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM
        ttm_results = calculate_ttm(financial_data)
        profit_ttm = ttm_results[f'{profit_col}_ttm']
        revenue_ttm = ttm_results[f'{revenue_col}_ttm']
        
        # 对齐数据并计算营业利润率
        profit_aligned, revenue_aligned = profit_ttm.align(revenue_ttm, join='inner')
        
        operating_margin = _safe_division(profit_aligned, revenue_aligned)
        
        # 过滤异常值
        operating_margin = operating_margin.replace([np.inf, -np.inf], np.nan)
        
        return operating_margin
        
    except Exception as e:
        logger.error(f"计算营业利润率TTM失败: {e}")
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


# 工具函数注册表
PROFITABILITY_TOOLS = {
    'ROE_ttm': calculate_roe_ttm,
    'ROA_ttm': calculate_roa_ttm, 
    'GrossProfitMargin_ttm': calculate_gross_profit_margin_ttm,
    'ProfitCost_ttm': calculate_profit_cost_ttm,
    'OperatingProfitMargin_ttm': calculate_operating_profit_margin_ttm,
}