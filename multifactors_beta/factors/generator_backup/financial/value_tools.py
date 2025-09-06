"""
估值因子计算工具

提供PE、PB、PS等估值相关的纯计算函数
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


def calculate_ep_ttm(financial_data: pd.DataFrame, market_cap: pd.Series = None, **kwargs) -> pd.Series:
    """
    计算EP_ttm（TTM盈利收益率）
    
    公式: TTM扣非净利润 / 总市值
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含DEDUCTEDPROFIT列
    market_cap : pd.Series, optional
        总市值数据，如果不提供则需要在kwargs中提供
        
    Returns
    -------
    pd.Series
        EP_ttm计算结果
    """
    try:
        earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
        
        # 检查必要字段
        required_cols = [earnings_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM盈利
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        
        # 获取市值数据
        if market_cap is None:
            if 'market_cap' in kwargs:
                market_cap = kwargs['market_cap']
            else:
                raise ValueError("需要提供market_cap数据")
        
        # 对齐数据并计算EP
        earnings_aligned, cap_aligned = earnings_ttm.align(market_cap, join='inner')
        
        ep = _safe_division(earnings_aligned, cap_aligned)
        
        # 过滤异常值
        ep = ep.replace([np.inf, -np.inf], np.nan)
        
        return ep
        
    except Exception as e:
        logger.error(f"计算EP_ttm失败: {e}")
        return pd.Series(dtype=float)


def calculate_bp_ttm(financial_data: pd.DataFrame, market_cap: pd.Series = None, **kwargs) -> pd.Series:
    """
    计算BP_ttm（TTM账面市值比）
    
    公式: 归属母公司股东权益 / 总市值
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含EQY_BELONGTO_PARCOMSH列
    market_cap : pd.Series, optional
        总市值数据
        
    Returns
    -------
    pd.Series
        BP_ttm计算结果
    """
    try:
        equity_col = 'EQY_BELONGTO_PARCOMSH'  # 归属母公司股东权益
        
        # 检查必要字段
        if equity_col not in financial_data.columns:
            raise ValueError(f"Missing required column: {equity_col}")
        
        # 获取股东权益数据（最新值）
        equity_data = financial_data[equity_col]
        
        # 获取市值数据
        if market_cap is None:
            if 'market_cap' in kwargs:
                market_cap = kwargs['market_cap']
            else:
                raise ValueError("需要提供market_cap数据")
        
        # 对齐数据并计算BP
        equity_aligned, cap_aligned = equity_data.align(market_cap, join='inner')
        
        bp = _safe_division(equity_aligned, cap_aligned)
        
        # 过滤异常值
        bp = bp.replace([np.inf, -np.inf], np.nan)
        
        return bp
        
    except Exception as e:
        logger.error(f"计算BP_ttm失败: {e}")
        return pd.Series(dtype=float)


def calculate_sp_ttm(financial_data: pd.DataFrame, market_cap: pd.Series = None, **kwargs) -> pd.Series:
    """
    计算SP_ttm（TTM销售市值比）
    
    公式: TTM营业收入 / 总市值
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含TOT_OPER_REV列
    market_cap : pd.Series, optional
        总市值数据
        
    Returns
    -------
    pd.Series
        SP_ttm计算结果
    """
    try:
        revenue_col = 'TOT_OPER_REV'  # 营业收入
        
        # 检查必要字段
        required_cols = [revenue_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM营业收入
        ttm_results = calculate_ttm(financial_data)
        revenue_ttm = ttm_results[f'{revenue_col}_ttm']
        
        # 获取市值数据
        if market_cap is None:
            if 'market_cap' in kwargs:
                market_cap = kwargs['market_cap']
            else:
                raise ValueError("需要提供market_cap数据")
        
        # 对齐数据并计算SP
        revenue_aligned, cap_aligned = revenue_ttm.align(market_cap, join='inner')
        
        sp = _safe_division(revenue_aligned, cap_aligned)
        
        # 过滤异常值
        sp = sp.replace([np.inf, -np.inf], np.nan)
        
        return sp
        
    except Exception as e:
        logger.error(f"计算SP_ttm失败: {e}")
        return pd.Series(dtype=float)


def calculate_cfp_ttm(financial_data: pd.DataFrame, market_cap: pd.Series = None, **kwargs) -> pd.Series:
    """
    计算CFP_ttm（TTM现金流市值比）
    
    公式: TTM经营活动现金流量净额 / 总市值
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含NET_CASH_FLOWS_OPER_ACT列
    market_cap : pd.Series, optional
        总市值数据
        
    Returns
    -------
    pd.Series
        CFP_ttm计算结果
    """
    try:
        cashflow_col = 'NET_CASH_FLOWS_OPER_ACT'  # 经营活动现金流量净额
        
        # 检查必要字段
        required_cols = [cashflow_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM现金流
        ttm_results = calculate_ttm(financial_data)
        cashflow_ttm = ttm_results[f'{cashflow_col}_ttm']
        
        # 获取市值数据
        if market_cap is None:
            if 'market_cap' in kwargs:
                market_cap = kwargs['market_cap']
            else:
                raise ValueError("需要提供market_cap数据")
        
        # 对齐数据并计算CFP
        cf_aligned, cap_aligned = cashflow_ttm.align(market_cap, join='inner')
        
        cfp = _safe_division(cf_aligned, cap_aligned)
        
        # 过滤异常值
        cfp = cfp.replace([np.inf, -np.inf], np.nan)
        
        return cfp
        
    except Exception as e:
        logger.error(f"计算CFP_ttm失败: {e}")
        return pd.Series(dtype=float)


def calculate_dp_ttm(financial_data: pd.DataFrame, market_cap: pd.Series = None, **kwargs) -> pd.Series:
    """
    计算DP_ttm（TTM股息收益率）
    
    公式: TTM现金股利 / 总市值
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，包含股利相关数据
    market_cap : pd.Series, optional
        总市值数据
        
    Returns
    -------
    pd.Series
        DP_ttm计算结果
    """
    try:
        # 尝试找到股利相关字段
        dividend_cols = ['CASH_DIV', 'DIVIDEND_PAYABLE', 'CASH_DIVIDENDS']
        dividend_col = None
        
        for col in dividend_cols:
            if col in financial_data.columns:
                dividend_col = col
                break
        
        if dividend_col is None:
            raise ValueError(f"未找到股利字段，尝试过: {dividend_cols}")
        
        # 检查必要字段
        required_cols = [dividend_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 计算TTM股利
        ttm_results = calculate_ttm(financial_data)
        dividend_ttm = ttm_results[f'{dividend_col}_ttm']
        
        # 获取市值数据
        if market_cap is None:
            if 'market_cap' in kwargs:
                market_cap = kwargs['market_cap']
            else:
                raise ValueError("需要提供market_cap数据")
        
        # 对齐数据并计算DP
        dividend_aligned, cap_aligned = dividend_ttm.align(market_cap, join='inner')
        
        dp = _safe_division(dividend_aligned, cap_aligned)
        
        # 过滤异常值
        dp = dp.replace([np.inf, -np.inf], np.nan)
        
        return dp
        
    except Exception as e:
        logger.error(f"计算DP_ttm失败: {e}")
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


# 估值工具函数注册表
VALUE_TOOLS = {
    'EP_ttm': calculate_ep_ttm,
    'BP_ttm': calculate_bp_ttm,
    'SP_ttm': calculate_sp_ttm,
    'CFP_ttm': calculate_cfp_ttm,
    'DP_ttm': calculate_dp_ttm,
}