"""
财务因子注册模块

直接组合generators的基础工具实现财务因子计算
"""
import pandas as pd
import numpy as np
import logging

from .factor_registry import register_factor

logger = logging.getLogger(__name__)

# 导入generators基础工具
from ..generators.financial import calculate_ttm


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


# ========== 盈利能力因子注册 ==========

@register_factor(
    name='ROE_ttm',
    category='profitability',
    description='TTM净资产收益率，衡量企业股东权益的盈利能力',
    dependencies=['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH', 'd_year', 'd_quarter'],
    formula='TTM扣非净利润 / 股东权益',
    data_frequency='季报',
    calculation_method='TTM'
)
def roe_ttm_factor(financial_data, **kwargs):
    """ROE_ttm因子计算"""
    try:
        earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
        equity_col = 'EQY_BELONGTO_PARCOMSH'  # 归属母公司股东权益
        
        # 检查必要字段
        required_cols = [earnings_col, equity_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 使用generators计算TTM
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        
        # 股东权益取最新值（季末值）
        equity_data = financial_data[equity_col]
        
        # 对齐数据并计算ROE
        earnings_aligned, equity_aligned = earnings_ttm.align(equity_data, join='inner')
        
        # 计算ROE
        roe = _safe_division(earnings_aligned, equity_aligned)
        
        return roe
        
    except Exception as e:
        logger.error(f"计算ROE_ttm失败: {e}")
        return pd.Series(dtype=float)


@register_factor(
    name='ROA_ttm',
    category='profitability',
    description='TTM总资产收益率，衡量企业总资产的盈利效率',
    dependencies=['DEDUCTEDPROFIT', 'TOT_ASSETS', 'd_year', 'd_quarter'],
    formula='TTM扣非净利润 / 总资产',
    data_frequency='季报',
    calculation_method='TTM'
)
def roa_ttm_factor(financial_data, **kwargs):
    """ROA_ttm因子计算"""
    try:
        earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
        assets_col = 'TOT_ASSETS'  # 总资产

        # 检查必要字段
        required_cols = [earnings_col, assets_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 使用generators计算TTM
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']

        # 总资产取最新值
        assets_data = financial_data[assets_col]

        # 对齐数据并计算ROA
        earnings_aligned, assets_aligned = earnings_ttm.align(assets_data, join='inner')

        # 计算ROA
        roa = _safe_division(earnings_aligned, assets_aligned)

        return roa

    except Exception as e:
        logger.error(f"计算ROA_ttm失败: {e}")
        return pd.Series(dtype=float)


@register_factor(
    name='GrossProfitMargin_ttm',
    category='profitability',
    description='TTM毛利率，衡量企业产品定价能力和成本控制水平',
    dependencies=['TOT_OPER_REV', 'TOT_OPER_COST', 'd_year', 'd_quarter'],
    formula='(TTM营业收入 - TTM营业成本) / TTM营业收入',
    data_frequency='季报',
    calculation_method='TTM'
)
def gross_profit_margin_ttm_factor(financial_data, **kwargs):
    """GrossProfitMargin_ttm因子计算"""
    try:
        revenue_col = 'TOT_OPER_REV'  # 营业收入
        cost_col = 'TOT_OPER_COST'    # 营业成本

        # 检查必要字段
        required_cols = [revenue_col, cost_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 使用generators计算TTM
        ttm_results = calculate_ttm(financial_data)
        revenue_ttm = ttm_results[f'{revenue_col}_ttm']
        cost_ttm = ttm_results[f'{cost_col}_ttm']

        # 对齐数据并计算毛利率
        revenue_aligned, cost_aligned = revenue_ttm.align(cost_ttm, join='inner')

        # 计算毛利率
        gross_profit_margin = _safe_division(
            revenue_aligned - cost_aligned,
            revenue_aligned
        )

        return gross_profit_margin

    except Exception as e:
        logger.error(f"计算毛利率TTM失败: {e}")
        return pd.Series(dtype=float)


# ========== 估值因子注册 ==========

@register_factor(
    name='EP_ttm',
    category='value',
    description='TTM盈利收益率，PE的倒数，衡量估值水平',
    dependencies=['DEDUCTEDPROFIT', 'd_year', 'd_quarter'],
    formula='TTM扣非净利润 / 总市值',
    data_frequency='季报+日频市值',
    calculation_method='TTM',
    requires_market_data=True
)
def ep_ttm_factor(financial_data, market_cap=None, **kwargs):
    """EP_ttm因子计算"""
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

        return ep

    except Exception as e:
        logger.error(f"计算EP_ttm失败: {e}")
        return pd.Series(dtype=float)


@register_factor(
    name='BP_ttm',
    category='value',
    description='TTM账面市值比，PB的倒数，衡量账面价值相对市值',
    dependencies=['EQY_BELONGTO_PARCOMSH'],
    formula='归属母公司股东权益 / 总市值',
    data_frequency='季报+日频市值',
    calculation_method='最新值',
    requires_market_data=True
)
def bp_ttm_factor(financial_data, market_cap=None, **kwargs):
    """BP_ttm因子计算"""
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

        return bp

    except Exception as e:
        logger.error(f"计算BP_ttm失败: {e}")
        return pd.Series(dtype=float)


# ========== 质量因子注册 ==========

@register_factor(
    name='AccrualRatio_ttm',
    category='quality',
    description='TTM应计项目比率，衡量盈利质量，值越小质量越高',
    dependencies=['NET_PROFIT_INCL_MIN_INT_INC', 'NET_CASH_FLOWS_OPER_ACT', 'd_year', 'd_quarter'],
    formula='(TTM净利润 - TTM经营现金流) / abs(TTM净利润)',
    data_frequency='季报',
    calculation_method='TTM'
)
def accrual_ratio_ttm_factor(financial_data, **kwargs):
    """AccrualRatio_ttm因子计算"""
    try:
        profit_col = 'NET_PROFIT_INCL_MIN_INT_INC'  # 净利润
        cashflow_col = 'NET_CASH_FLOWS_OPER_ACT'    # 经营现金流

        # 检查必要字段
        required_cols = [profit_col, cashflow_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 使用generators计算TTM
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

        return accrual_ratio

    except Exception as e:
        logger.error(f"计算应计项目比率TTM失败: {e}")
        return pd.Series(dtype=float)


# 导出注册的因子函数列表
__all__ = [
    'roe_ttm_factor',
    'roa_ttm_factor',
    'gross_profit_margin_ttm_factor',
    'ep_ttm_factor',
    'bp_ttm_factor',
    'accrual_ratio_ttm_factor',
]
