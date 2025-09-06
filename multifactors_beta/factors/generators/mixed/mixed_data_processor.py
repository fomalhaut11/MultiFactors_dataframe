"""
混合数据处理器

提供财务数据与市场数据结合处理的纯工具函数
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MixedDataProcessor:
    """混合数据处理器 - 纯静态方法工具类"""
    
    @staticmethod
    def align_financial_with_market(financial_data: pd.Series, 
                                   market_data: pd.Series,
                                   method: str = 'ffill') -> Tuple[pd.Series, pd.Series]:
        """
        对齐财务数据与市场数据的时间序列
        
        Parameters
        ----------
        financial_data : pd.Series
            财务数据，通常是季报数据，索引为 [ReportDates, StockCodes]
        market_data : pd.Series  
            市场数据，通常是日频数据，索引为 [TradingDates, StockCodes]
        method : str
            对齐方法，'ffill' 或 'nearest'
            
        Returns
        -------
        tuple
            对齐后的 (financial_aligned, market_aligned)
        """
        if method == 'ffill':
            # 前向填充财务数据到市场数据频率
            financial_aligned = financial_data.reindex(
                market_data.index, method='ffill'
            )
            market_aligned = market_data
        else:
            # 其他对齐方法
            financial_aligned, market_aligned = financial_data.align(
                market_data, join='inner', method=method
            )
            
        return financial_aligned, market_aligned
    
    @staticmethod
    def calculate_relative_ratio(numerator_data: pd.Series,
                               denominator_data: pd.Series,
                               handle_zero: str = 'nan') -> pd.Series:
        """
        计算相对比率，安全处理分母为零的情况
        
        Parameters
        ----------
        numerator_data : pd.Series
            分子数据
        denominator_data : pd.Series
            分母数据  
        handle_zero : str
            零值处理方式，'nan', 'inf', 'drop'
            
        Returns
        -------
        pd.Series
            比率结果
        """
        # 对齐数据
        num_aligned, den_aligned = numerator_data.align(
            denominator_data, join='inner'
        )
        
        # 计算比率
        ratio = num_aligned / den_aligned
        
        # 处理异常值
        if handle_zero == 'nan':
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
        elif handle_zero == 'drop':
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        # 'inf' 选项保持无穷值
        
        return ratio
    
    @staticmethod
    def calculate_market_cap_weighted_ratio(financial_metric: pd.Series,
                                          market_cap: pd.Series) -> pd.Series:
        """
        计算市值加权的财务指标比率
        
        Parameters
        ----------
        financial_metric : pd.Series
            财务指标数据
        market_cap : pd.Series
            市值数据
            
        Returns
        -------
        pd.Series
            市值调整后的比率
        """
        return MixedDataProcessor.calculate_relative_ratio(
            financial_metric, market_cap, handle_zero='nan'
        )
    
    @staticmethod
    def standardize_by_market_beta(factor_data: pd.Series,
                                 market_returns: pd.Series,
                                 window: int = 252) -> pd.Series:
        """
        使用市场Beta标准化因子数据
        
        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        market_returns : pd.Series
            市场收益率数据
        window : int
            滚动窗口大小
            
        Returns
        -------
        pd.Series
            Beta调整后的因子
        """
        # 计算滚动Beta
        aligned_factor, aligned_market = factor_data.align(
            market_returns, join='inner'
        )
        
        # 按股票分组计算Beta
        result = pd.Series(index=aligned_factor.index, dtype=float)
        
        for stock in aligned_factor.index.get_level_values('StockCodes').unique():
            stock_factor = aligned_factor.xs(stock, level='StockCodes')
            stock_market = aligned_market.xs(stock, level='StockCodes')
            
            # 计算滚动相关系数作为Beta的代理
            rolling_corr = stock_factor.rolling(window).corr(stock_market)
            
            # Beta调整
            adjusted = stock_factor / (rolling_corr + 1e-8)  # 避免除零
            
            result.loc[(slice(None), stock)] = adjusted.values
            
        return result


# 便捷函数定义
def align_financial_with_market(financial_data: pd.Series, 
                              market_data: pd.Series,
                              method: str = 'ffill') -> Tuple[pd.Series, pd.Series]:
    """对齐财务数据与市场数据 - 便捷函数"""
    return MixedDataProcessor.align_financial_with_market(
        financial_data, market_data, method
    )


def calculate_relative_ratio(numerator_data: pd.Series,
                           denominator_data: pd.Series,
                           handle_zero: str = 'nan') -> pd.Series:
    """计算相对比率 - 便捷函数"""
    return MixedDataProcessor.calculate_relative_ratio(
        numerator_data, denominator_data, handle_zero
    )


def calculate_market_cap_weighted_ratio(financial_metric: pd.Series,
                                      market_cap: pd.Series) -> pd.Series:
    """计算市值加权比率 - 便捷函数"""
    return MixedDataProcessor.calculate_market_cap_weighted_ratio(
        financial_metric, market_cap
    )