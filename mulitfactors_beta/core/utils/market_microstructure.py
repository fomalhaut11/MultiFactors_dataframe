"""市场微观结构工具模块"""
import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class MarketCapFilter:
    """市值筛选工具"""
    
    @staticmethod
    def filter_by_market_cap(
        price_data: pd.DataFrame,
        num_groups: int = 10,
        target_group: Union[int, list] = 0,
        exclude_bse: bool = True,
        min_amount: float = 0
    ) -> pd.DataFrame:
        """
        按市值分组筛选股票
        
        Parameters:
        -----------
        price_data : DataFrame，必须包含'MC'(市值)和'amt'(成交额)列
        num_groups : 分组数量
        target_group : 目标组(可以是单个数字或列表)，0表示最小市值组
        exclude_bse : 是否排除北交所股票
        min_amount : 最小成交额要求
        
        Returns:
        --------
        筛选后的股票数据
        """
        result = price_data.copy()
        
        # 排除北交所股票(股票代码以8或更高开头)
        if exclude_bse:
            stock_codes = result.index.get_level_values('StockCodes')
            result = result[stock_codes.str[0].astype(int) < 8]
        
        # 筛选成交额
        if min_amount > 0:
            result = result[result['amt'] > min_amount]
        
        # 按日期分组计算市值分位
        def _assign_groups(date_data: pd.DataFrame) -> pd.Series:
            """为单个日期的数据分配市值组"""
            try:
                # 使用qcut进行分位数分组
                groups = pd.qcut(
                    date_data['MC'], 
                    q=num_groups, 
                    labels=False, 
                    duplicates='drop'
                )
                return groups
            except Exception as e:
                logger.warning(f"Market cap grouping failed: {e}")
                # 如果分组失败，返回空Series
                return pd.Series(np.nan, index=date_data.index)
        
        # 计算市值分组
        market_cap_groups = result.groupby(level='TradingDates').apply(
            lambda x: _assign_groups(x)
        )
        
        # 整理索引
        if isinstance(market_cap_groups.index, pd.MultiIndex) and market_cap_groups.index.nlevels == 3:
            market_cap_groups.index = market_cap_groups.index.droplevel(0)
        
        # 筛选目标组
        if isinstance(target_group, list):
            mask = market_cap_groups.isin(target_group)
        else:
            mask = market_cap_groups == target_group
        
        return result[mask]
    
    @staticmethod
    def get_market_cap_quantiles(
        price_data: pd.DataFrame,
        quantiles: list = [0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> pd.DataFrame:
        """
        获取市值分位数
        
        Parameters:
        -----------
        price_data : DataFrame，必须包含'MC'(市值)列
        quantiles : 要计算的分位数列表
        
        Returns:
        --------
        每个日期的市值分位数
        """
        def _calc_quantiles(date_data: pd.DataFrame) -> pd.Series:
            """计算单个日期的市值分位数"""
            return date_data['MC'].quantile(quantiles)
        
        quantile_data = price_data.groupby(level='TradingDates').apply(_calc_quantiles)
        quantile_data.columns = [f'MC_q{int(q*100)}' for q in quantiles]
        
        return quantile_data


class LiquidityMetrics:
    """流动性指标计算"""
    
    @staticmethod
    def calculate_amihud_illiquidity(
        price_data: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        计算Amihud非流动性指标
        
        Parameters:
        -----------
        price_data : DataFrame，必须包含'close', 'amt'(成交额)列
        window : 计算窗口
        
        Returns:
        --------
        Amihud非流动性指标
        """
        # 计算日收益率
        returns = price_data['close'].groupby(level='StockCodes').pct_change()
        
        # 计算|r|/volume
        illiquidity = np.abs(returns) / (price_data['amt'] + 1e-10)
        
        # 计算滚动平均
        amihud = illiquidity.groupby(level='StockCodes').rolling(
            window=window, min_periods=1
        ).mean()
        
        # 整理索引
        amihud.index = amihud.index.droplevel(0)
        
        return amihud * 1e6  # 放大便于观察
    
    @staticmethod
    def calculate_turnover_rate(
        price_data: pd.DataFrame,
        shares_outstanding: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        计算换手率
        
        Parameters:
        -----------
        price_data : DataFrame，必须包含'volume'(成交量)列
        shares_outstanding : 流通股本数据，如果为None则使用总股本
        
        Returns:
        --------
        换手率
        """
        if shares_outstanding is None:
            # 如果没有提供流通股本，假设使用总市值/股价估算
            if 'MC' in price_data.columns and 'close' in price_data.columns:
                shares_outstanding = price_data['MC'] / price_data['close']
            else:
                raise ValueError("Need shares_outstanding or MC and close columns")
        
        turnover = price_data['volume'] / shares_outstanding
        return turnover
    
    @staticmethod
    def calculate_bid_ask_spread(
        order_data: pd.DataFrame,
        method: str = "quoted"
    ) -> pd.Series:
        """
        计算买卖价差
        
        Parameters:
        -----------
        order_data : DataFrame，必须包含'bid1'(买一价)和'ask1'(卖一价)列
        method : 
            - "quoted": 报价价差
            - "relative": 相对价差
            - "effective": 有效价差(需要成交价格)
        
        Returns:
        --------
        买卖价差
        """
        if method == "quoted":
            spread = order_data['ask1'] - order_data['bid1']
        elif method == "relative":
            mid_price = (order_data['ask1'] + order_data['bid1']) / 2
            spread = (order_data['ask1'] - order_data['bid1']) / mid_price
        elif method == "effective":
            if 'trade_price' not in order_data.columns:
                raise ValueError("Need trade_price column for effective spread")
            mid_price = (order_data['ask1'] + order_data['bid1']) / 2
            spread = 2 * np.abs(order_data['trade_price'] - mid_price)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return spread