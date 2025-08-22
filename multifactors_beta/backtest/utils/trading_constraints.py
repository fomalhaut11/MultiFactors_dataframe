"""
交易约束检查模块

检查中国A股市场的交易限制，包括停牌、涨跌停等
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class TradingConstraints:
    """
    交易约束检查器
    
    检查股票是否可以正常交易，包括：
    1. 停牌检查（无成交量或最高价=最低价）
    2. 涨跌停检查（基于开盘价相对昨日收盘价的涨幅）
    """
    
    def __init__(self,
                 main_board_limit: float = 0.10,  # 主板涨跌停限制10%
                 growth_board_limit: float = 0.20,  # 创业板涨跌停限制20%
                 star_board_limit: float = 0.20):   # 科创板涨跌停限制20%
        """
        初始化交易约束检查器
        
        Parameters
        ----------
        main_board_limit : float
            主板涨跌停限制（默认10%）
        growth_board_limit : float
            创业板涨跌停限制（默认20%）
        star_board_limit : float
            科创板涨跌停限制（默认20%）
        """
        self.main_board_limit = main_board_limit
        self.growth_board_limit = growth_board_limit
        self.star_board_limit = star_board_limit
        
        logger.info(f"交易约束检查器初始化: 主板{main_board_limit:.1%}, 创业板{growth_board_limit:.1%}, 科创板{star_board_limit:.1%}")
    
    def check_trading_availability(self,
                                 current_date: pd.Timestamp,
                                 stocks: list,
                                 market_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        检查股票当日是否可以正常交易
        
        Parameters
        ----------
        current_date : pd.Timestamp
            当前日期
        stocks : list
            股票代码列表
        market_data : Dict[str, pd.DataFrame]
            市场数据字典，包含：
            - 'open': 开盘价
            - 'high': 最高价  
            - 'low': 最低价
            - 'close': 收盘价
            - 'volume': 成交量
            - 'prev_close': 昨日收盘价（复权）
            
        Returns
        -------
        Dict[str, bool]
            股票可交易状态 {股票代码: 是否可交易}
        """
        tradable_status = {}
        
        for stock in stocks:
            try:
                # 检查是否停牌
                is_suspended = self._check_suspension(current_date, stock, market_data)
                
                # 检查是否涨跌停
                is_limit_up_down = self._check_limit_up_down(current_date, stock, market_data)
                
                # 只有非停牌且非涨跌停才可交易
                tradable_status[stock] = not (is_suspended or is_limit_up_down)
                
                if not tradable_status[stock]:
                    reason = "停牌" if is_suspended else "涨跌停"
                    logger.debug(f"{current_date.date()} {stock}: 不可交易 ({reason})")
                    
            except Exception as e:
                logger.warning(f"检查 {stock} 交易状态时出错: {str(e)}")
                tradable_status[stock] = False
        
        return tradable_status
    
    def _check_suspension(self,
                         date: pd.Timestamp,
                         stock: str,
                         market_data: Dict[str, pd.DataFrame]) -> bool:
        """
        检查股票是否停牌
        
        停牌判断标准：
        1. 成交量为0
        2. 最高价等于最低价
        """
        try:
            # 获取当日数据
            volume = self._get_stock_data(date, stock, market_data, 'volume')
            high = self._get_stock_data(date, stock, market_data, 'high')
            low = self._get_stock_data(date, stock, market_data, 'low')
            
            # 检查成交量
            if volume is None or volume <= 0:
                logger.debug(f"{stock} {date.date()}: 成交量为0，判定为停牌")
                return True
            
            # 检查最高价是否等于最低价
            if high is not None and low is not None and abs(high - low) < 0.001:
                logger.debug(f"{stock} {date.date()}: 最高价=最低价，判定为停牌")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"检查停牌状态时出错 {stock}: {str(e)}")
            return True  # 出错时保守处理，认为停牌
    
    def _check_limit_up_down(self,
                           date: pd.Timestamp,
                           stock: str,
                           market_data: Dict[str, pd.DataFrame]) -> bool:
        """
        检查股票是否涨跌停
        
        涨跌停判断标准：
        1. 主板股票（非300、688开头）：开盘价相对昨收盘价涨幅≥10%
        2. 创业板股票（300开头）：开盘价相对昨收盘价涨幅≥20%
        3. 科创板股票（688开头）：开盘价相对昨收盘价涨幅≥20%
        """
        try:
            # 获取当日开盘价
            open_price = self._get_stock_data(date, stock, market_data, 'open')
            if open_price is None:
                return True
            
            # 获取昨日收盘价（复权）
            prev_close = self._get_stock_data(date, stock, market_data, 'prev_close')
            if prev_close is None or prev_close <= 0:
                # 如果没有prev_close，尝试从close数据中获取前一日数据
                prev_close = self._get_prev_close_from_history(date, stock, market_data)
                if prev_close is None or prev_close <= 0:
                    return True
            
            # 计算涨幅
            price_change_ratio = (open_price - prev_close) / prev_close
            
            # 根据股票类型确定涨跌停限制
            limit_ratio = self._get_limit_ratio(stock)
            
            # 检查是否涨跌停（涨幅的绝对值达到限制）
            if abs(price_change_ratio) >= limit_ratio:
                direction = "涨停" if price_change_ratio > 0 else "跌停"
                logger.debug(f"{stock} {date.date()}: {direction} (涨幅{price_change_ratio:.2%})")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"检查涨跌停状态时出错 {stock}: {str(e)}")
            return True  # 出错时保守处理，认为涨跌停
    
    def _get_stock_data(self,
                       date: pd.Timestamp,
                       stock: str,
                       market_data: Dict[str, pd.DataFrame],
                       field: str) -> Optional[float]:
        """获取指定股票指定字段的数据"""
        try:
            if field not in market_data:
                return None
            
            data_df = market_data[field]
            if stock not in data_df.columns:
                return None
            
            if date not in data_df.index:
                return None
            
            value = data_df.loc[date, stock]
            return value if not pd.isna(value) else None
            
        except Exception:
            return None
    
    def _get_prev_close_from_history(self,
                                   date: pd.Timestamp,
                                   stock: str,
                                   market_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """从历史收盘价数据中获取前一日收盘价"""
        try:
            if 'close' not in market_data:
                return None
            
            close_df = market_data['close']
            if stock not in close_df.columns:
                return None
            
            # 获取当前日期之前的数据
            stock_series = close_df[stock]
            prev_data = stock_series[stock_series.index < date]
            
            if len(prev_data) == 0:
                return None
            
            # 获取最近一个有效的收盘价
            last_valid_price = prev_data.dropna().iloc[-1] if len(prev_data.dropna()) > 0 else None
            return last_valid_price
            
        except Exception:
            return None
    
    def _get_limit_ratio(self, stock: str) -> float:
        """根据股票代码确定涨跌停限制比例"""
        if stock.startswith('300'):
            # 创业板
            return self.growth_board_limit
        elif stock.startswith('688'):
            # 科创板
            return self.star_board_limit
        else:
            # 主板
            return self.main_board_limit
    
    def get_tradable_stocks(self,
                          date: pd.Timestamp,
                          stocks: list,
                          market_data: Dict[str, pd.DataFrame]) -> Set[str]:
        """
        获取当日可交易的股票集合
        
        Returns
        -------
        Set[str]
            可交易股票代码集合
        """
        tradable_status = self.check_trading_availability(date, stocks, market_data)
        return {stock for stock, tradable in tradable_status.items() if tradable}
    
    def filter_tradable_weights(self,
                              date: pd.Timestamp,
                              target_weights: pd.Series,
                              market_data: Dict[str, pd.DataFrame],
                              rebalance_method: str = 'proportional') -> pd.Series:
        """
        过滤掉不可交易股票的权重，并重新分配
        
        Parameters
        ----------
        date : pd.Timestamp
            当前日期
        target_weights : pd.Series
            目标权重
        market_data : Dict[str, pd.DataFrame]
            市场数据
        rebalance_method : str
            重新分配方法：
            - 'proportional': 按比例重新分配
            - 'equal': 平均分配给可交易股票
            - 'zero': 不可交易股票权重设为0，不重新分配
            
        Returns
        -------
        pd.Series
            调整后的权重
        """
        stocks = list(target_weights.index)
        tradable_stocks = self.get_tradable_stocks(date, stocks, market_data)
        
        # 分离可交易和不可交易股票的权重
        tradable_weights = target_weights[target_weights.index.isin(tradable_stocks)]
        untradable_weights = target_weights[~target_weights.index.isin(tradable_stocks)]
        
        if len(untradable_weights) == 0:
            # 所有股票都可交易
            return target_weights
        
        logger.info(f"{date.date()}: {len(untradable_weights)}只股票不可交易，总权重{untradable_weights.sum():.4f}")
        
        # 重新分配不可交易股票的权重
        if rebalance_method == 'proportional':
            # 按比例重新分配
            if tradable_weights.sum() > 0:
                adjustment_factor = (tradable_weights.sum() + untradable_weights.sum()) / tradable_weights.sum()
                adjusted_tradable_weights = tradable_weights * adjustment_factor
            else:
                adjusted_tradable_weights = tradable_weights
        elif rebalance_method == 'equal':
            # 平均分配
            if len(tradable_stocks) > 0:
                total_weight_to_distribute = tradable_weights.sum() + untradable_weights.sum()
                equal_weight = total_weight_to_distribute / len(tradable_stocks)
                adjusted_tradable_weights = pd.Series(equal_weight, index=tradable_stocks)
            else:
                adjusted_tradable_weights = tradable_weights
        elif rebalance_method == 'zero':
            # 不重新分配
            adjusted_tradable_weights = tradable_weights
        else:
            raise ValueError(f"未知的重新分配方法: {rebalance_method}")
        
        # 构建最终权重
        final_weights = pd.Series(0.0, index=target_weights.index)
        final_weights[adjusted_tradable_weights.index] = adjusted_tradable_weights
        
        logger.debug(f"权重调整完成: 可交易权重和={final_weights.sum():.4f}")
        
        return final_weights

# 便捷函数
def create_trading_constraints(market_type: str = 'china_a_share') -> TradingConstraints:
    """
    创建预定义的交易约束检查器
    
    Parameters
    ----------
    market_type : str
        市场类型，目前支持：
        - 'china_a_share': 中国A股市场
        
    Returns
    -------
    TradingConstraints
        交易约束检查器实例
    """
    if market_type == 'china_a_share':
        return TradingConstraints(
            main_board_limit=0.10,    # 主板10%
            growth_board_limit=0.20,  # 创业板20%  
            star_board_limit=0.20     # 科创板20%
        )
    else:
        raise ValueError(f"不支持的市场类型: {market_type}")