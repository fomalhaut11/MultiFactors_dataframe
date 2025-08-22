"""
收益率计算器

计算各种类型的股票收益率
保持与原始实现完全一致的算法
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal
from pathlib import Path

from .base_processor import BaseDataProcessor


class ReturnCalculator(BaseDataProcessor):
    """收益率计算器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化收益率计算器
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        
    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        required_params = ['price_df', 'date_series']
        for param in required_params:
            if param not in kwargs:
                self.logger.error(f"缺少必需参数: {param}")
                return False
        return True
        
    def process(self, **kwargs):
        """收益率计算器不使用通用process方法"""
        raise NotImplementedError("请使用具体的计算方法")
        
    def calculate_log_return(self, 
                           price_df: pd.DataFrame,
                           date_series: pd.DatetimeIndex,
                           return_type: Literal["o2o", "c2c", "vwap", "GapReturn"] = "o2o",
                           input_type: str = "unadjusted") -> pd.DataFrame:
        """
        计算对数收益率
        完全保持原始算法不变
        
        Args:
            price_df: 价格数据DataFrame
            date_series: 日期序列
            return_type: 收益率类型
            input_type: 输入类型
            
        Returns:
            对数收益率DataFrame
        """
        next_log_return = []
        
        # 返回的收益率数据的时间戳为开仓时间
        if input_type == "unadjusted":
            for i in range(len(date_series) - 1):
                
                if return_type == "o2o":
                    openprice = (
                        price_df.loc[date_series[i], :]["o"]
                        * price_df.loc[date_series[i], :]["adjfactor"]
                    )
                    closeprice = (
                        price_df.loc[date_series[i + 1], :]["o"]
                        * price_df.loc[date_series[i + 1], :]["adjfactor"]
                    )
                elif return_type == "c2c":
                    openprice = (
                        price_df.loc[date_series[i], :]["c"]
                        * price_df.loc[date_series[i], :]["adjfactor"]
                    )
                    closeprice = (
                        price_df.loc[date_series[i + 1], :]["c"]
                        * price_df.loc[date_series[i + 1], :]["adjfactor"]
                    )
                elif return_type == "vwap":
                    openprice = (
                        price_df.loc[date_series[i], :]["vwap"]
                        * price_df.loc[date_series[i], :]["adjfactor"]
                    )
                    closeprice = (
                        price_df.loc[date_series[i + 1], :]["vwap"]
                        * price_df.loc[date_series[i + 1], :]["adjfactor"]
                    )
                elif return_type == "GapReturn":
                    if i == 0:
                        continue
                    closeprice = (
                        price_df.loc[date_series[i+1], :]["o"]
                        * price_df.loc[date_series[i+1], :]["adjfactor"]
                    )
                    openprice = (
                        price_df.loc[date_series[i], :]["c"]
                        * price_df.loc[date_series[i], :]["adjfactor"]
                    )
                else:
                    raise ValueError(f"不支持的收益率类型: {return_type}")
                    
                logreturn = np.log(closeprice / openprice)
                logreturn.name = date_series[i]
                
                next_log_return.append(logreturn)
                
            next_log_return = pd.concat(next_log_return, axis=1)
            next_log_return = next_log_return.unstack().to_frame()
            next_log_return.columns = ["LogReturn"]
            next_log_return.index.names = ["TradingDates", "StockCodes"]
            return next_log_return
            
        elif input_type == "adjusted":
            raise NotImplementedError("不支持adjusted")
        else:
            raise ValueError(f"不支持的输入类型: {input_type}")
            
    def calculate_n_days_return(self, 
                              log_return_daily: pd.DataFrame, 
                              lag: int = 20) -> pd.DataFrame:
        """
        计算N天滚动收益率
        完全保持原始算法不变
        
        Args:
            log_return_daily: 日收益率数据
            lag: 滚动天数
            
        Returns:
            N天滚动收益率
        """
        log_return_daily = log_return_daily.sort_index(level=0)
        log_return_daily_reversed = log_return_daily.iloc[::-1]
        rolling_sum = log_return_daily_reversed.groupby(
            level='StockCodes'
        ).rolling(window=lag).sum().reset_index(level=0, drop=True)
        rolling_sum = rolling_sum.iloc[::-1]
        rolling_sum = rolling_sum.sort_index(level=0)
        return rolling_sum
        
    def calculate_return_after_release(self, 
                                     log_return_daily: pd.DataFrame,
                                     released_dates_df: pd.DataFrame,
                                     lag: int = 20) -> tuple:
        """
        计算财报发布后收益率
        完全保持原始算法不变
        
        Args:
            log_return_daily: 日收益率数据
            released_dates_df: 财报发布日期数据
            lag: 计算天数
            
        Returns:
            (股票收益率, 超额收益率)
        """
        # 计算市场收益率
        market_return = log_return_daily.groupby("TradingDates").mean()
        
        # 处理财报发布日期数据
        released_dates_df = released_dates_df.swaplevel(0, 1, axis=0).sort_index(level=0)
        released_dates_df = released_dates_df.loc[
            released_dates_df.index.get_level_values(0) >= pd.Timestamp("2016-01-01")
        ]
        released_dates_df_unstackdata = released_dates_df.unstack(level=0)
        log_return_unstack = log_return_daily.unstack(level=0)
        data1 = released_dates_df_unstackdata.join(log_return_unstack, how="left")
        
        # 初始化输出数组
        outputnumpy = np.zeros(np.shape(data1['ReleasedDates']))
        outputnumpy1 = np.zeros(np.shape(data1['ReleasedDates']))
        
        # 计算收益率
        i = 0
        for stock_code, data2 in data1.iterrows():
            returndata = data2['LogReturn']
            released_dates = data2["ReleasedDates"].to_frame()
            j = 0
            for report_date, released_date in released_dates.iterrows():
                mask = returndata.index.get_level_values(0) >= released_date.values[0]
                filtered_data = returndata[mask].head(lag).sum()
                outputnumpy[i, j] = filtered_data
                mask2 = market_return.index.get_level_values(0) >= released_date.values[0]
                filtered_data2 = market_return[mask2].head(lag).sum()
                outputnumpy1[i, j] = filtered_data - filtered_data2
                j += 1
            i += 1
            
        # 创建输出DataFrame
        outputdf = pd.DataFrame(
            outputnumpy, 
            index=data1.index, 
            columns=data1['ReleasedDates'].columns
        )
        alphareturn_df1 = pd.DataFrame(
            outputnumpy1, 
            index=data1.index, 
            columns=data1['ReleasedDates'].columns
        )
        
        return outputdf, alphareturn_df1