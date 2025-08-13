"""
时间序列处理器 - 处理财报数据的时间序列变换
"""
import pandas as pd
import numpy as np
from typing import Union, Literal, Optional, Tuple
import logging
from ..config import time_series_config, get_quarter_end_date

logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """时间序列处理器，用于财报数据的各种时间序列变换"""
    
    @staticmethod
    def get_report_period_date(year: int, quarter: int) -> pd.Timestamp:
        """
        获取报告期结束日期（使用配置）
        
        Parameters:
        -----------
        year : 年份
        quarter : 季度
        
        Returns:
        --------
        报告期结束日期
        """
        return get_quarter_end_date(year, quarter)
    
    @staticmethod
    def calculate_ttm(data: pd.DataFrame, 
                     quarter_col: str = 'd_quarter') -> pd.DataFrame:
        """
        计算TTM（Trailing Twelve Months）数据
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        quarter_col : 季度标识列名
        
        Returns:
        --------
        TTM数据
        """
        # 获取数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(quarter_col, errors='ignore')
        
        def _calc_ttm(stock_data):
            """计算单个股票的TTM"""
            result = pd.DataFrame(index=stock_data.index, 
                                columns=numeric_cols, 
                                dtype=float)
            
            for idx in range(len(stock_data)):
                if idx < 4:
                    continue
                    
                quarter = stock_data[quarter_col].iloc[idx]
                
                if quarter == 1:
                    # Q1: 当前Q1 + 去年Q4 - 去年Q1
                    result.iloc[idx] = (
                        stock_data.iloc[idx][numeric_cols] + 
                        stock_data.iloc[idx-1][numeric_cols] - 
                        stock_data.iloc[idx-4][numeric_cols]
                    )
                elif quarter == 2:
                    # Q2: 当前Q2 + 去年Q4 - 去年Q2
                    result.iloc[idx] = (
                        stock_data.iloc[idx][numeric_cols] + 
                        stock_data.iloc[idx-2][numeric_cols] - 
                        stock_data.iloc[idx-4][numeric_cols]
                    )
                elif quarter == 3:
                    # Q3: 当前Q3 + 去年Q4 - 去年Q3
                    result.iloc[idx] = (
                        stock_data.iloc[idx][numeric_cols] + 
                        stock_data.iloc[idx-3][numeric_cols] - 
                        stock_data.iloc[idx-4][numeric_cols]
                    )
                else:
                    # Q4: 直接使用当年数据
                    result.iloc[idx] = stock_data.iloc[idx][numeric_cols]
                    
            return result
        
        # 按股票分组计算
        ttm_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_ttm)
            
        return ttm_data
    
    @staticmethod
    def calculate_yoy(data: pd.DataFrame,
                     numeric_only: bool = True) -> pd.DataFrame:
        """
        计算同比增长率（Year-over-Year）
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        numeric_only : 是否只处理数值列
        
        Returns:
        --------
        同比增长率
        """
        if numeric_only:
            data = data.select_dtypes(include=[np.number])
            
        def _calc_yoy(stock_data):
            """计算单个股票的同比"""
            # 向前移动4个季度（一年）
            last_year = stock_data.shift(4)
            
            # 计算同比增长率
            yoy = (stock_data - last_year) / last_year.replace(0, np.nan)
            
            return yoy
        
        yoy_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_yoy)
            
        return yoy_data
    
    @staticmethod
    def calculate_qoq(data: pd.DataFrame,
                     quarter_col: str = 'd_quarter',
                     numeric_only: bool = True) -> pd.DataFrame:
        """
        计算环比增长率（Quarter-over-Quarter）
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        quarter_col : 季度标识列名
        numeric_only : 是否只处理数值列
        
        Returns:
        --------
        环比增长率
        """
        # 获取数值列
        if numeric_only:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop(quarter_col, errors='ignore')
        else:
            numeric_cols = data.columns.drop(quarter_col, errors='ignore')
            
        def _calc_qoq(stock_data):
            """计算单个股票的环比"""
            result = pd.DataFrame(index=stock_data.index, 
                                columns=numeric_cols, 
                                dtype=float)
            
            for idx in range(2, len(stock_data)):
                quarter = stock_data[quarter_col].iloc[idx]
                
                if quarter == 1:
                    # Q1 vs Q4
                    current_q = stock_data.iloc[idx][numeric_cols]
                    last_q = stock_data.iloc[idx-1][numeric_cols] - stock_data.iloc[idx-2][numeric_cols]
                elif quarter == 2:
                    # Q2 vs Q1
                    current_q = stock_data.iloc[idx][numeric_cols] - stock_data.iloc[idx-1][numeric_cols]
                    last_q = stock_data.iloc[idx-1][numeric_cols]
                elif quarter == 3:
                    # Q3 vs Q2
                    current_q = stock_data.iloc[idx][numeric_cols] - stock_data.iloc[idx-1][numeric_cols]
                    last_q = stock_data.iloc[idx-1][numeric_cols] - stock_data.iloc[idx-2][numeric_cols]
                else:
                    # Q4 vs Q3
                    current_q = stock_data.iloc[idx][numeric_cols] - stock_data.iloc[idx-1][numeric_cols]
                    last_q = stock_data.iloc[idx-1][numeric_cols] - stock_data.iloc[idx-2][numeric_cols]
                
                # 计算环比
                result.iloc[idx] = (current_q - last_q) / last_q.replace(0, np.nan)
                
            return result
        
        qoq_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_qoq)
            
        return qoq_data
    
    @staticmethod
    def calculate_zscores_timeseries(data: pd.DataFrame,
                          window: int = 12,
                          min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算Z-Score标准化
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        window : 滚动窗口大小
        min_periods : 最小有效期数
        
        Returns:
        --------
        Z-Score标准化数据
        """
        if min_periods is None:
            min_periods = max(5, window // 2)
        
        def _calc_zscore(stock_data):
            """计算单个股票的Z-Score"""
            rolling_mean = stock_data.rolling(window=window, min_periods=min_periods).mean()
            rolling_std = stock_data.rolling(window=window, min_periods=min_periods).std()
            
            zscore = (stock_data - rolling_mean) / rolling_std.replace(0, np.nan)
            
            return zscore
        
        zscore_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_zscore)
            
        return zscore_data
    
    @staticmethod
    def calculate_rank_timeseries(data: pd.DataFrame,
                          window: int = 12,
                          min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算滚动排名（Rank）
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        window : 滚动窗口大小
        min_periods : 最小有效期数
        
        Returns:
        --------
        滚动排名数据
        """
        if min_periods is None:
            min_periods = max(5, window // 2)
        
        def _calc_rank(stock_data):
            """计算单个股票的滚动排名"""
            rank = stock_data.rolling(window=window, min_periods=min_periods).apply(
                lambda x: pd.Series(x).rank().iloc[-1], raw=False)
            
            return rank
        
        rank_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_rank)
            
        return rank_data
    
    def calculate_moving_average(self, 
                                 data: pd.DataFrame,
                                 window: int = 12,
                                 min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算移动平均（Moving Average）
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        window : 滚动窗口大小
        min_periods : 最小有效期数
        
        Returns:
        --------
        移动平均数据
        """
        if min_periods is None:
            min_periods = max(5, window // 2)
        
        def _calc_ma(stock_data):
            """计算单个股票的移动平均"""
            ma = stock_data.rolling(window=window, min_periods=min_periods).mean()
            return ma
        
        ma_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_ma)
            
        return ma_data
    
    @staticmethod
    def calculate_single_quarter(data: pd.DataFrame,
                               quarter_col: str = 'd_quarter') -> pd.DataFrame:
        """
        计算单季度数据（累计数据转单季）
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        quarter_col : 季度标识列名
        
        Returns:
        --------
        单季度数据
        """
        # 获取数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(quarter_col, errors='ignore')
        
        def _calc_single_quarter(stock_data):
            """计算单个股票的单季数据"""
            result = pd.DataFrame(index=stock_data.index, 
                                columns=numeric_cols, 
                                dtype=float)
            
            for idx in range(len(stock_data)):
                if idx == 0:
                    continue
                    
                quarter = stock_data[quarter_col].iloc[idx]
                
                if quarter == 1:
                    # Q1：直接使用
                    result.iloc[idx] = stock_data.iloc[idx][numeric_cols]
                else:
                    # 其他季度：当季 - 上季
                    result.iloc[idx] = (
                        stock_data.iloc[idx][numeric_cols] - 
                        stock_data.iloc[idx-1][numeric_cols]
                    )
                    
            return result
        
        single_q_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_single_quarter)
            
        return single_q_data
    
    @staticmethod
    def calculate_avg(data: pd.DataFrame,
                     numeric_only: bool = True) -> pd.DataFrame:
        """
        计算期末期初平均值
        
        Parameters:
        -----------
        data : DataFrame，多级索引(ReportDates, StockCodes)
        numeric_only : 是否只处理数值列
        
        Returns:
        --------
        平均值
        """
        if numeric_only:
            data = data.select_dtypes(include=[np.number])
            
        def _calc_avg(stock_data):
            """计算单个股票的平均值"""
            # 上一期数据
            last_period = stock_data.shift(1)
            
            # 计算平均值
            avg = (stock_data + last_period) / 2
            
            return avg
        
        avg_data = data.groupby(level='StockCodes', group_keys=False).apply(_calc_avg)
            
        return avg_data
    
    @staticmethod
    def expand_to_daily(factor_data: pd.DataFrame,
                       release_dates: pd.DataFrame,
                       trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        将财报数据扩展到日频
        
        Parameters:
        -----------
        factor_data : 财报因子数据，索引为(ReportDates, StockCodes)
        release_dates : 财报发布日期，包含'ReleasedDates'列
        trading_dates : 交易日序列
        
        Returns:
        --------
        日频因子数据，索引为(TradingDates, StockCodes)
        """
        # 合并财报数据和发布日期
        factor_with_release = factor_data.join(release_dates[['ReleasedDates']])
        
        # 按股票分组处理
        def _expand_stock(stock_data):
            """将单个股票的财报数据扩展到日频"""
            # 创建日频DataFrame
            daily_data = pd.DataFrame(index=trading_dates)
            
            # 首先按发布日期排序，然后按报表截止日期排序
            # 这样可以确保同一天发布的多个财报，最新的报表期间优先
            stock_data = stock_data.reset_index()
            stock_data = stock_data.sort_values(['ReleasedDates', 'ReportDates'])
            
            # 向前填充
            for _, row in stock_data.iterrows():
                release_date = row['ReleasedDates']
                report_date = row['ReportDates']
                
                # 找到发布日期之后的所有交易日
                mask = trading_dates >= release_date
                valid_dates = trading_dates[mask]
                
                # 填充因子值
                for col in factor_data.columns:
                    if col not in ['ReleasedDates', 'ReportDates']:
                        value = row[col]
                        # 如果是Series，取第一个值
                        if isinstance(value, pd.Series):
                            value = value.iloc[0] if len(value) > 0 else np.nan
                        daily_data.loc[valid_dates, col] = value
                
                # 记录最新使用的报表截止日期（内部使用）
                daily_data.loc[valid_dates, '_LatestReportDate'] = report_date
                        
            return daily_data
        
        # 按股票代码分组
        results = []
        for stock_code, stock_data in factor_with_release.groupby(level='StockCodes'):
            daily_data = _expand_stock(stock_data)
            daily_data['StockCodes'] = stock_code
            results.append(daily_data)
        
        # 合并所有股票的数据
        expanded = pd.concat(results, ignore_index=False)
        
        # 设置MultiIndex
        expanded = expanded.reset_index()
        expanded = expanded.rename(columns={'index': 'TradingDates'})
        expanded = expanded.set_index(['TradingDates', 'StockCodes'])
        
        # 删除内部使用的列（保留它会破坏向后兼容性）
        internal_cols = [col for col in expanded.columns if col.startswith('_')]
        if internal_cols:
            expanded = expanded.drop(columns=internal_cols)
        
        # 如果只有一列，返回Series
        if len(expanded.columns) == 1:
            return expanded.iloc[:, 0]
        else:
            return expanded
    
    @staticmethod
    def expand_to_daily_with_metadata(factor_data: pd.DataFrame,
                                    release_dates: pd.DataFrame,
                                    trading_dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        将财报数据扩展到日频，同时返回元数据
        
        与 expand_to_daily 相同，但额外返回报表日期映射信息
        
        Returns:
        --------
        (expanded_data, metadata) : 扩展后的数据和元数据
            metadata 包含 LatestReportDate 列，显示每个交易日使用的报表期
        """
        # 合并财报数据和发布日期
        factor_with_release = factor_data.join(release_dates[['ReleasedDates']])
        
        # 按股票分组处理
        def _expand_stock(stock_data):
            """将单个股票的财报数据扩展到日频"""
            # 创建日频DataFrame
            daily_data = pd.DataFrame(index=trading_dates)
            metadata = pd.DataFrame(index=trading_dates)
            
            # 首先按发布日期排序，然后按报表截止日期排序
            stock_data = stock_data.reset_index()
            stock_data = stock_data.sort_values(['ReleasedDates', 'ReportDates'])
            
            # 向前填充
            for _, row in stock_data.iterrows():
                release_date = row['ReleasedDates']
                report_date = row['ReportDates']
                
                # 找到发布日期之后的所有交易日
                mask = trading_dates >= release_date
                
                # 填充因子值
                for col in factor_data.columns:
                    if col not in ['ReleasedDates', 'ReportDates']:
                        daily_data.loc[mask, col] = row[col]
                
                # 记录报表日期
                metadata.loc[mask, 'LatestReportDate'] = report_date
                        
            return daily_data, metadata
        
        # 按股票代码分组
        results = []
        metadata_list = []
        
        for stock_code, stock_data in factor_with_release.groupby(level='StockCodes'):
            daily_data, metadata = _expand_stock(stock_data)
            daily_data['StockCodes'] = stock_code
            metadata['StockCodes'] = stock_code
            results.append(daily_data)
            metadata_list.append(metadata)
        
        # 合并数据
        expanded = pd.concat(results, ignore_index=False)
        metadata_df = pd.concat(metadata_list, ignore_index=False)
        
        # 设置MultiIndex
        for df in [expanded, metadata_df]:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'TradingDates'}, inplace=True)
            df.set_index(['TradingDates', 'StockCodes'], inplace=True)
        
        return expanded, metadata_df