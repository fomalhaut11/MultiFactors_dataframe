"""
优化的时间序列处理器 - 提供高性能的数据处理方法
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimizedTimeSeriesProcessor:
    """优化的时间序列处理器，专注性能和内存效率"""
    
    @staticmethod
    def _get_effective_trading_date(release_date: pd.Timestamp, 
                                   trading_dates: pd.DatetimeIndex) -> pd.Timestamp:
        """
        获取财报发布日期对应的生效交易日
        
        Parameters:
        -----------
        release_date : 财报发布日期
        trading_dates : 交易日序列
        
        Returns:
        --------
        生效的交易日期：
        - 如果发布日期是交易日，返回发布日期
        - 如果发布日期是非交易日，返回下一个交易日
        - 如果没有后续交易日，返回None
        """
        # 如果发布日期是交易日，直接使用
        if release_date in trading_dates:
            return release_date
        
        # 如果是非交易日，找到下一个交易日
        future_dates = trading_dates[trading_dates > release_date]
        if len(future_dates) > 0:
            return future_dates[0]
        
        # 如果没有后续交易日，返回None
        return None
    
    @staticmethod
    def expand_to_daily_vectorized(factor_data: pd.DataFrame,
                                  release_dates: pd.DataFrame,
                                  trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        向量化的日频扩展方法，大幅提升性能
        
        Parameters:
        -----------
        factor_data : 财报因子数据，索引为(ReportDates, StockCodes)
        release_dates : 财报发布日期，包含'ReleasedDates'列
        trading_dates : 交易日序列
        
        Returns:
        --------
        日频因子数据，索引为(TradingDates, StockCodes)
        """
        logger.debug(f"开始向量化日频扩展，数据形状: {factor_data.shape}")
        
        # 合并财报数据和发布日期
        factor_with_release = factor_data.join(release_dates[['ReleasedDates']], how='inner')
        
        if factor_with_release.empty:
            logger.warning("合并后数据为空，请检查索引对齐")
            return pd.DataFrame()
        
        # 重置索引，便于处理
        factor_reset = factor_with_release.reset_index()
        
        # 按发布日期排序，确保数据的时间顺序
        factor_sorted = factor_reset.sort_values(['StockCodes', 'ReleasedDates', 'ReportDates'])
        
        # 获取所有唯一的股票代码
        stock_codes = factor_sorted['StockCodes'].unique()
        
        # 创建结果容器
        results = []
        
        # 预计算交易日期的索引映射，提高查找效率
        trading_dates_index = pd.Series(
            range(len(trading_dates)), 
            index=trading_dates
        )
        
        # 批量处理股票
        for stock_code in stock_codes:
            stock_data = factor_sorted[factor_sorted['StockCodes'] == stock_code].copy()
            
            if stock_data.empty:
                continue
                
            # 为该股票创建日频数据框架
            daily_result = pd.DataFrame(
                index=trading_dates,
                columns=factor_data.columns,
                dtype=float
            )
            
            # 使用向量化方法填充数据
            daily_result = OptimizedTimeSeriesProcessor._fill_daily_data_vectorized(
                daily_result, stock_data, trading_dates_index
            )
            
            # 添加股票代码
            daily_result['StockCodes'] = stock_code
            results.append(daily_result)
        
        if not results:
            logger.warning("没有生成任何结果数据")
            return pd.DataFrame()
        
        # 合并所有股票的数据
        expanded = pd.concat(results, ignore_index=False)
        
        # 设置MultiIndex
        expanded = expanded.reset_index()
        expanded = expanded.rename(columns={'index': 'TradingDates'})
        expanded = expanded.set_index(['TradingDates', 'StockCodes'])
        
        # 如果只有一列，返回Series
        if len(expanded.columns) == 1:
            result = expanded.iloc[:, 0]
        else:
            result = expanded
            
        logger.debug(f"完成向量化日频扩展，结果形状: {result.shape}")
        return result
    
    @staticmethod
    def _fill_daily_data_vectorized(daily_result: pd.DataFrame, 
                                   stock_data: pd.DataFrame,
                                   trading_dates_index: pd.Series) -> pd.DataFrame:
        """
        使用向量化方法填充单个股票的日频数据
        
        Parameters:
        -----------
        daily_result : 待填充的日频DataFrame
        stock_data : 单个股票的财报数据
        trading_dates_index : 交易日期索引映射
        
        Returns:
        --------
        填充后的日频数据
        """
        # 🔥 修复：正确处理非交易日发布的财报
        # 不再过滤发布日期，而是使用辅助函数找到生效交易日
        
        # 为每个发布日期创建掩码
        for _, row in stock_data.iterrows():
            release_date = row['ReleasedDates']
            
            # 获取生效交易日：发布日本身或下一个交易日
            effective_date = OptimizedTimeSeriesProcessor._get_effective_trading_date(
                release_date, trading_dates_index.index
            )
            
            if effective_date is None:
                continue  # 没有后续交易日，跳过
                
            # 找到生效日期之后的所有交易日索引
            effective_idx = trading_dates_index.get(effective_date)
            if effective_idx is None:
                continue
                
            # 使用布尔索引批量更新
            mask = trading_dates_index >= effective_idx
            valid_dates = trading_dates_index.index[mask]
            
            # 批量填充所有因子列
            for col in daily_result.columns:
                if col in row and pd.notna(row[col]):
                    daily_result.loc[valid_dates, col] = row[col]
        
        return daily_result
    
    @staticmethod
    def expand_to_daily_memory_efficient(factor_data: pd.DataFrame,
                                       release_dates: pd.DataFrame,
                                       trading_dates: pd.DatetimeIndex,
                                       chunk_size: int = 100) -> pd.DataFrame:
        """
        内存高效的日频扩展方法，适用于大数据集
        
        Parameters:
        -----------
        factor_data : 财报因子数据
        release_dates : 财报发布日期
        trading_dates : 交易日序列
        chunk_size : 批处理大小
        
        Returns:
        --------
        日频因子数据
        """
        logger.debug(f"开始内存高效日频扩展，块大小: {chunk_size}")
        
        # 合并数据
        factor_with_release = factor_data.join(release_dates[['ReleasedDates']], how='inner')
        factor_reset = factor_with_release.reset_index()
        factor_sorted = factor_reset.sort_values(['StockCodes', 'ReleasedDates', 'ReportDates'])
        
        # 获取股票代码并分块处理
        stock_codes = factor_sorted['StockCodes'].unique()
        
        results = []
        
        # 分块处理股票
        for i in range(0, len(stock_codes), chunk_size):
            chunk_stocks = stock_codes[i:i + chunk_size]
            logger.debug(f"处理股票块 {i//chunk_size + 1}/{(len(stock_codes)-1)//chunk_size + 1}")
            
            chunk_data = factor_sorted[factor_sorted['StockCodes'].isin(chunk_stocks)]
            
            # 对该块使用向量化方法
            chunk_result = OptimizedTimeSeriesProcessor._process_stock_chunk(
                chunk_data, trading_dates, factor_data.columns
            )
            
            if not chunk_result.empty:
                results.append(chunk_result)
            
            # 强制垃圾回收以释放内存
            del chunk_data
            
        if not results:
            return pd.DataFrame()
        
        # 合并结果
        final_result = pd.concat(results, ignore_index=False)
        
        # 设置索引
        if 'TradingDates' not in final_result.index.names:
            final_result = final_result.reset_index()
            final_result = final_result.rename(columns={'index': 'TradingDates'})
            final_result = final_result.set_index(['TradingDates', 'StockCodes'])
        
        logger.debug(f"完成内存高效日频扩展，结果形状: {final_result.shape}")
        return final_result
    
    @staticmethod
    def _process_stock_chunk(chunk_data: pd.DataFrame,
                           trading_dates: pd.DatetimeIndex,
                           factor_columns: pd.Index) -> pd.DataFrame:
        """
        处理单个股票块
        
        Parameters:
        -----------
        chunk_data : 股票块数据
        trading_dates : 交易日序列
        factor_columns : 因子列名
        
        Returns:
        --------
        处理后的日频数据
        """
        chunk_results = []
        
        for stock_code, stock_data in chunk_data.groupby('StockCodes'):
            # 创建该股票的日频框架
            daily_data = pd.DataFrame(
                index=trading_dates,
                columns=factor_columns,
                dtype=float
            )
            
            # 🎯 优化的填充逻辑 - 避免重复覆盖
            # 按发布日期排序，确保按时间顺序处理
            stock_data_sorted = stock_data.sort_values('ReleasedDates')
            
            for i, (_, row) in enumerate(stock_data_sorted.iterrows()):
                release_date = row['ReleasedDates']
                
                # 🔥 修复：正确处理非交易日发布的财报
                effective_date = OptimizedTimeSeriesProcessor._get_effective_trading_date(
                    release_date, trading_dates
                )
                
                if effective_date is not None:
                    # 🔥 关键改进：使用区间赋值，避免重复覆盖
                    if i < len(stock_data_sorted) - 1:
                        next_release_date = stock_data_sorted.iloc[i + 1]['ReleasedDates']
                        next_effective_date = OptimizedTimeSeriesProcessor._get_effective_trading_date(
                            next_release_date, trading_dates
                        )
                        if next_effective_date is not None:
                            mask = (trading_dates >= effective_date) & (trading_dates < next_effective_date)
                        else:
                            mask = trading_dates >= effective_date
                    else:
                        mask = trading_dates >= effective_date
                    
                    for col in factor_columns:
                        if col in row and pd.notna(row[col]):
                            daily_data.loc[mask, col] = row[col]
            
            daily_data['StockCodes'] = stock_code
            chunk_results.append(daily_data)
        
        if chunk_results:
            return pd.concat(chunk_results, ignore_index=False)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def benchmark_expand_methods(factor_data: pd.DataFrame,
                               release_dates: pd.DataFrame,
                               trading_dates: pd.DatetimeIndex) -> dict:
        """
        对不同扩展方法进行性能基准测试
        
        Returns:
        --------
        基准测试结果字典
        """
        import time
        from ..base.time_series_processor import TimeSeriesProcessor
        
        results = {}
        
        # 测试原始方法
        try:
            start_time = time.time()
            original_result = TimeSeriesProcessor.expand_to_daily(
                factor_data, release_dates, trading_dates
            )
            original_time = time.time() - start_time
            results['original'] = {
                'time': original_time,
                'shape': original_result.shape if hasattr(original_result, 'shape') else None,
                'success': True
            }
        except Exception as e:
            results['original'] = {
                'time': None,
                'shape': None,
                'success': False,
                'error': str(e)
            }
        
        # 测试向量化方法
        try:
            start_time = time.time()
            vectorized_result = OptimizedTimeSeriesProcessor.expand_to_daily_vectorized(
                factor_data, release_dates, trading_dates
            )
            vectorized_time = time.time() - start_time
            results['vectorized'] = {
                'time': vectorized_time,
                'shape': vectorized_result.shape if hasattr(vectorized_result, 'shape') else None,
                'success': True
            }
        except Exception as e:
            results['vectorized'] = {
                'time': None,
                'shape': None,
                'success': False,
                'error': str(e)
            }
        
        # 测试内存高效方法
        try:
            start_time = time.time()
            memory_result = OptimizedTimeSeriesProcessor.expand_to_daily_memory_efficient(
                factor_data, release_dates, trading_dates
            )
            memory_time = time.time() - start_time
            results['memory_efficient'] = {
                'time': memory_time,
                'shape': memory_result.shape if hasattr(memory_result, 'shape') else None,
                'success': True
            }
        except Exception as e:
            results['memory_efficient'] = {
                'time': None,
                'shape': None,
                'success': False,
                'error': str(e)
            }
        
        return results