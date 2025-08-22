"""
优化的收益率计算器

使用向量化运算和并行处理优化收益率计算性能
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, List, Tuple
from pathlib import Path
import numba
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

from .base_processor import BaseDataProcessor
from .return_calculator import ReturnCalculator
from .parallel_optimizer import ParallelOptimizer


class OptimizedReturnCalculator(ReturnCalculator):
    """优化的收益率计算器"""
    
    def __init__(self, config_path: Optional[str] = None,
                 use_parallel: bool = True,
                 n_workers: Optional[int] = None):
        """
        初始化优化的收益率计算器
        
        Args:
            config_path: 配置文件路径
            use_parallel: 是否使用并行处理
            n_workers: 并行工作数
        """
        super().__init__(config_path)
        self.use_parallel = use_parallel
        self.parallel_optimizer = ParallelOptimizer(n_workers=n_workers) if use_parallel else None
        
    def calculate_log_return_vectorized(self, 
                                      price_df: pd.DataFrame,
                                      date_series: pd.DatetimeIndex,
                                      return_type: Literal["o2o", "c2c", "vwap"] = "o2o") -> pd.DataFrame:
        """
        向量化计算对数收益率（更快的实现）
        
        Args:
            price_df: 价格数据DataFrame
            date_series: 日期序列
            return_type: 收益率类型
            
        Returns:
            对数收益率DataFrame
        """
        # 获取价格列名
        price_col = {'o2o': 'o', 'c2c': 'c', 'vwap': 'vwap'}[return_type]
        
        # 提取需要的数据
        dates = date_series[:-1]  # 开仓日期
        
        # 获取所有股票代码
        all_stocks = price_df.index.get_level_values(1).unique()
        
        # 预分配结果数组
        n_dates = len(dates)
        n_stocks = len(all_stocks)
        
        # 使用向量化操作计算收益率
        self.logger.info(f"使用向量化方法计算 {return_type} 收益率...")
        
        # 构建完整的日期-股票索引
        full_index = pd.MultiIndex.from_product(
            [dates, all_stocks],
            names=['TradingDates', 'StockCodes']
        )
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame(
            index=full_index,
            columns=['LogReturn'],
            dtype=np.float64
        )
        
        # 批量计算每个股票的收益率
        for stock in all_stocks:
            try:
                # 获取该股票的价格数据
                stock_prices = price_df.xs(stock, level=1)
                
                # 确保日期对齐
                stock_prices = stock_prices.reindex(date_series)
                
                # 计算复权价格
                adj_prices = stock_prices[price_col] * stock_prices['adjfactor']
                
                # 计算对数收益率
                log_returns = np.log(adj_prices.shift(-1) / adj_prices).iloc[:-1]
                
                # 填充结果
                for date, log_ret in zip(dates, log_returns):
                    if not pd.isna(log_ret):
                        result_df.loc[(date, stock), 'LogReturn'] = log_ret
                        
            except Exception as e:
                self.logger.debug(f"股票 {stock} 计算失败: {e}")
                continue
                
        return result_df
        
    def calculate_n_days_return_optimized(self, 
                                        log_return_daily: pd.DataFrame, 
                                        lag: int = 20) -> pd.DataFrame:
        """
        优化的N天滚动收益率计算
        
        Args:
            log_return_daily: 日收益率数据
            lag: 滚动天数
            
        Returns:
            N天滚动收益率
        """
        self.logger.info(f"使用优化方法计算 {lag} 天滚动收益率...")
        
        # 重塑数据为宽表格式（日期为行，股票为列）
        wide_returns = log_return_daily['LogReturn'].unstack(level='StockCodes')
        
        # 使用pandas的滚动窗口功能（已经优化过）
        rolling_returns = wide_returns.rolling(window=lag, min_periods=1).sum()
        
        # 转换回长表格式
        result = rolling_returns.stack()
        result.name = 'LogReturn'
        result = result.to_frame()
        
        return result
        
    def batch_calculate_returns(self, 
                              price_df: pd.DataFrame,
                              periods: Dict[str, pd.DatetimeIndex],
                              return_types: List[str],
                              rolling_windows: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """
        批量计算多种收益率
        
        Args:
            price_df: 价格数据
            periods: 期间字典 {'daily': dates, 'weekly': dates, ...}
            return_types: 收益率类型列表
            rolling_windows: 滚动窗口列表
            
        Returns:
            收益率结果字典
        """
        results = {}
        
        if self.use_parallel and self.parallel_optimizer:
            # 使用并行计算
            self.logger.info("使用并行方式批量计算收益率...")
            
            # 准备任务
            tasks = []
            for period_name, date_series in periods.items():
                for return_type in return_types:
                    task = {
                        'name': f'{period_name}_{return_type}',
                        'func': self.calculate_log_return_vectorized,
                        'args': (price_df, date_series),
                        'kwargs': {'return_type': return_type}
                    }
                    tasks.append(task)
                    
            # 并行执行
            parallel_results = self.parallel_optimizer._execute_tasks_parallel(tasks)
            results.update(parallel_results)
            
            # 计算滚动收益率
            if rolling_windows and 'daily_o2o' in results:
                daily_return = results['daily_o2o']
                rolling_tasks = []
                
                for window in rolling_windows:
                    task = {
                        'name': f'rolling_{window}d',
                        'func': self.calculate_n_days_return_optimized,
                        'args': (daily_return,),
                        'kwargs': {'lag': window}
                    }
                    rolling_tasks.append(task)
                    
                rolling_results = self.parallel_optimizer._execute_tasks_parallel(rolling_tasks)
                results.update(rolling_results)
                
        else:
            # 串行计算
            self.logger.info("使用串行方式批量计算收益率...")
            
            for period_name, date_series in periods.items():
                for return_type in return_types:
                    key = f'{period_name}_{return_type}'
                    self.logger.info(f"计算 {key}...")
                    results[key] = self.calculate_log_return_vectorized(
                        price_df, date_series, return_type
                    )
                    
            # 计算滚动收益率
            if rolling_windows and 'daily_o2o' in results:
                daily_return = results['daily_o2o']
                for window in rolling_windows:
                    key = f'rolling_{window}d'
                    self.logger.info(f"计算 {key}...")
                    results[key] = self.calculate_n_days_return_optimized(
                        daily_return, lag=window
                    )
                    
        return results
        

# Numba加速的辅助函数
@jit(nopython=True, parallel=True)
def calculate_log_returns_numba(open_prices: np.ndarray, 
                               close_prices: np.ndarray,
                               adj_factors_open: np.ndarray,
                               adj_factors_close: np.ndarray) -> np.ndarray:
    """
    使用Numba加速的对数收益率计算
    
    Args:
        open_prices: 开盘价数组
        close_prices: 收盘价数组
        adj_factors_open: 开盘复权因子
        adj_factors_close: 收盘复权因子
        
    Returns:
        对数收益率数组
    """
    n = len(open_prices)
    log_returns = np.empty(n)
    
    for i in prange(n):
        adj_open = open_prices[i] * adj_factors_open[i]
        adj_close = close_prices[i] * adj_factors_close[i]
        
        if adj_open > 0 and adj_close > 0:
            log_returns[i] = np.log(adj_close / adj_open)
        else:
            log_returns[i] = np.nan
            
    return log_returns


@jit(nopython=True, parallel=True)
def calculate_rolling_sum_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    使用Numba加速的滚动求和
    
    Args:
        data: 输入数据（2D数组，行为时间，列为股票）
        window: 窗口大小
        
    Returns:
        滚动求和结果
    """
    n_times, n_stocks = data.shape
    result = np.empty_like(data)
    
    for j in prange(n_stocks):
        for i in range(n_times):
            if i < window - 1:
                # 不足窗口大小，计算可用数据的和
                result[i, j] = np.sum(data[:i+1, j])
            else:
                # 完整窗口
                result[i, j] = np.sum(data[i-window+1:i+1, j])
                
    return result


class FastReturnCalculator:
    """超快速收益率计算器（使用Numba）"""
    
    @staticmethod
    def calculate_returns_batch(price_data: Dict[str, np.ndarray],
                              return_type: str = "o2o") -> np.ndarray:
        """
        批量计算收益率
        
        Args:
            price_data: 价格数据字典，包含 'open', 'close', 'vwap', 'adjfactor'
            return_type: 收益率类型
            
        Returns:
            收益率数组
        """
        if return_type == "o2o":
            open_prices = price_data['open'][:-1]
            close_prices = price_data['open'][1:]
            adj_open = price_data['adjfactor'][:-1]
            adj_close = price_data['adjfactor'][1:]
        elif return_type == "c2c":
            open_prices = price_data['close'][:-1]
            close_prices = price_data['close'][1:]
            adj_open = price_data['adjfactor'][:-1]
            adj_close = price_data['adjfactor'][1:]
        elif return_type == "vwap":
            open_prices = price_data['vwap'][:-1]
            close_prices = price_data['vwap'][1:]
            adj_open = price_data['adjfactor'][:-1]
            adj_close = price_data['adjfactor'][1:]
        else:
            raise ValueError(f"不支持的收益率类型: {return_type}")
            
        return calculate_log_returns_numba(
            open_prices, close_prices, adj_open, adj_close
        )