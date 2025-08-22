"""
数据处理混入类 - 提供通用的数据处理方法
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, Tuple
import logging
from .validation import DataValidator, ValidationError, ErrorHandler
from ..config import factor_config, get_column_name

logger = logging.getLogger(__name__)


class DataProcessingMixin:
    """数据处理混入类，提供通用的数据处理方法"""
    
    # 从配置中获取设置
    @property
    def USE_OPTIMIZED_PROCESSOR(self):
        return factor_config.PERFORMANCE_CONFIG['use_optimized_processor']
    
    @property 
    def MEMORY_EFFICIENT_THRESHOLD(self):
        return factor_config.PERFORMANCE_CONFIG['memory_efficient_threshold']
    
    def _expand_and_align_data(self,
                              factor_data: pd.DataFrame,
                              market_cap: Optional[pd.Series] = None,
                              release_dates: Optional[pd.DataFrame] = None,
                              trading_dates: Optional[pd.DatetimeIndex] = None,
                              use_market_cap_lag: bool = True) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        通用的数据扩展和对齐方法
        
        Parameters:
        -----------
        factor_data : 因子数据
        market_cap : 市值数据（可选）
        release_dates : 发布日期数据（可选）
        trading_dates : 交易日序列（可选）
        use_market_cap_lag : 是否使用滞后一期的市值
        
        Returns:
        --------
        如果提供了市值数据，返回 (factor_aligned, market_cap_aligned)
        否则返回 factor_data
        """
        from ..base.time_series_processor import TimeSeriesProcessor
        
        # 扩展到日频（如果需要）
        if release_dates is not None and trading_dates is not None:
            factor_data = self._expand_to_daily_smart(
                factor_data, 
                release_dates,
                trading_dates
            )
        
        # 如果没有市值数据，直接返回
        if market_cap is None:
            return factor_data
            
        # 处理市值滞后
        if use_market_cap_lag:
            market_cap_processed = market_cap.groupby(level='StockCodes').shift(1)
        else:
            market_cap_processed = market_cap
            
        # 对齐数据
        factor_aligned, market_cap_aligned = factor_data.align(
            market_cap_processed, 
            join='inner'
        )
        
        return factor_aligned, market_cap_aligned
    
    def _calculate_ratio_factor(self,
                               numerator: pd.DataFrame,
                               market_cap: pd.Series,
                               release_dates: Optional[pd.DataFrame] = None,
                               trading_dates: Optional[pd.DatetimeIndex] = None,
                               use_market_cap_lag: bool = True) -> pd.Series:
        """
        计算比率因子的通用方法
        
        Parameters:
        -----------
        numerator : 分子数据
        market_cap : 市值数据（分母）
        release_dates : 发布日期
        trading_dates : 交易日序列
        use_market_cap_lag : 是否使用滞后市值
        
        Returns:
        --------
        比率因子
        """
        # 扩展和对齐数据
        numerator_aligned, market_cap_aligned = self._expand_and_align_data(
            numerator,
            market_cap,
            release_dates,
            trading_dates,
            use_market_cap_lag
        )
        
        # 计算比率
        ratio = numerator_aligned / market_cap_aligned
        
        # 预处理
        if hasattr(self, 'preprocess'):
            ratio = self.preprocess(ratio)
            
        return ratio
    
    def _validate_required_columns(self, 
                                  data: pd.DataFrame, 
                                  required_columns: list) -> None:
        """
        验证必需的列是否存在（使用新的验证器）
        
        Parameters:
        -----------
        data : 待验证的DataFrame
        required_columns : 必需的列名列表
        
        Raises:
        -------
        ValidationError : 如果缺少必需的列
        """
        try:
            DataValidator.validate_financial_data(data, required_columns)
        except ValidationError as e:
            logger.error(f"财务数据验证失败: {e}")
            raise
    
    def _safe_division(self, 
                      numerator: pd.Series, 
                      denominator: pd.Series,
                      fill_value: float = np.nan) -> pd.Series:
        """
        安全的除法 （使用新的错误处理器）
        
        Parameters:
        -----------
        numerator : 分子
        denominator : 分母
        fill_value : 当分母为0时的填充值
        
        Returns:
        --------
        除法结果
        """
        return ErrorHandler.safe_division(numerator, denominator, fill_value)
    
    def _log_data_info(self, 
                      data: Union[pd.DataFrame, pd.Series], 
                      name: str) -> None:
        """
        记录数据信息，用于调试
        
        Parameters:
        -----------
        data : 数据
        name : 数据名称
        """
        if isinstance(data, pd.DataFrame):
            logger.debug(f"{name} shape: {data.shape}, columns: {list(data.columns)}")
        else:
            logger.debug(f"{name} shape: {data.shape}")
            
        # 记录数据统计信息
        if hasattr(data, 'describe'):
            logger.debug(f"{name} stats:\n{data.describe()}")
    
    def _expand_to_daily_smart(self,
                              factor_data: pd.DataFrame,
                              release_dates: pd.DataFrame,
                              trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        智能选择最优的日频扩展方法
        
        Parameters:
        -----------
        factor_data : 因子数据
        release_dates : 发布日期
        trading_dates : 交易日序列
        
        Returns:
        --------
        扩展后的日频数据
        """
        if not self.USE_OPTIMIZED_PROCESSOR:
            # 使用原始方法
            from ..base.time_series_processor import TimeSeriesProcessor
            return TimeSeriesProcessor.expand_to_daily(
                factor_data, release_dates, trading_dates
            )
        
        # 检查数据规模
        unique_stocks = factor_data.index.get_level_values('StockCodes').nunique()
        
        if unique_stocks > self.MEMORY_EFFICIENT_THRESHOLD:
            # 使用内存高效方法
            from .optimized_time_series_processor import OptimizedTimeSeriesProcessor
            logger.info(f"使用内存高效模式处理 {unique_stocks} 个股票")
            return OptimizedTimeSeriesProcessor.expand_to_daily_memory_efficient(
                factor_data, release_dates, trading_dates
            )
        else:
            # 使用向量化方法
            from .optimized_time_series_processor import OptimizedTimeSeriesProcessor
            logger.debug(f"使用向量化模式处理 {unique_stocks} 个股票")
            return OptimizedTimeSeriesProcessor.expand_to_daily_vectorized(
                factor_data, release_dates, trading_dates
            )