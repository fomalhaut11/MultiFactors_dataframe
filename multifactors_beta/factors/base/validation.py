"""
数据验证和错误处理模块
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """验证错误异常类"""
    pass


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_dataframe_structure(df: pd.DataFrame, 
                                   required_columns: List[str],
                                   required_index_levels: Optional[List[str]] = None,
                                   min_rows: int = 1) -> None:
        """
        验证DataFrame的结构
        
        Parameters:
        -----------
        df : DataFrame to validate
        required_columns : 必需的列名
        required_index_levels : 必需的索引层级名称
        min_rows : 最小行数
        
        Raises:
        -------
        ValidationError : 如果验证失败
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(df)}")
        
        if len(df) < min_rows:
            raise ValidationError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        
        # 检查必需的列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # 检查索引层级
        if required_index_levels:
            if not isinstance(df.index, pd.MultiIndex):
                raise ValidationError("Expected MultiIndex, got regular Index")
            
            missing_levels = [level for level in required_index_levels 
                            if level not in df.index.names]
            if missing_levels:
                raise ValidationError(f"Missing required index levels: {missing_levels}")
    
    @staticmethod
    def validate_series_structure(series: pd.Series,
                                 required_index_levels: Optional[List[str]] = None,
                                 min_length: int = 1) -> None:
        """
        验证Series的结构
        
        Parameters:
        -----------
        series : Series to validate
        required_index_levels : 必需的索引层级名称
        min_length : 最小长度
        
        Raises:
        -------
        ValidationError : 如果验证失败
        """
        if not isinstance(series, pd.Series):
            raise ValidationError(f"Expected Series, got {type(series)}")
        
        if len(series) < min_length:
            raise ValidationError(f"Series has {len(series)} elements, minimum required: {min_length}")
        
        # 检查索引层级
        if required_index_levels:
            if not isinstance(series.index, pd.MultiIndex):
                raise ValidationError("Expected MultiIndex, got regular Index")
            
            missing_levels = [level for level in required_index_levels 
                            if level not in series.index.names]
            if missing_levels:
                raise ValidationError(f"Missing required index levels: {missing_levels}")
    
    @staticmethod
    def validate_financial_data(financial_data: pd.DataFrame,
                              required_columns: List[str]) -> None:
        """
        验证财务数据的格式和内容
        
        Parameters:
        -----------
        financial_data : 财务数据
        required_columns : 必需的列名
        
        Raises:
        -------
        ValidationError : 如果验证失败
        """
        # 基础结构验证
        DataValidator.validate_dataframe_structure(
            financial_data,
            required_columns=required_columns,
            required_index_levels=['ReportDates', 'StockCodes'],
            min_rows=1
        )
        
        # 检查数据类型
        numeric_columns = [col for col in required_columns if col not in ['d_quarter']]
        for col in numeric_columns:
            if col in financial_data.columns:
                if not pd.api.types.is_numeric_dtype(financial_data[col]):
                    logger.warning(f"Column {col} is not numeric, attempting conversion")
                    try:
                        financial_data[col] = pd.to_numeric(financial_data[col], errors='coerce')
                    except Exception as e:
                        raise ValidationError(f"Cannot convert {col} to numeric: {e}")
        
        # 检查季度字段
        if 'd_quarter' in required_columns and 'd_quarter' in financial_data.columns:
            valid_quarters = {1, 2, 3, 4}
            invalid_quarters = set(financial_data['d_quarter'].dropna().unique()) - valid_quarters
            if invalid_quarters:
                raise ValidationError(f"Invalid quarter values found: {invalid_quarters}")
        
        # 检查数据完整性
        total_nulls = financial_data[numeric_columns].isnull().sum().sum()
        total_elements = len(financial_data) * len(numeric_columns)
        null_ratio = total_nulls / total_elements if total_elements > 0 else 0
        
        if null_ratio > 0.8:
            logger.warning(f"High null ratio in financial data: {null_ratio:.2%}")
    
    @staticmethod
    def validate_market_cap_data(market_cap: pd.Series) -> None:
        """
        验证市值数据
        
        Parameters:
        -----------
        market_cap : 市值数据
        
        Raises:
        -------
        ValidationError : 如果验证失败
        """
        DataValidator.validate_series_structure(
            market_cap,
            required_index_levels=['TradingDates', 'StockCodes'],
            min_length=1
        )
        
        # 检查数值范围
        if not pd.api.types.is_numeric_dtype(market_cap):
            raise ValidationError("Market cap data must be numeric")
        
        negative_values = (market_cap < 0).sum()
        if negative_values > 0:
            logger.warning(f"Found {negative_values} negative market cap values")
        
        zero_values = (market_cap == 0).sum()
        if zero_values > 0:
            logger.warning(f"Found {zero_values} zero market cap values")
    
    @staticmethod
    def validate_date_data(dates: pd.DatetimeIndex,
                          name: str = "dates") -> None:
        """
        验证日期数据
        
        Parameters:
        -----------
        dates : 日期索引
        name : 数据名称（用于错误消息）
        
        Raises:
        -------
        ValidationError : 如果验证失败
        """
        if not isinstance(dates, pd.DatetimeIndex):
            raise ValidationError(f"{name} must be DatetimeIndex, got {type(dates)}")
        
        if len(dates) == 0:
            raise ValidationError(f"{name} cannot be empty")
        
        # 检查排序
        if not dates.is_monotonic_increasing:
            logger.warning(f"{name} is not sorted in ascending order")
        
        # 检查重复
        duplicates = dates.duplicated().sum()
        if duplicates > 0:
            raise ValidationError(f"Found {duplicates} duplicate dates in {name}")


def validate_inputs(financial_columns: Optional[List[str]] = None,
                   market_cap_required: bool = False,
                   release_dates_required: bool = False):
    """
    输入验证装饰器
    
    Parameters:
    -----------
    financial_columns : 财务数据必需的列名
    market_cap_required : 是否需要市值数据
    release_dates_required : 是否需要发布日期数据
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # 验证财务数据
                if 'financial_data' in kwargs or len(args) > 0:
                    financial_data = kwargs.get('financial_data', args[0] if args else None)
                    if financial_data is not None and financial_columns:
                        DataValidator.validate_financial_data(financial_data, financial_columns)
                
                # 验证市值数据
                if market_cap_required:
                    market_cap = kwargs.get('market_cap', args[1] if len(args) > 1 else None)
                    if market_cap is not None:
                        DataValidator.validate_market_cap_data(market_cap)
                
                # 验证发布日期数据
                if release_dates_required:
                    release_dates = kwargs.get('release_dates')
                    if release_dates is not None:
                        DataValidator.validate_dataframe_structure(
                            release_dates,
                            required_columns=['ReleasedDates'],
                            required_index_levels=['ReportDates', 'StockCodes']
                        )
                
                return func(self, *args, **kwargs)
                
            except ValidationError as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise ValidationError(f"Calculation failed: {e}") from e
        
        return wrapper
    return decorator


class ErrorHandler:
    """错误处理工具类"""
    
    @staticmethod
    def safe_division(numerator: Union[pd.Series, pd.DataFrame],
                     denominator: Union[pd.Series, pd.DataFrame],
                     fill_value: float = np.nan,
                     handle_inf: bool = True) -> Union[pd.Series, pd.DataFrame]:
        """
        安全的除法运算
        
        Parameters:
        -----------
        numerator : 分子
        denominator : 分母
        fill_value : 当分母为0时的填充值
        handle_inf : 是否处理无穷大值
        
        Returns:
        --------
        除法结果
        """
        # 将0替换为NaN，避免除零警告
        denominator_safe = denominator.replace(0, np.nan)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator_safe
        
        # 处理无穷大和NaN
        if handle_inf:
            result = result.replace([np.inf, -np.inf], fill_value)
        
        return result
    
    @staticmethod
    def handle_missing_data(data: Union[pd.Series, pd.DataFrame],
                          method: str = 'forward_fill',
                          max_consecutive: int = 5) -> Union[pd.Series, pd.DataFrame]:
        """
        处理缺失数据
        
        Parameters:
        -----------
        data : 输入数据
        method : 处理方法 ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        max_consecutive : 最大连续缺失值个数
        
        Returns:
        --------
        处理后的数据
        """
        if method == 'forward_fill':
            return data.fillna(method='ffill', limit=max_consecutive)
        elif method == 'backward_fill':
            return data.fillna(method='bfill', limit=max_consecutive)
        elif method == 'interpolate':
            return data.interpolate(limit=max_consecutive)
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    @staticmethod
    def detect_outliers(data: pd.Series,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.Series:
        """
        检测异常值
        
        Parameters:
        -----------
        data : 输入数据
        method : 检测方法 ('iqr', 'zscore', 'modified_zscore')
        threshold : 阈值
        
        Returns:
        --------
        布尔序列，True表示异常值
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    @staticmethod
    def log_data_quality_report(data: Union[pd.Series, pd.DataFrame],
                              name: str) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Parameters:
        -----------
        data : 输入数据
        name : 数据名称
        
        Returns:
        --------
        数据质量报告字典
        """
        report = {
            'name': name,
            'shape': data.shape,
            'dtype': str(data.dtype) if isinstance(data, pd.Series) else data.dtypes.to_dict(),
            'total_elements': data.size,
            'missing_count': data.isnull().sum() if isinstance(data, pd.Series) else data.isnull().sum().sum(),
            'missing_ratio': data.isnull().mean() if isinstance(data, pd.Series) else data.isnull().mean().mean(),
        }
        
        if isinstance(data, (pd.Series, pd.DataFrame)) and data.select_dtypes(include=[np.number]).size > 0:
            numeric_data = data.select_dtypes(include=[np.number])
            if isinstance(numeric_data, pd.DataFrame) and not numeric_data.empty:
                numeric_data = numeric_data.stack()
            elif isinstance(numeric_data, pd.Series):
                pass
            else:
                numeric_data = data if pd.api.types.is_numeric_dtype(data) else None
            
            if numeric_data is not None and len(numeric_data) > 0:
                report.update({
                    'mean': numeric_data.mean(),
                    'std': numeric_data.std(),
                    'min': numeric_data.min(),
                    'max': numeric_data.max(),
                    'zero_count': (numeric_data == 0).sum(),
                    'negative_count': (numeric_data < 0).sum(),
                    'infinite_count': np.isinf(numeric_data).sum(),
                })
        
        logger.info(f"Data Quality Report for {name}:")
        for key, value in report.items():
            logger.info(f"  {key}: {value}")
        
        return report