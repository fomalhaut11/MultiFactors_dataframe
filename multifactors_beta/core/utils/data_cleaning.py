"""数据清洗和预处理工具模块"""
import pandas as pd
import numpy as np
from typing import Union, Literal
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class OutlierHandler:
    """异常值处理器"""
    
    @staticmethod
    def remove_outlier(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        method: Literal["mean", "median", "IQR"] = "mean",
        param: float = 3.0,
        threshold: float = None
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        去除异常值
        
        Parameters:
        -----------
        data : 输入数据
        method : 异常值检测方法
            - "mean": 均值标准差法
            - "median": 中位数绝对偏差法(MAD)
            - "IQR": 四分位距法
        param : 阈值参数（与threshold相同，为了兼容）
        threshold : 阈值参数（已废弃，使用param）
        
        Returns:
        --------
        处理后的数据
        """
        # 兼容处理：如果提供了threshold但没有param，使用threshold
        if threshold is not None and param == 3.0:
            param = threshold
            
        # 创建副本避免修改原数据
        if isinstance(data, np.ndarray):
            result = data.astype(float).copy()
            return OutlierHandler._remove_outlier_numpy(result, method, param)
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            result = data.copy()
            return OutlierHandler._remove_outlier_pandas(result, method, param)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @staticmethod
    def _remove_outlier_numpy(data: np.ndarray, method: str, threshold: float) -> np.ndarray:
        """处理numpy数组的异常值"""
        # 处理无穷值和NaN
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            return data
            
        finite_data = data[finite_mask]
        
        if method == "mean":
            mean_val = np.mean(finite_data)
            std_val = np.std(finite_data)
            upper_bound = mean_val + threshold * std_val
            lower_bound = mean_val - threshold * std_val
            
        elif method == "median":
            median_val = np.median(finite_data)
            mad = np.median(np.abs(finite_data - median_val))
            upper_bound = median_val + threshold * mad
            lower_bound = median_val - threshold * mad
            
        elif method == "IQR":
            q1 = np.percentile(finite_data, 25)
            q3 = np.percentile(finite_data, 75)
            iqr = q3 - q1
            upper_bound = q3 + threshold * iqr
            lower_bound = q1 - threshold * iqr
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 替换异常值
        data[~finite_mask] = np.median(finite_data)
        data[data > upper_bound] = upper_bound
        data[data < lower_bound] = lower_bound
        
        return data
    
    @staticmethod
    def _remove_outlier_pandas(
        data: Union[pd.DataFrame, pd.Series], 
        method: str, 
        threshold: float
    ) -> Union[pd.DataFrame, pd.Series]:
        """处理pandas数据的异常值"""
        if isinstance(data, pd.Series):
            return OutlierHandler._remove_outlier_series(data, method, threshold)
        else:
            return data.apply(
                lambda col: OutlierHandler._remove_outlier_series(col, method, threshold)
            )
    
    @staticmethod
    def _remove_outlier_series(series: pd.Series, method: str, threshold: float) -> pd.Series:
        """处理单个Series的异常值"""
        result = series.copy()
        finite_mask = series.notna() & ~series.isin([np.inf, -np.inf])
        
        if not finite_mask.any():
            return result
            
        finite_data = series[finite_mask]
        
        if method == "mean":
            mean_val = finite_data.mean()
            std_val = finite_data.std()
            upper_bound = mean_val + threshold * std_val
            lower_bound = mean_val - threshold * std_val
            
        elif method == "median":
            median_val = finite_data.median()
            mad = (finite_data - median_val).abs().median()
            upper_bound = median_val + threshold * mad
            lower_bound = median_val - threshold * mad
            
        elif method == "IQR":
            q1 = finite_data.quantile(0.25)
            q3 = finite_data.quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + threshold * iqr
            lower_bound = q1 - threshold * iqr
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 填充缺失值和异常值
        result.fillna(finite_data.median(), inplace=True)
        result[result > upper_bound] = upper_bound
        result[result < lower_bound] = lower_bound
        
        return result


class Normalizer:
    """数据标准化处理器"""
    
    @staticmethod
    def normalize(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        method: Literal["zscore", "minmax", "robust", "rank"] = "zscore"
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        标准化数据
        
        Parameters:
        -----------
        data : 输入数据
        method : 标准化方法
            - "zscore": Z-score标准化
            - "minmax": 最小-最大标准化
            - "robust": 稳健标准化(使用中位数和MAD)
            - "rank": 排序标准化(转换为正态分布)
        
        Returns:
        --------
        标准化后的数据
        """
        if isinstance(data, np.ndarray):
            return Normalizer._normalize_numpy(data, method)
        elif isinstance(data, pd.Series):
            return Normalizer._normalize_series(data, method)
        elif isinstance(data, pd.DataFrame):
            return data.apply(lambda col: Normalizer._normalize_series(col, method))
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @staticmethod
    def _normalize_numpy(data: np.ndarray, method: str) -> np.ndarray:
        """标准化numpy数组"""
        result = data.astype(float).copy()
        
        if method == "zscore":
            mean_val = np.nanmean(result)
            std_val = np.nanstd(result)
            if std_val > 1e-6:
                result = (result - mean_val) / std_val
            else:
                result = result * 0
                
        elif method == "minmax":
            min_val = np.nanmin(result)
            max_val = np.nanmax(result)
            if max_val - min_val > 1e-6:
                result = (result - min_val) / (max_val - min_val)
            else:
                result = result * 0
                
        elif method == "robust":
            median_val = np.nanmedian(result)
            mad = np.nanmedian(np.abs(result - median_val))
            if mad > 1e-6:
                result = (result - median_val) / (1.4826 * mad)
            else:
                result = result * 0
                
        elif method == "rank":
            # 排序转换为正态分布
            ranks = np.argsort(np.argsort(result))
            n = len(result)
            result = norm.ppf((ranks + 0.5) / n)
            
        return result
    
    @staticmethod
    def _normalize_series(series: pd.Series, method: str) -> pd.Series:
        """标准化pandas Series"""
        result = series.copy()
        
        if method == "zscore":
            mean_val = result.mean()
            std_val = result.std()
            if std_val > 1e-6:
                result = (result - mean_val) / std_val
            else:
                result = result * 0
                
        elif method == "minmax":
            min_val = result.min()
            max_val = result.max()
            if max_val - min_val > 1e-6:
                result = (result - min_val) / (max_val - min_val)
            else:
                result = result * 0
                
        elif method == "robust":
            median_val = result.median()
            mad = (result - median_val).abs().median()
            if mad > 1e-6:
                result = (result - median_val) / (1.4826 * mad)
            else:
                result = result * 0
                
        elif method == "rank":
            # 使用pandas的rank方法
            ranks = result.rank(pct=True)
            result = norm.ppf(ranks)
            result[result == np.inf] = 3
            result[result == -np.inf] = -3
            
        return result


class DataCleaner:
    """数据清洗综合工具"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗DataFrame
        - 删除全零行/列
        - 合并重复索引
        """
        result = df.copy()
        
        # 如果有多级索引，先group by合并
        if isinstance(result.index, pd.MultiIndex):
            result = result.groupby(level=list(range(result.index.nlevels))).sum()
        
        # 删除全零行
        result = result[(result != 0).any(axis=1)]
        
        # 删除全零列
        result = result.loc[:, (result != 0).any(axis=0)]
        
        logger.info(f"Cleaned dataframe shape: {result.shape}")
        return result
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        method: Literal["drop", "forward_fill", "backward_fill", "interpolate", "median"] = "median"
    ) -> pd.DataFrame:
        """
        处理缺失值
        """
        result = df.copy()
        
        if method == "drop":
            result = result.dropna()
        elif method == "forward_fill":
            result = result.fillna(method='ffill')
        elif method == "backward_fill":
            result = result.fillna(method='bfill')
        elif method == "interpolate":
            result = result.interpolate()
        elif method == "median":
            result = result.fillna(result.median())
            
        return result
    
    @staticmethod
    def delete_all_zeros(
        df: pd.DataFrame,
        axis: Union[None, int, str] = None
    ) -> pd.DataFrame:
        """
        删除全为0的行和列
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据框
        axis : None, 0, 1, 'index', 'columns', optional
            删除的轴向
            - None: 同时删除全0的行和列
            - 0 或 'index': 只删除全0的行
            - 1 或 'columns': 只删除全0的列
            
        Returns
        -------
        pd.DataFrame
            删除全0行列后的数据框
        """
        result = df.copy()
        
        if axis is None:
            # 删除全为0的列
            result = result.loc[:, (result != 0).any(axis=0)]
            # 删除全为0的行
            result = result.loc[(result != 0).any(axis=1), :]
        elif axis in [0, 'index']:
            # 只删除全为0的行
            result = result.loc[(result != 0).any(axis=1), :]
        elif axis in [1, 'columns']:
            # 只删除全为0的列
            result = result.loc[:, (result != 0).any(axis=0)]
        else:
            raise ValueError(f"Invalid axis: {axis}")
            
        return result