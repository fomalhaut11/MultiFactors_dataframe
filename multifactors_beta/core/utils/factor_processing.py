"""因子处理工具模块"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)


class FactorOrthogonalizer:
    """因子正交化处理器"""
    
    @staticmethod
    def sequential_orthogonalize(
        factors: pd.DataFrame,
        normalize: bool = True,
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        顺序正交化因子
        
        Parameters:
        -----------
        factors : DataFrame，多级索引(TradingDates, StockCodes)，列为因子
        normalize : 是否标准化
        remove_outliers : 是否去除异常值
        
        Returns:
        --------
        正交化后的因子DataFrame
        """
        # 导入所需模块
        from .data_cleaning import OutlierHandler, Normalizer

        # 按日期分组处理
        def _orthogonalize_date(date_data: pd.DataFrame) -> pd.DataFrame:
            """对单个日期的数据进行正交化"""
            result = pd.DataFrame(index=date_data.index, columns=date_data.columns, dtype=float)

            for i, col in enumerate(date_data.columns):
                factor_data = date_data[col].values

                # 去除异常值
                if remove_outliers:
                    factor_data = OutlierHandler.remove_outlier(factor_data, method="IQR")

                # 标准化
                if normalize:
                    factor_data = Normalizer.normalize(factor_data, method="zscore")

                if i == 0:
                    # 第一个因子直接使用
                    result.iloc[:, i] = factor_data
                else:
                    # 对之前的因子进行回归，取残差
                    try:
                        X = sm.add_constant(result.iloc[:, :i].values)
                        model = sm.OLS(factor_data, X)
                        residuals = model.fit().resid
                        
                        # 重新标准化残差
                        if normalize:
                            residuals = Normalizer.normalize(residuals, method="zscore")
                        
                        result.iloc[:, i] = residuals
                    except Exception as e:
                        logger.warning(f"Orthogonalization failed for factor {col}: {e}")
                        result.iloc[:, i] = factor_data
            
            return result
        
        # 按日期分组应用
        result = factors.groupby(level='TradingDates').apply(_orthogonalize_date)
        
        # 保持原始索引结构
        if isinstance(result.index, pd.MultiIndex) and result.index.nlevels == 3:
            result.index = result.index.droplevel(0)
        
        return result
    
    @staticmethod
    def add_new_factor_orthogonal(
        base_factors: pd.DataFrame,
        new_factors: pd.DataFrame,
        normalize: bool = True,
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        在已有正交化因子基础上添加新因子并正交化
        
        Parameters:
        -----------
        base_factors : 已经正交化的基础因子
        new_factors : 新因子
        
        Returns:
        --------
        合并后的正交化因子
        """
        # 导入所需模块
        from .data_cleaning import OutlierHandler, Normalizer
        
        # 合并数据
        merged_data = base_factors.join(new_factors, how='inner')
        merged_data = merged_data.dropna()
        
        base_cols = base_factors.columns.tolist()
        new_cols = new_factors.columns.tolist()
        
        def _orthogonalize_new_factors(date_data: pd.DataFrame) -> pd.DataFrame:
            """对新因子进行正交化"""
            result = date_data.copy()
            
            # 获取基础因子数据
            base_data = date_data[base_cols].values
            X = sm.add_constant(base_data)
            
            # 对每个新因子进行正交化
            for col in new_cols:
                factor_data = date_data[col].values
                
                # 去除异常值
                if remove_outliers:
                    factor_data = OutlierHandler.remove_outlier(factor_data, method="IQR")
                
                # 标准化
                if normalize:
                    factor_data = Normalizer.normalize(factor_data, method="zscore")
                
                try:
                    # 回归并取残差
                    model = sm.OLS(factor_data, X)
                    residuals = model.fit().resid
                    
                    # 重新标准化
                    if normalize:
                        residuals = Normalizer.normalize(residuals, method="zscore")
                    
                    result[col] = residuals
                except Exception as e:
                    logger.warning(f"Orthogonalization failed for new factor {col}: {e}")
                    result[col] = factor_data
            
            return result[base_cols + new_cols]
        
        # 按日期分组处理
        result = merged_data.groupby(level='TradingDates').apply(_orthogonalize_new_factors)
        
        # 保持索引结构
        if isinstance(result.index, pd.MultiIndex) and result.index.nlevels == 3:
            result.index = result.index.droplevel(0)
        
        return result


class FactorProcessor:
    """因子处理综合工具"""
    
    @staticmethod
    def neutralize_factor(
        factor: pd.Series,
        benchmark_factors: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        因子中性化
        
        Parameters:
        -----------
        factor : 需要中性化的因子
        benchmark_factors : 基准因子(如行业、风格等)
        returns : 收益率数据(可选)
        
        Returns:
        --------
        中性化后的因子
        """
        # 导入所需模块
        from .data_cleaning import OutlierHandler, Normalizer
        
        # 合并数据
        data = pd.DataFrame({'factor': factor})
        data = data.join(benchmark_factors, how='inner')
        if returns is not None:
            data['returns'] = returns
        
        # 删除缺失值
        data = data.dropna()
        
        def _neutralize_date(date_data: pd.DataFrame) -> pd.Series:
            """对单个日期的数据进行中性化"""
            # 去除异常值和标准化
            y = date_data['factor'].values
            y = OutlierHandler.remove_outlier(y, method="IQR")
            y = Normalizer.normalize(y, method="zscore")
            
            # 构建回归变量
            X = date_data[benchmark_factors.columns].values
            X = sm.add_constant(X)
            
            try:
                # OLS回归
                model = sm.OLS(y, X)
                result = model.fit()
                
                # 返回残差(中性化后的因子)
                residuals = result.resid
                
                # 重新标准化
                residuals = Normalizer.normalize(residuals, method="zscore")
                
                return pd.Series(residuals, index=date_data.index)
            except Exception as e:
                logger.warning(f"Neutralization failed: {e}")
                return pd.Series(y, index=date_data.index)
        
        # 按日期分组处理
        result = data.groupby(level='TradingDates').apply(
            lambda x: _neutralize_date(x)
        )
        
        # 整理索引
        if isinstance(result.index, pd.MultiIndex) and result.index.nlevels == 3:
            # 删除多余的索引层级
            result.index = result.index.droplevel(0)
        
        return result
    
    @staticmethod
    def winsorize_factor(
        factor: pd.Series,
        method: str = "mad",
        threshold: float = 3.0
    ) -> pd.Series:
        """
        因子去极值处理
        """
        from .data_cleaning import OutlierHandler
        
        # 按日期分组去极值
        def _winsorize_date(date_data: pd.Series) -> pd.Series:
            values = OutlierHandler.remove_outlier(
                date_data.values, 
                method="median" if method == "mad" else method,
                threshold=threshold
            )
            return pd.Series(values, index=date_data.index)
        
        return factor.groupby(level='TradingDates').apply(_winsorize_date)
    
    @staticmethod
    def standardize_factor(
        factor: pd.Series,
        method: str = "zscore"
    ) -> pd.Series:
        """
        因子标准化
        """
        from .data_cleaning import Normalizer
        
        # 按日期分组标准化
        def _standardize_date(date_data: pd.Series) -> pd.Series:
            values = Normalizer.normalize(date_data.values, method=method)
            return pd.Series(values, index=date_data.index)
        
        return factor.groupby(level='TradingDates').apply(_standardize_date)