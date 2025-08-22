"""
正交化方法

实现因子的正交化处理，包括Gram-Schmidt和残差正交化
"""

from typing import Dict, Optional, Any, List, Union
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class OrthogonalCombiner:
    """
    正交化组合器
    
    提供多种正交化方法，消除因子间的相关性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化正交化组合器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.method = self.config.get('orthogonal_method', 'gram_schmidt')
        self.handle_missing = self.config.get('handle_missing', 'forward_fill')
        self.normalize_after = self.config.get('normalize_after', True)
        self.min_observations = self.config.get('min_observations', 10)
    
    def orthogonalize(self,
                     factors: Dict[str, pd.Series],
                     base_factor: Optional[str] = None,
                     method: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        正交化因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子字典
        base_factor : str, optional
            基准因子（保持不变）
        method : str, optional
            正交化方法，覆盖初始化时的设置
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        if not factors:
            raise ValueError("No factors to orthogonalize")
        
        method = method or self.method
        
        # 对齐因子
        aligned_factors = self._align_factors(factors)
        
        if method == 'gram_schmidt':
            return self._gram_schmidt_orthogonalize(aligned_factors, base_factor)
        elif method == 'residual':
            return self._residual_orthogonalize(aligned_factors, base_factor)
        elif method == 'symmetric':
            return self._symmetric_orthogonalize(aligned_factors)
        else:
            raise ValueError(f"Unknown orthogonalization method: {method}")
    
    def _align_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        对齐因子索引
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子
            
        Returns
        -------
        Dict[str, pd.Series]
            对齐后的因子
        """
        # 找到公共索引
        common_index = None
        for factor in factors.values():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        if len(common_index) == 0:
            raise ValueError("No common index found among factors")
        
        # 对齐并处理缺失值
        aligned = {}
        for name, factor in factors.items():
            aligned_factor = factor.reindex(common_index)
            
            # 处理缺失值
            if self.handle_missing == 'forward_fill':
                aligned_factor = aligned_factor.fillna(method='ffill')
            elif self.handle_missing == 'drop':
                pass  # 在后续步骤统一处理
            elif self.handle_missing == 'zero':
                aligned_factor = aligned_factor.fillna(0)
            
            aligned[name] = aligned_factor
        
        return aligned
    
    def _gram_schmidt_orthogonalize(self,
                                   factors: Dict[str, pd.Series],
                                   base_factor: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Gram-Schmidt正交化
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
        base_factor : str, optional
            基准因子
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        factor_names = list(factors.keys())
        
        # 调整顺序，基准因子放在第一位
        if base_factor and base_factor in factor_names:
            factor_names.remove(base_factor)
            factor_names.insert(0, base_factor)
        
        # 转换为DataFrame便于处理
        factor_df = pd.DataFrame(factors)[factor_names]  # 按调整后的顺序
        
        # 获取日期
        dates = factor_df.index.get_level_values(0).unique()
        
        # 存储正交化后的结果
        orthogonal_results = []
        
        for date in dates:
            # 获取当日数据
            date_data = factor_df.xs(date, level=0)
            
            # 检查数据量
            if len(date_data.dropna()) < self.min_observations:
                continue
            
            # 转换为numpy数组
            data = date_data.values
            n_obs, n_factors = data.shape
            
            # Gram-Schmidt过程
            orthogonal = np.zeros_like(data)
            
            for i in range(n_factors):
                orthogonal[:, i] = data[:, i]
                
                # 减去之前向量的投影
                for j in range(i):
                    # 计算投影系数
                    if np.nanstd(orthogonal[:, j]) > 0:
                        # 处理缺失值
                        valid_mask = ~(np.isnan(data[:, i]) | np.isnan(orthogonal[:, j]))
                        if valid_mask.sum() > 0:
                            projection = np.nansum(data[valid_mask, i] * orthogonal[valid_mask, j]) / \
                                       np.nansum(orthogonal[valid_mask, j] ** 2)
                            orthogonal[:, i] -= projection * orthogonal[:, j]
                
                # 标准化（可选）
                if self.normalize_after:
                    std = np.nanstd(orthogonal[:, i])
                    if std > 0:
                        orthogonal[:, i] = (orthogonal[:, i] - np.nanmean(orthogonal[:, i])) / std
            
            # 保存结果
            for idx, stock in enumerate(date_data.index):
                for i, factor_name in enumerate(factor_names):
                    if not np.isnan(orthogonal[idx, i]):
                        orthogonal_results.append((date, stock, factor_name, orthogonal[idx, i]))
        
        # 重构为字典格式
        orthogonal_factors = {name: [] for name in factor_names}
        
        for date, stock, factor_name, value in orthogonal_results:
            orthogonal_factors[factor_name].append(((date, stock), value))
        
        # 转换为Series
        for name in factor_names:
            if orthogonal_factors[name]:
                index = pd.MultiIndex.from_tuples([idx for idx, _ in orthogonal_factors[name]])
                values = [val for _, val in orthogonal_factors[name]]
                orthogonal_factors[name] = pd.Series(values, index=index, name=name)
            else:
                # 返回空Series
                orthogonal_factors[name] = pd.Series(
                    [], 
                    index=pd.MultiIndex.from_tuples([]),
                    name=name
                )
        
        logger.info(f"Orthogonalized {len(factors)} factors using Gram-Schmidt method")
        return orthogonal_factors
    
    def _residual_orthogonalize(self,
                               factors: Dict[str, pd.Series],
                               base_factor: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        残差正交化
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
        base_factor : str, optional
            基准因子
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        if not base_factor or base_factor not in factors:
            # 如果没有指定基准因子，使用第一个
            base_factor = next(iter(factors.keys()))
            logger.info(f"No base factor specified, using {base_factor}")
        
        orthogonal_factors = {}
        base_series = factors[base_factor]
        
        # 基准因子保持不变
        orthogonal_factors[base_factor] = base_series.copy()
        
        # 其他因子取残差
        for name, factor in factors.items():
            if name == base_factor:
                continue
            
            # 按日期回归
            residuals = []
            dates = factor.index.get_level_values(0).unique()
            
            for date in dates:
                try:
                    # 获取当日数据
                    base_day = base_series.xs(date, level=0)
                    factor_day = factor.xs(date, level=0)
                    
                    # 对齐
                    common_stocks = base_day.index.intersection(factor_day.index)
                    if len(common_stocks) < self.min_observations:
                        continue
                    
                    # 准备数据
                    X = base_day.loc[common_stocks].values.reshape(-1, 1)
                    y = factor_day.loc[common_stocks].values
                    
                    # 处理缺失值
                    valid_mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                    if valid_mask.sum() < self.min_observations:
                        continue
                    
                    X_valid = X[valid_mask].reshape(-1, 1)
                    y_valid = y[valid_mask]
                    
                    # 线性回归
                    reg = LinearRegression()
                    reg.fit(X_valid, y_valid)
                    
                    # 计算所有股票的残差（包括有缺失值的）
                    predictions = reg.predict(X)
                    residual = y - predictions.flatten()
                    
                    # 标准化残差（可选）
                    if self.normalize_after:
                        std = np.nanstd(residual)
                        if std > 0:
                            residual = (residual - np.nanmean(residual)) / std
                    
                    # 保存残差
                    for i, stock in enumerate(common_stocks):
                        if not np.isnan(residual[i]):
                            residuals.append((date, stock, residual[i]))
                            
                except Exception as e:
                    logger.warning(f"Failed to process date {date}: {e}")
                    continue
            
            # 构建残差Series
            if residuals:
                residual_index = pd.MultiIndex.from_tuples(
                    [(d, s) for d, s, _ in residuals]
                )
                residual_values = [r for _, _, r in residuals]
                orthogonal_factors[name] = pd.Series(
                    residual_values,
                    index=residual_index,
                    name=name
                )
            else:
                # 返回空Series
                orthogonal_factors[name] = pd.Series(
                    [],
                    index=pd.MultiIndex.from_tuples([]),
                    name=name
                )
        
        logger.info(f"Orthogonalized {len(factors)} factors using residual method")
        return orthogonal_factors
    
    def _symmetric_orthogonalize(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        对称正交化（所有因子同等对待）
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        # 转换为DataFrame
        factor_df = pd.DataFrame(factors)
        factor_names = list(factors.keys())
        
        # 获取日期
        dates = factor_df.index.get_level_values(0).unique()
        
        # 存储正交化后的结果
        orthogonal_results = []
        
        for date in dates:
            # 获取当日数据
            date_data = factor_df.xs(date, level=0)
            
            # 检查数据量
            valid_data = date_data.dropna()
            if len(valid_data) < self.min_observations:
                continue
            
            # 计算相关矩阵
            corr_matrix = date_data.corr()
            
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
            
            # 构造正交化矩阵（使用特征向量）
            # 这里使用主成分方向作为正交基
            transform_matrix = eigenvectors.T
            
            # 标准化数据
            standardized = date_data.values.copy()
            for i in range(len(factor_names)):
                col_data = standardized[:, i]
                valid_mask = ~np.isnan(col_data)
                if valid_mask.sum() > 0:
                    mean = np.mean(col_data[valid_mask])
                    std = np.std(col_data[valid_mask])
                    if std > 0:
                        standardized[:, i] = (col_data - mean) / std
            
            # 应用正交变换
            orthogonal = standardized @ transform_matrix.T
            
            # 保存结果
            for idx, stock in enumerate(date_data.index):
                for i, factor_name in enumerate(factor_names):
                    if not np.isnan(orthogonal[idx, i]):
                        orthogonal_results.append((date, stock, factor_name, orthogonal[idx, i]))
        
        # 重构为字典格式
        orthogonal_factors = {name: [] for name in factor_names}
        
        for date, stock, factor_name, value in orthogonal_results:
            orthogonal_factors[factor_name].append(((date, stock), value))
        
        # 转换为Series
        for name in factor_names:
            if orthogonal_factors[name]:
                index = pd.MultiIndex.from_tuples([idx for idx, _ in orthogonal_factors[name]])
                values = [val for _, val in orthogonal_factors[name]]
                orthogonal_factors[name] = pd.Series(values, index=index, name=name)
            else:
                orthogonal_factors[name] = pd.Series(
                    [],
                    index=pd.MultiIndex.from_tuples([]),
                    name=name
                )
        
        logger.info(f"Orthogonalized {len(factors)} factors using symmetric method")
        return orthogonal_factors
    
    def calculate_correlation_reduction(self,
                                       original_factors: Dict[str, pd.Series],
                                       orthogonal_factors: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算正交化前后的相关性降低程度
        
        Parameters
        ----------
        original_factors : Dict[str, pd.Series]
            原始因子
        orthogonal_factors : Dict[str, pd.Series]
            正交化后的因子
            
        Returns
        -------
        Dict[str, float]
            相关性降低的统计信息
        """
        # 计算原始相关矩阵
        original_df = pd.DataFrame(original_factors)
        original_corr = original_df.corr()
        
        # 计算正交化后的相关矩阵
        orthogonal_df = pd.DataFrame(orthogonal_factors)
        orthogonal_corr = orthogonal_df.corr()
        
        # 提取非对角线元素
        n = len(original_corr)
        original_off_diag = []
        orthogonal_off_diag = []
        
        for i in range(n):
            for j in range(i+1, n):
                original_off_diag.append(abs(original_corr.iloc[i, j]))
                orthogonal_off_diag.append(abs(orthogonal_corr.iloc[i, j]))
        
        # 计算统计信息
        stats = {
            'original_mean_corr': np.mean(original_off_diag),
            'orthogonal_mean_corr': np.mean(orthogonal_off_diag),
            'correlation_reduction': np.mean(original_off_diag) - np.mean(orthogonal_off_diag),
            'reduction_percentage': (np.mean(original_off_diag) - np.mean(orthogonal_off_diag)) / 
                                   (np.mean(original_off_diag) + 1e-10) * 100
        }
        
        return stats