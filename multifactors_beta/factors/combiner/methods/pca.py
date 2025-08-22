"""
PCA主成分分析方法

使用PCA进行因子降维和特征提取
"""

from typing import Dict, Optional, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PCACombiner:
    """
    PCA组合器
    
    使用主成分分析进行因子降维和组合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化PCA组合器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.n_components = self.config.get('n_components', None)  # 主成分数量
        self.explained_variance_ratio = self.config.get('explained_variance_ratio', 0.95)  # 解释方差比例
        self.standardize = self.config.get('standardize', True)  # 是否标准化
        self.handle_missing = self.config.get('handle_missing', 'mean')  # 缺失值处理
        self.min_observations = self.config.get('min_observations', 10)
        self.rolling_window = self.config.get('rolling_window', None)  # 滚动窗口
        
        # 存储PCA模型和统计信息
        self.pca_models = {}
        self.pca_stats = {}
    
    def fit_transform(self,
                     factors: Dict[str, pd.Series],
                     return_components: bool = True) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        拟合并转换因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子字典
        return_components : bool
            是否返回所有主成分，否则只返回第一主成分
            
        Returns
        -------
        Union[pd.Series, Dict[str, pd.Series]]
            主成分因子
        """
        if not factors:
            raise ValueError("No factors to transform")
        
        # 对齐因子
        aligned_factors = self._align_factors(factors)
        
        if self.rolling_window:
            return self._rolling_pca(aligned_factors, return_components)
        else:
            return self._static_pca(aligned_factors, return_components)
    
    def _align_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        对齐因子索引
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
        
        # 对齐
        aligned = {}
        for name, factor in factors.items():
            aligned[name] = factor.reindex(common_index)
        
        return aligned
    
    def _static_pca(self,
                   factors: Dict[str, pd.Series],
                   return_components: bool) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        静态PCA（全局拟合）
        """
        # 转换为DataFrame
        factor_df = pd.DataFrame(factors)
        factor_names = list(factors.keys())
        
        # 获取日期
        dates = factor_df.index.get_level_values(0).unique()
        
        # 存储PCA结果
        pca_results = []
        
        for date in dates:
            # 获取当日数据
            date_data = factor_df.xs(date, level=0)
            
            # 处理缺失值
            if self.handle_missing == 'drop':
                date_data = date_data.dropna()
            elif self.handle_missing == 'mean':
                date_data = date_data.fillna(date_data.mean())
            elif self.handle_missing == 'zero':
                date_data = date_data.fillna(0)
            
            # 检查数据量
            if len(date_data) < self.min_observations:
                continue
            
            # 标准化
            if self.standardize:
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(date_data)
            else:
                standardized_data = date_data.values
            
            # 确定主成分数量
            n_components = self._determine_n_components(standardized_data)
            
            # PCA拟合
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(standardized_data)
            
            # 存储PCA模型和统计信息
            self.pca_models[date] = pca
            self.pca_stats[date] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                'n_components': n_components,
                'loadings': pd.DataFrame(
                    pca.components_.T,
                    index=factor_names,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
            }
            
            # 保存结果
            for idx, stock in enumerate(date_data.index):
                for comp_idx in range(n_components):
                    pca_results.append((
                        date,
                        stock,
                        f'PC{comp_idx+1}',
                        components[idx, comp_idx]
                    ))
        
        # 重构为因子格式
        if return_components:
            # 返回所有主成分
            component_dict = {}
            for comp_idx in range(n_components):
                comp_name = f'PC{comp_idx+1}'
                comp_data = [(d, s, v) for d, s, c, v in pca_results if c == comp_name]
                
                if comp_data:
                    index = pd.MultiIndex.from_tuples([(d, s) for d, s, _ in comp_data])
                    values = [v for _, _, v in comp_data]
                    component_dict[comp_name] = pd.Series(values, index=index, name=comp_name)
            
            logger.info(f"Extracted {len(component_dict)} principal components")
            return component_dict
        else:
            # 只返回第一主成分
            pc1_data = [(d, s, v) for d, s, c, v in pca_results if c == 'PC1']
            
            if pc1_data:
                index = pd.MultiIndex.from_tuples([(d, s) for d, s, _ in pc1_data])
                values = [v for _, _, v in pc1_data]
                pc1 = pd.Series(values, index=index, name='PC1')
                logger.info("Extracted first principal component")
                return pc1
            else:
                return pd.Series([], index=pd.MultiIndex.from_tuples([]), name='PC1')
    
    def _rolling_pca(self,
                    factors: Dict[str, pd.Series],
                    return_components: bool) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        滚动PCA
        """
        # 转换为DataFrame
        factor_df = pd.DataFrame(factors)
        factor_names = list(factors.keys())
        
        # 获取日期
        dates = factor_df.index.get_level_values(0).unique()
        dates = sorted(dates)
        
        # 存储PCA结果
        pca_results = []
        
        for i in range(len(dates)):
            current_date = dates[i]
            
            # 确定窗口范围
            start_idx = max(0, i - self.rolling_window + 1)
            window_dates = dates[start_idx:i+1]
            
            if len(window_dates) < self.min_observations:
                continue
            
            # 获取窗口内的数据
            window_data = factor_df[
                factor_df.index.get_level_values(0).isin(window_dates)
            ]
            
            # 按日期分组处理
            all_data = []
            for date in window_dates:
                date_data = window_data.xs(date, level=0)
                
                # 处理缺失值
                if self.handle_missing == 'mean':
                    date_data = date_data.fillna(date_data.mean())
                elif self.handle_missing == 'zero':
                    date_data = date_data.fillna(0)
                
                all_data.append(date_data.values)
            
            # 合并所有窗口内数据
            combined_data = np.vstack(all_data)
            
            # 标准化
            if self.standardize:
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(combined_data)
            else:
                standardized_data = combined_data
            
            # 确定主成分数量
            n_components = self._determine_n_components(standardized_data)
            
            # PCA拟合
            pca = PCA(n_components=n_components)
            pca.fit(standardized_data)
            
            # 对当前日期的数据进行转换
            current_data = factor_df.xs(current_date, level=0)
            
            # 处理缺失值
            if self.handle_missing == 'mean':
                current_data = current_data.fillna(current_data.mean())
            elif self.handle_missing == 'zero':
                current_data = current_data.fillna(0)
            
            # 标准化当前数据
            if self.standardize:
                current_standardized = scaler.transform(current_data)
            else:
                current_standardized = current_data.values
            
            # 转换
            components = pca.transform(current_standardized)
            
            # 存储PCA模型和统计信息
            self.pca_models[current_date] = pca
            self.pca_stats[current_date] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                'n_components': n_components,
                'window_size': len(window_dates)
            }
            
            # 保存结果
            for idx, stock in enumerate(current_data.index):
                for comp_idx in range(n_components):
                    pca_results.append((
                        current_date,
                        stock,
                        f'PC{comp_idx+1}',
                        components[idx, comp_idx]
                    ))
        
        # 重构为因子格式
        if return_components:
            # 获取最大主成分数
            max_components = max([stat['n_components'] for stat in self.pca_stats.values()])
            
            component_dict = {}
            for comp_idx in range(max_components):
                comp_name = f'PC{comp_idx+1}'
                comp_data = [(d, s, v) for d, s, c, v in pca_results if c == comp_name]
                
                if comp_data:
                    index = pd.MultiIndex.from_tuples([(d, s) for d, s, _ in comp_data])
                    values = [v for _, _, v in comp_data]
                    component_dict[comp_name] = pd.Series(values, index=index, name=comp_name)
            
            logger.info(f"Extracted {len(component_dict)} principal components using rolling window")
            return component_dict
        else:
            # 只返回第一主成分
            pc1_data = [(d, s, v) for d, s, c, v in pca_results if c == 'PC1']
            
            if pc1_data:
                index = pd.MultiIndex.from_tuples([(d, s) for d, s, _ in pc1_data])
                values = [v for _, _, v in pc1_data]
                pc1 = pd.Series(values, index=index, name='PC1')
                logger.info("Extracted first principal component using rolling window")
                return pc1
            else:
                return pd.Series([], index=pd.MultiIndex.from_tuples([]), name='PC1')
    
    def _determine_n_components(self, data: np.ndarray) -> int:
        """
        确定主成分数量
        """
        if self.n_components is not None:
            # 固定数量
            return min(self.n_components, min(data.shape))
        else:
            # 基于解释方差比例
            # 先做一个完整的PCA来确定需要的成分数
            pca_full = PCA()
            pca_full.fit(data)
            
            cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum_ratio >= self.explained_variance_ratio) + 1
            
            return min(n_components, min(data.shape))
    
    def get_loadings(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        获取因子载荷（贡献度）
        
        Parameters
        ----------
        date : pd.Timestamp, optional
            指定日期，如果不指定则返回最新的
            
        Returns
        -------
        pd.DataFrame
            因子载荷矩阵
        """
        if date is None:
            # 获取最新日期
            if self.pca_stats:
                date = max(self.pca_stats.keys())
            else:
                return pd.DataFrame()
        
        if date in self.pca_stats and 'loadings' in self.pca_stats[date]:
            return self.pca_stats[date]['loadings']
        elif date in self.pca_models:
            # 重新计算载荷
            pca = self.pca_models[date]
            factor_names = [f'Factor{i+1}' for i in range(pca.components_.shape[1])]
            loadings = pd.DataFrame(
                pca.components_.T,
                index=factor_names,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            return loadings
        else:
            return pd.DataFrame()
    
    def get_variance_explained(self, date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        获取方差解释信息
        
        Parameters
        ----------
        date : pd.Timestamp, optional
            指定日期
            
        Returns
        -------
        Dict[str, Any]
            方差解释信息
        """
        if date is None:
            # 获取最新日期
            if self.pca_stats:
                date = max(self.pca_stats.keys())
            else:
                return {}
        
        if date in self.pca_stats:
            return {
                'explained_variance_ratio': self.pca_stats[date]['explained_variance_ratio'],
                'cumulative_variance_ratio': self.pca_stats[date]['cumulative_variance_ratio'],
                'n_components': self.pca_stats[date]['n_components']
            }
        else:
            return {}
    
    def reconstruct_factors(self,
                          components: Union[pd.Series, Dict[str, pd.Series]],
                          date: Optional[pd.Timestamp] = None) -> Dict[str, pd.Series]:
        """
        从主成分重构原始因子
        
        Parameters
        ----------
        components : Union[pd.Series, Dict[str, pd.Series]]
            主成分数据
        date : pd.Timestamp, optional
            使用的PCA模型日期
            
        Returns
        -------
        Dict[str, pd.Series]
            重构的因子
        """
        if date is None and self.pca_models:
            date = max(self.pca_models.keys())
        
        if date not in self.pca_models:
            raise ValueError(f"No PCA model found for date {date}")
        
        pca = self.pca_models[date]
        
        # 处理输入格式
        if isinstance(components, pd.Series):
            components_df = pd.DataFrame({'PC1': components})
        else:
            components_df = pd.DataFrame(components)
        
        # 重构
        reconstructed = pca.inverse_transform(components_df.values)
        
        # 转换为因子格式
        factor_names = [f'Factor{i+1}' for i in range(reconstructed.shape[1])]
        reconstructed_factors = {}
        
        for i, name in enumerate(factor_names):
            reconstructed_factors[name] = pd.Series(
                reconstructed[:, i],
                index=components_df.index,
                name=name
            )
        
        return reconstructed_factors