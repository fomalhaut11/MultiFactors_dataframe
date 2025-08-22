"""
相关性筛选器

基于因子间相关性进行筛选，去除高相关因子
"""

from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class CorrelationFilter(BaseFilter):
    """
    相关性筛选器
    
    去除高相关的因子，保持因子池的多样性
    """
    
    def __init__(self,
                 max_correlation: float = 0.7,
                 method: str = 'hierarchical',
                 keep_best: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化相关性筛选器
        
        Parameters
        ----------
        max_correlation : float
            最大允许相关性阈值
        method : str
            筛选方法：'hierarchical', 'greedy', 'cluster'
        keep_best : bool
            在相关因子中是否保留评分最高的
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        self.max_correlation = max_correlation
        self.method = method
        self.keep_best = keep_best
        
        # 验证方法
        valid_methods = ['hierarchical', 'greedy', 'cluster']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
    
    def filter(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               correlation_matrix: Optional[pd.DataFrame] = None,
               **kwargs) -> Dict[str, pd.Series]:
        """
        基于相关性筛选因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            待筛选的因子
        evaluation_results : Dict, optional
            评估结果，用于决定保留哪个因子
        correlation_matrix : pd.DataFrame, optional
            预计算的相关性矩阵
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            筛选后的因子
        """
        self.validate_inputs(factors, evaluation_results)
        
        if len(factors) <= 1:
            return factors
        
        # 计算或获取相关性矩阵
        if correlation_matrix is None:
            correlation_matrix = self._calculate_correlation_matrix(factors)
        
        # 筛选因子名称，只保留在相关性矩阵中的因子
        available_factors = {
            name: factor for name, factor in factors.items()
            if name in correlation_matrix.index and name in correlation_matrix.columns
        }
        
        if len(available_factors) <= 1:
            logger.warning("Not enough factors in correlation matrix for filtering")
            return factors
        
        # 应用不同的筛选方法
        if self.method == 'hierarchical':
            selected_names = self._hierarchical_filter(
                available_factors, correlation_matrix, evaluation_results
            )
        elif self.method == 'greedy':
            selected_names = self._greedy_filter(
                available_factors, correlation_matrix, evaluation_results
            )
        elif self.method == 'cluster':
            selected_names = self._cluster_filter(
                available_factors, correlation_matrix, evaluation_results
            )
        else:
            selected_names = list(available_factors.keys())
        
        # 构建筛选结果
        filtered = {name: factors[name] for name in selected_names if name in factors}
        
        # 添加不在相关性矩阵中的因子
        for name, factor in factors.items():
            if name not in available_factors and name not in filtered:
                filtered[name] = factor
        
        # 记录筛选历史
        self.filter_history.append({
            'original_count': len(factors),
            'filtered_count': len(filtered),
            'method': self.method,
            'max_correlation': self.max_correlation
        })
        
        logger.info(
            f"Correlation filter ({self.method}): {len(factors)} -> {len(filtered)} factors"
        )
        
        return filtered
    
    def _calculate_correlation_matrix(self, factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
            
        Returns
        -------
        pd.DataFrame
            相关性矩阵
        """
        # 对齐所有因子到公共索引
        common_index = None
        for factor in factors.values():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        if len(common_index) == 0:
            logger.warning("No common index found for correlation calculation")
            # 返回单位矩阵
            factor_names = list(factors.keys())
            return pd.DataFrame(
                np.eye(len(factor_names)),
                index=factor_names,
                columns=factor_names
            )
        
        # 构建DataFrame
        aligned_data = {}
        for name, factor in factors.items():
            aligned_factor = factor.reindex(common_index).dropna()
            if len(aligned_factor) > 0:
                aligned_data[name] = aligned_factor
        
        if len(aligned_data) < 2:
            factor_names = list(factors.keys())
            return pd.DataFrame(
                np.eye(len(factor_names)),
                index=factor_names,
                columns=factor_names
            )
        
        factor_df = pd.DataFrame(aligned_data)
        
        # 计算Spearman相关性
        correlation_matrix = factor_df.corr(method='spearman')
        
        return correlation_matrix
    
    def _hierarchical_filter(self,
                             factors: Dict[str, pd.Series],
                             correlation_matrix: pd.DataFrame,
                             evaluation_results: Optional[Dict] = None) -> List[str]:
        """
        层次化筛选方法
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
        correlation_matrix : pd.DataFrame
            相关性矩阵
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        List[str]
            选中的因子名称
        """
        factor_names = list(factors.keys())
        to_remove = set()
        
        # 获取因子评分用于排序
        scores = {}
        if evaluation_results and self.keep_best:
            for name in factor_names:
                if name in evaluation_results:
                    result = evaluation_results[name]
                    # 尝试获取总分
                    if hasattr(result, 'total_score'):
                        scores[name] = result.total_score
                    elif hasattr(result, 'metrics') and 'total_score' in result.metrics:
                        scores[name] = result.metrics['total_score']
                    else:
                        scores[name] = 0
                else:
                    scores[name] = 0
        else:
            scores = {name: 0 for name in factor_names}
        
        # 按评分排序，评分高的优先保留
        sorted_factors = sorted(factor_names, key=lambda x: scores.get(x, 0), reverse=True)
        
        for i, factor1 in enumerate(sorted_factors):
            if factor1 in to_remove:
                continue
            
            for factor2 in sorted_factors[i+1:]:
                if factor2 in to_remove:
                    continue
                
                if (factor1 in correlation_matrix.index and 
                    factor2 in correlation_matrix.index):
                    corr = abs(correlation_matrix.loc[factor1, factor2])
                    
                    if corr > self.max_correlation:
                        # 移除评分较低的因子
                        if scores.get(factor1, 0) >= scores.get(factor2, 0):
                            to_remove.add(factor2)
                        else:
                            to_remove.add(factor1)
                            break  # factor1被移除，跳出内层循环
        
        return [name for name in factor_names if name not in to_remove]
    
    def _greedy_filter(self,
                       factors: Dict[str, pd.Series],
                       correlation_matrix: pd.DataFrame,
                       evaluation_results: Optional[Dict] = None) -> List[str]:
        """
        贪心筛选方法
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
        correlation_matrix : pd.DataFrame
            相关性矩阵
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        List[str]
            选中的因子名称
        """
        factor_names = list(factors.keys())
        
        # 获取因子评分
        scores = {}
        if evaluation_results:
            for name in factor_names:
                if name in evaluation_results:
                    result = evaluation_results[name]
                    if hasattr(result, 'total_score'):
                        scores[name] = result.total_score
                    else:
                        scores[name] = 0
                else:
                    scores[name] = 0
        else:
            scores = {name: 1.0 for name in factor_names}
        
        # 贪心选择
        selected = []
        remaining = factor_names.copy()
        
        while remaining:
            # 选择评分最高的因子
            best_factor = max(remaining, key=lambda x: scores.get(x, 0))
            selected.append(best_factor)
            remaining.remove(best_factor)
            
            # 移除与该因子高相关的其他因子
            to_remove = []
            for factor in remaining:
                if (best_factor in correlation_matrix.index and 
                    factor in correlation_matrix.index):
                    corr = abs(correlation_matrix.loc[best_factor, factor])
                    if corr > self.max_correlation:
                        to_remove.append(factor)
            
            for factor in to_remove:
                remaining.remove(factor)
        
        return selected
    
    def _cluster_filter(self,
                        factors: Dict[str, pd.Series],
                        correlation_matrix: pd.DataFrame,
                        evaluation_results: Optional[Dict] = None) -> List[str]:
        """
        聚类筛选方法
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典
        correlation_matrix : pd.DataFrame
            相关性矩阵
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        List[str]
            选中的因子名称
        """
        from sklearn.cluster import AgglomerativeClustering
        
        factor_names = list(factors.keys())
        
        # 将相关性转换为距离
        distance_matrix = 1 - np.abs(correlation_matrix.values)
        
        # 聚类
        n_clusters = max(1, int(len(factor_names) * (1 - self.max_correlation)))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 从每个聚类选择最佳因子
        selected = []
        scores = {}
        
        if evaluation_results:
            for name in factor_names:
                if name in evaluation_results:
                    result = evaluation_results[name]
                    if hasattr(result, 'total_score'):
                        scores[name] = result.total_score
                    else:
                        scores[name] = 0
                else:
                    scores[name] = 0
        else:
            scores = {name: 1.0 for name in factor_names}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_factors = [
                factor_names[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            # 选择该聚类中评分最高的因子
            best_factor = max(cluster_factors, key=lambda x: scores.get(x, 0))
            selected.append(best_factor)
        
        return selected
    
    def get_correlation_summary(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        获取相关性摘要
        
        Parameters
        ----------
        correlation_matrix : pd.DataFrame
            相关性矩阵
            
        Returns
        -------
        Dict[str, Any]
            相关性摘要
        """
        # 提取上三角矩阵的相关性值（排除对角线）
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        if len(correlations) == 0:
            return {
                'mean_correlation': 0,
                'max_correlation': 0,
                'high_correlation_pairs': 0
            }
        
        abs_correlations = np.abs(correlations)
        
        return {
            'mean_correlation': float(np.mean(abs_correlations)),
            'max_correlation': float(np.max(abs_correlations)),
            'min_correlation': float(np.min(abs_correlations)),
            'high_correlation_pairs': int(np.sum(abs_correlations > self.max_correlation)),
            'total_pairs': len(correlations)
        }