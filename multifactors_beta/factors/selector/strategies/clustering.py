"""
聚类选择策略

基于聚类方法选择因子
"""

from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..base.selector_base import SelectorBase

logger = logging.getLogger(__name__)


class ClusteringSelector(SelectorBase):
    """
    聚类选择器
    
    使用聚类方法将因子分组，从每组选择最佳因子
    """
    
    def __init__(self,
                 n_clusters: Optional[int] = None,
                 clustering_method: str = 'kmeans',
                 feature_selection: str = 'performance',
                 factors_per_cluster: int = 1,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化聚类选择器
        
        Parameters
        ----------
        n_clusters : int, optional
            聚类数量，如果为None则自动确定
        clustering_method : str
            聚类方法：'kmeans', 'hierarchical'
        feature_selection : str
            特征选择方法：'performance', 'correlation', 'mixed'
        factors_per_cluster : int
            每个聚类选择的因子数量
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.feature_selection = feature_selection
        self.factors_per_cluster = factors_per_cluster
        
        # 验证参数
        valid_methods = ['kmeans', 'hierarchical']
        if self.clustering_method not in valid_methods:
            raise ValueError(f"Invalid clustering_method: {clustering_method}. Must be one of {valid_methods}")
        
        valid_features = ['performance', 'correlation', 'mixed']
        if self.feature_selection not in valid_features:
            raise ValueError(f"Invalid feature_selection: {feature_selection}. Must be one of {valid_features}")
        
        if self.factors_per_cluster < 1:
            raise ValueError("factors_per_cluster must be at least 1")
        
        logger.info(
            f"Initialized ClusteringSelector: method={clustering_method}, "
            f"feature_selection={feature_selection}, factors_per_cluster={factors_per_cluster}"
        )
    
    def select(self,
               factors_pool: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               constraints: Optional[Dict] = None,
               **kwargs) -> Dict[str, Any]:
        """
        基于聚类选择因子
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict, optional
            评估结果
        constraints : Dict, optional
            选择约束
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        self.validate_factors(factors_pool)
        
        if len(factors_pool) < 2:
            logger.warning("Need at least 2 factors for clustering")
            return self._build_result(list(factors_pool.keys()), factors_pool, {}, evaluation_results)
        
        # 提取特征用于聚类
        features, factor_names = self._extract_features(factors_pool, evaluation_results)
        
        if features is None or len(features) < 2:
            logger.warning("Failed to extract features for clustering")
            return self._build_empty_result(factors_pool)
        
        # 执行聚类
        cluster_labels = self._perform_clustering(features, factor_names)
        
        # 从每个聚类选择最佳因子
        selected_names = self._select_from_clusters(
            cluster_labels, factor_names, evaluation_results, constraints
        )
        
        # 计算得分
        scores = self.score_factors(factors_pool, evaluation_results, **kwargs)
        
        # 构建结果
        result = self._build_result(
            selected_names, factors_pool, scores, evaluation_results,
            cluster_info={'labels': cluster_labels, 'n_clusters': len(np.unique(cluster_labels))}
        )
        
        # 记录历史
        self.save_selection_history(
            selected_names, scores, 'clustering',
            {
                'clustering_method': self.clustering_method,
                'n_clusters': len(np.unique(cluster_labels)),
                'factors_per_cluster': self.factors_per_cluster
            }
        )
        
        return result
    
    def score_factors(self,
                      factors_pool: Dict[str, pd.Series],
                      evaluation_results: Optional[Dict] = None,
                      **kwargs) -> Dict[str, float]:
        """
        为因子打分
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            因子得分字典
        """
        if not evaluation_results:
            return {name: 1.0 for name in factors_pool.keys()}
        
        scores = {}
        
        for name in factors_pool.keys():
            if name not in evaluation_results:
                scores[name] = 0.0
                continue
            
            result = evaluation_results[name]
            
            # 提取总分作为得分
            score = self._extract_score(result, 'total_score') or 0.0
            scores[name] = score
        
        return scores
    
    def _extract_features(self,
                          factors_pool: Dict[str, pd.Series],
                          evaluation_results: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        提取用于聚类的特征
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (特征矩阵, 因子名称列表)
        """
        factor_names = list(factors_pool.keys())
        
        if self.feature_selection == 'performance':
            return self._extract_performance_features(factor_names, evaluation_results)
        elif self.feature_selection == 'correlation':
            return self._extract_correlation_features(factors_pool)
        elif self.feature_selection == 'mixed':
            return self._extract_mixed_features(factors_pool, evaluation_results)
        else:
            return None, factor_names
    
    def _extract_performance_features(self,
                                      factor_names: List[str],
                                      evaluation_results: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        提取性能特征
        
        Parameters
        ----------
        factor_names : List[str]
            因子名称列表
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (特征矩阵, 有效因子名称列表)
        """
        if not evaluation_results:
            return None, factor_names
        
        features = []
        valid_names = []
        
        # 定义要提取的性能指标
        metrics = ['total_score', 'ic_mean', 'icir']
        
        for name in factor_names:
            if name not in evaluation_results:
                continue
            
            result = evaluation_results[name]
            factor_features = []
            
            # 提取各项指标
            for metric in metrics:
                value = self._extract_score(result, metric)
                if value is not None:
                    factor_features.append(value)
                else:
                    factor_features.append(0.0)
            
            if factor_features:
                features.append(factor_features)
                valid_names.append(name)
        
        if not features:
            return None, factor_names
        
        features_array = np.array(features)
        
        # 标准化特征
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        return features_normalized, valid_names
    
    def _extract_correlation_features(self,
                                      factors_pool: Dict[str, pd.Series]) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        提取相关性特征
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (特征矩阵, 因子名称列表)
        """
        try:
            # 计算因子相关性矩阵
            factor_names = list(factors_pool.keys())
            
            # 对齐所有因子数据
            common_index = None
            for factor in factors_pool.values():
                if common_index is None:
                    common_index = factor.index
                else:
                    common_index = common_index.intersection(factor.index)
            
            if len(common_index) == 0:
                return None, factor_names
            
            # 构建因子DataFrame
            factor_data = {}
            for name, factor in factors_pool.items():
                aligned_factor = factor.reindex(common_index).dropna()
                if len(aligned_factor) > 0:
                    factor_data[name] = aligned_factor
            
            if len(factor_data) < 2:
                return None, factor_names
            
            factor_df = pd.DataFrame(factor_data)
            correlation_matrix = factor_df.corr(method='spearman')
            
            # 使用相关性矩阵作为特征
            features = correlation_matrix.values
            valid_names = list(correlation_matrix.index)
            
            return features, valid_names
            
        except Exception as e:
            logger.error(f"Failed to extract correlation features: {e}")
            return None, list(factors_pool.keys())
    
    def _extract_mixed_features(self,
                                factors_pool: Dict[str, pd.Series],
                                evaluation_results: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        提取混合特征（性能+相关性）
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (特征矩阵, 因子名称列表)
        """
        # 获取性能特征
        perf_features, perf_names = self._extract_performance_features(
            list(factors_pool.keys()), evaluation_results
        )
        
        # 获取相关性特征
        corr_features, corr_names = self._extract_correlation_features(factors_pool)
        
        if perf_features is None and corr_features is None:
            return None, list(factors_pool.keys())
        
        # 找到共同的因子
        if perf_features is not None and corr_features is not None:
            common_names = list(set(perf_names) & set(corr_names))
            
            if not common_names:
                return perf_features, perf_names
            
            # 重新排列特征矩阵
            perf_indices = [perf_names.index(name) for name in common_names]
            corr_indices = [corr_names.index(name) for name in common_names]
            
            aligned_perf = perf_features[perf_indices]
            aligned_corr = corr_features[corr_indices]
            
            # 对相关性特征进行降维
            if aligned_corr.shape[1] > 3:
                pca = PCA(n_components=min(3, aligned_corr.shape[0]-1))
                aligned_corr = pca.fit_transform(aligned_corr)
            
            # 组合特征
            mixed_features = np.hstack([aligned_perf, aligned_corr])
            
            return mixed_features, common_names
        
        elif perf_features is not None:
            return perf_features, perf_names
        else:
            return corr_features, corr_names
    
    def _perform_clustering(self,
                            features: np.ndarray,
                            factor_names: List[str]) -> np.ndarray:
        """
        执行聚类
        
        Parameters
        ----------
        features : np.ndarray
            特征矩阵
        factor_names : List[str]
            因子名称列表
            
        Returns
        -------
        np.ndarray
            聚类标签
        """
        n_samples = features.shape[0]
        
        # 确定聚类数量
        if self.n_clusters is None:
            # 自动确定聚类数量（启发式方法）
            n_clusters = min(max(2, n_samples // 3), 8)
        else:
            n_clusters = min(self.n_clusters, n_samples)
        
        logger.info(f"Performing {self.clustering_method} clustering with {n_clusters} clusters")
        
        try:
            if self.clustering_method == 'kmeans':
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
            elif self.clustering_method == 'hierarchical':
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
            
            cluster_labels = clusterer.fit_predict(features)
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # 返回随机标签作为备选
            return np.random.randint(0, n_clusters, n_samples)
    
    def _select_from_clusters(self,
                              cluster_labels: np.ndarray,
                              factor_names: List[str],
                              evaluation_results: Optional[Dict] = None,
                              constraints: Optional[Dict] = None) -> List[str]:
        """
        从每个聚类选择最佳因子
        
        Parameters
        ----------
        cluster_labels : np.ndarray
            聚类标签
        factor_names : List[str]
            因子名称列表
        evaluation_results : Dict, optional
            评估结果
        constraints : Dict, optional
            约束条件
            
        Returns
        -------
        List[str]
            选中的因子名称
        """
        selected = []
        
        # 获取所有聚类ID
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            # 获取该聚类中的因子
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_factors = [factor_names[i] for i in cluster_indices]
            
            # 按评分排序选择最佳因子
            cluster_selected = self._select_best_from_cluster(
                cluster_factors, evaluation_results
            )
            
            selected.extend(cluster_selected)
        
        # 应用约束
        if constraints:
            selected = self._apply_constraints(selected, constraints, evaluation_results)
        
        logger.info(
            f"Selected {len(selected)} factors from {len(unique_clusters)} clusters"
        )
        
        return selected
    
    def _select_best_from_cluster(self,
                                  cluster_factors: List[str],
                                  evaluation_results: Optional[Dict] = None) -> List[str]:
        """
        从单个聚类中选择最佳因子
        
        Parameters
        ----------
        cluster_factors : List[str]
            聚类中的因子列表
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        List[str]
            选中的因子
        """
        if not evaluation_results:
            # 如果没有评估结果，随机选择
            n_select = min(self.factors_per_cluster, len(cluster_factors))
            return cluster_factors[:n_select]
        
        # 计算每个因子的得分
        factor_scores = []
        for name in cluster_factors:
            if name in evaluation_results:
                result = evaluation_results[name]
                score = self._extract_score(result, 'total_score') or 0.0
                factor_scores.append((score, name))
            else:
                factor_scores.append((0.0, name))
        
        # 按得分排序
        factor_scores.sort(reverse=True)
        
        # 选择最高分的因子
        n_select = min(self.factors_per_cluster, len(factor_scores))
        selected = [name for _, name in factor_scores[:n_select]]
        
        return selected
    
    def _extract_score(self, result: Any, metric_name: str) -> Optional[float]:
        """
        从评估结果中提取得分
        
        Parameters
        ----------
        result : Any
            评估结果对象
        metric_name : str
            指标名称
            
        Returns
        -------
        float or None
            得分值
        """
        try:
            # 直接属性访问
            if hasattr(result, metric_name):
                value = getattr(result, metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 从metrics字典获取
            if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                value = result.metrics.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            # 字典访问
            if isinstance(result, dict):
                value = result.get(metric_name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract {metric_name}: {e}")
            return None
    
    def _apply_constraints(self,
                           selected_factors: List[str],
                           constraints: Dict,
                           evaluation_results: Optional[Dict] = None) -> List[str]:
        """
        应用约束条件
        
        Parameters
        ----------
        selected_factors : List[str]
            已选中的因子
        constraints : Dict
            约束条件
        evaluation_results : Dict, optional
            评估结果
            
        Returns
        -------
        List[str]
            约束后的因子列表
        """
        filtered = selected_factors.copy()
        
        # 最大因子数约束
        max_factors = constraints.get('max_factors')
        if max_factors is not None and len(filtered) > max_factors:
            # 按总分排序，选择最高的
            if evaluation_results:
                factor_scores = []
                for name in filtered:
                    if name in evaluation_results:
                        result = evaluation_results[name]
                        score = self._extract_score(result, 'total_score') or 0
                        factor_scores.append((score, name))
                    else:
                        factor_scores.append((0, name))
                
                factor_scores.sort(reverse=True)
                filtered = [name for _, name in factor_scores[:max_factors]]
            else:
                filtered = filtered[:max_factors]
        
        # 其他约束
        blacklist = constraints.get('blacklist', [])
        if blacklist:
            filtered = [name for name in filtered if name not in blacklist]
        
        whitelist = constraints.get('whitelist')
        if whitelist:
            filtered = [name for name in filtered if name in whitelist]
        
        return filtered
    
    def _build_result(self,
                      selected_names: List[str],
                      factors_pool: Dict[str, pd.Series],
                      scores: Dict[str, float],
                      evaluation_results: Optional[Dict] = None,
                      cluster_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        构建选择结果
        
        Parameters
        ----------
        selected_names : List[str]
            选中的因子名称
        factors_pool : Dict[str, pd.Series]
            因子池
        scores : Dict[str, float]
            因子得分
        evaluation_results : Dict, optional
            评估结果
        cluster_info : Dict, optional
            聚类信息
            
        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        # 选中的因子数据
        selected_factors = {
            name: factors_pool[name] for name in selected_names
            if name in factors_pool
        }
        
        # 选择得分
        selection_scores = {
            name: scores.get(name, 0.0) for name in selected_names
        }
        
        # 选择原因
        selection_reasons = {
            name: f"Selected from cluster (score: {scores.get(name, 0.0):.2f})"
            for name in selected_names
        }
        
        # 未选中的因子及原因
        rejected_factors = {
            name: "Not selected from cluster"
            for name in factors_pool.keys()
            if name not in selected_names
        }
        
        result = {
            'selected_factors': selected_names,
            'factors_data': selected_factors,
            'selection_scores': selection_scores,
            'selection_reasons': selection_reasons,
            'rejected_factors': rejected_factors,
            'selection_method': 'clustering',
            'selection_params': {
                'clustering_method': self.clustering_method,
                'feature_selection': self.feature_selection,
                'factors_per_cluster': self.factors_per_cluster,
                'n_clusters': cluster_info.get('n_clusters') if cluster_info else None
            },
            'summary': {
                'total_candidates': len(factors_pool),
                'selected_count': len(selected_names),
                'clusters_formed': cluster_info.get('n_clusters') if cluster_info else 0,
                'avg_score': np.mean(list(selection_scores.values())) if selection_scores else 0
            }
        }
        
        # 添加聚类信息
        if cluster_info:
            result['cluster_info'] = cluster_info
        
        return result
    
    def _build_empty_result(self, factors_pool: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        构建空选择结果
        
        Parameters
        ----------
        factors_pool : Dict[str, pd.Series]
            因子池
            
        Returns
        -------
        Dict[str, Any]
            空结果
        """
        return {
            'selected_factors': [],
            'factors_data': {},
            'selection_scores': {},
            'selection_reasons': {},
            'rejected_factors': {name: "Clustering failed" for name in factors_pool.keys()},
            'selection_method': 'clustering',
            'selection_params': {
                'clustering_method': self.clustering_method,
                'feature_selection': self.feature_selection,
                'factors_per_cluster': self.factors_per_cluster
            },
            'summary': {
                'total_candidates': len(factors_pool),
                'selected_count': 0,
                'clusters_formed': 0,
                'avg_score': 0
            }
        }