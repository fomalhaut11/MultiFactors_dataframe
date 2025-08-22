"""
独特性维度评估
评估因子的独特性和信息增量
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import logging

from .base_dimension import BaseDimension, DimensionScore
from ....tester.base import TestResult
from ...correlation.correlation_analyzer import CorrelationAnalyzer

logger = logging.getLogger(__name__)


class UniquenessDimension(BaseDimension):
    """
    独特性维度
    
    评估指标：
    1. 因子相关性 - 与其他因子的相关程度
    2. 正交性 - 与现有因子的正交程度
    3. 独立贡献 - 独立于其他因子的收益贡献
    4. 信息增量 - 对组合的信息增量
    5. 冗余度 - 信息冗余程度
    """
    
    def __init__(self, weight: float = 0.10, config: Optional[Dict] = None):
        """
        初始化独特性维度
        
        Parameters
        ----------
        weight : float
            维度权重，默认10%
        config : Dict, optional
            配置参数
        """
        super().__init__(name="Uniqueness", weight=weight, config=config)
        
        # 相关性阈值
        self.correlation_thresholds = self.config.get('correlation_thresholds', {
            'excellent': 0.30,   # 与其他因子的最大相关性小于0.3
            'good': 0.50,
            'fair': 0.70,
            'poor': 0.85
        })
        
        # 平均相关性阈值
        self.avg_correlation_thresholds = self.config.get('avg_correlation_thresholds', {
            'excellent': 0.15,   # 平均相关性小于0.15
            'good': 0.25,
            'fair': 0.35,
            'poor': 0.50
        })
        
        # 独立贡献阈值（R²增量）
        self.contribution_thresholds = self.config.get('contribution_thresholds', {
            'excellent': 0.05,   # R²增量大于5%
            'good': 0.03,
            'fair': 0.01,
            'poor': 0.005
        })
        
        # 信息比率增量阈值
        self.ir_increment_thresholds = self.config.get('ir_increment_thresholds', {
            'excellent': 0.10,   # IR增量大于0.1
            'good': 0.05,
            'fair': 0.02,
            'poor': 0.01
        })
        
        # 初始化相关性分析器
        self.correlation_analyzer = CorrelationAnalyzer(config=config)
    
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算独特性维度得分
        
        Parameters
        ----------
        data : Dict[str, Any]
            包含TestResult、相关性分析结果等数据
            
        Returns
        -------
        DimensionScore
            维度评分结果
        """
        if not self.validate_data(data):
            return self._create_empty_score()
        
        # 提取指标
        metrics = self.extract_metrics(data)
        
        # 计算各指标得分
        scores = {}
        
        # 最大相关性得分
        max_correlation = metrics.get('max_correlation', 1.0)
        scores['max_correlation_score'] = self._score_max_correlation(max_correlation)
        
        # 平均相关性得分
        avg_correlation = metrics.get('avg_correlation', 1.0)
        scores['avg_correlation_score'] = self._score_avg_correlation(avg_correlation)
        
        # 独立贡献得分
        independent_contribution = metrics.get('independent_contribution', 0)
        scores['contribution_score'] = self._score_contribution(independent_contribution)
        
        # 正交性得分
        orthogonality = metrics.get('orthogonality', 0)
        scores['orthogonality_score'] = self._score_orthogonality(orthogonality)
        
        # 信息增量得分
        info_increment = metrics.get('info_increment', 0)
        scores['info_increment_score'] = self._score_info_increment(info_increment)
        
        # 计算加权总分
        total_score = self.calculate_weighted_score(scores)
        
        # 获取等级和描述
        grade = self.get_grade(total_score)
        description = self._generate_description(total_score, metrics)
        
        return DimensionScore(
            dimension_name=self.name,
            score=total_score,
            weight=self.weight,
            metrics=metrics,
            grade=grade,
            description=description
        )
    
    def extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        从数据中提取独特性相关指标
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据
            
        Returns
        -------
        Dict[str, float]
            提取的指标
        """
        metrics = {}
        
        # 从相关性分析结果提取
        correlation_result = data.get('correlation_analysis')
        if correlation_result:
            # 因子相关性矩阵
            correlation_matrix = correlation_result.get('correlation_matrix')
            if correlation_matrix is not None and not correlation_matrix.empty:
                # 当前因子名称
                current_factor = data.get('factor_name', 'current_factor')
                
                if current_factor in correlation_matrix.columns:
                    # 获取当前因子与其他因子的相关性
                    factor_correlations = correlation_matrix[current_factor].drop(current_factor, errors='ignore')
                    
                    if len(factor_correlations) > 0:
                        # 最大相关性（绝对值）
                        metrics['max_correlation'] = factor_correlations.abs().max()
                        # 平均相关性（绝对值）
                        metrics['avg_correlation'] = factor_correlations.abs().mean()
                        # 相关性标准差
                        metrics['correlation_std'] = factor_correlations.abs().std()
                        # 高相关因子数量（绝对值>0.7）
                        metrics['high_corr_count'] = (factor_correlations.abs() > 0.7).sum()
                        # 中等相关因子数量（0.3<绝对值<=0.7）
                        metrics['medium_corr_count'] = ((factor_correlations.abs() > 0.3) & 
                                                        (factor_correlations.abs() <= 0.7)).sum()
                else:
                    # 如果没有当前因子，使用整体相关性统计
                    corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
                    if len(corr_values) > 0:
                        metrics['max_correlation'] = np.abs(corr_values).max()
                        metrics['avg_correlation'] = np.abs(corr_values).mean()
            
            # 聚类结果
            clustering_result = correlation_result.get('clustering')
            if clustering_result:
                # 聚类数量（越多说明因子越分散）
                metrics['n_clusters'] = clustering_result.get('n_clusters', 1)
                # 当前因子是否为独立聚类
                cluster_labels = clustering_result.get('cluster_labels')
                if cluster_labels is not None:
                    current_factor = data.get('factor_name')
                    if current_factor and current_factor in cluster_labels:
                        current_cluster = cluster_labels[current_factor]
                        cluster_size = (cluster_labels == current_cluster).sum()
                        metrics['cluster_size'] = cluster_size
                        metrics['is_singleton_cluster'] = (cluster_size == 1)
        
        # 从TestResult提取
        test_result = data.get('test_result')
        if isinstance(test_result, TestResult):
            # 从回归结果提取独立贡献
            if test_result.regression_result:
                reg_result = test_result.regression_result
                
                # R²值（因子的解释力）
                if hasattr(reg_result, 'rsquared'):
                    metrics['rsquared'] = reg_result.rsquared
                
                # 如果有多因子回归结果
                if hasattr(reg_result, 'incremental_rsquared'):
                    metrics['independent_contribution'] = reg_result.incremental_rsquared
                elif hasattr(reg_result, 'rsquared'):
                    # 单因子R²作为独立贡献的近似
                    metrics['independent_contribution'] = reg_result.rsquared
            
            # 从性能指标提取
            if test_result.performance_metrics:
                perf = test_result.performance_metrics
                
                # 信息比率（可以反映独特性）
                if 'information_ratio' in perf:
                    metrics['information_ratio'] = perf['information_ratio']
                elif 'icir' in perf:
                    metrics['information_ratio'] = perf['icir']
        
        # 从其他因子对比结果提取（如果有）
        benchmark_factors = data.get('benchmark_factors')
        if benchmark_factors:
            # 与基准因子的相关性
            benchmark_correlations = []
            current_factor_data = data.get('factor_data')
            
            if current_factor_data is not None:
                for factor_name, factor_data in benchmark_factors.items():
                    if isinstance(factor_data, pd.Series) and isinstance(current_factor_data, pd.Series):
                        # 对齐索引
                        aligned_current = current_factor_data.reindex(factor_data.index)
                        valid_idx = aligned_current.notna() & factor_data.notna()
                        if valid_idx.sum() > 100:  # 至少100个有效数据点
                            corr = aligned_current[valid_idx].corr(factor_data[valid_idx])
                            benchmark_correlations.append(abs(corr))
            
            if benchmark_correlations:
                metrics['benchmark_max_corr'] = max(benchmark_correlations)
                metrics['benchmark_avg_corr'] = np.mean(benchmark_correlations)
        
        # 计算正交性得分（1 - 平均相关性）
        avg_corr = metrics.get('avg_correlation', 0.5)
        metrics['orthogonality'] = max(0, 1 - avg_corr)
        
        # 估算信息增量
        if 'information_ratio' in metrics:
            # 基于IR估算信息增量
            ir = metrics['information_ratio']
            # 假设基准组合IR为0.5
            baseline_ir = 0.5
            metrics['info_increment'] = max(0, ir - baseline_ir)
        elif 'independent_contribution' in metrics:
            # 基于R²增量估算
            metrics['info_increment'] = metrics['independent_contribution']
        else:
            # 基于相关性估算（低相关性意味着高信息增量）
            metrics['info_increment'] = metrics.get('orthogonality', 0) * 0.05
        
        # 计算冗余度（高相关性意味着高冗余）
        metrics['redundancy'] = min(1.0, avg_corr)
        
        return metrics
    
    def _score_max_correlation(self, correlation: float) -> float:
        """
        计算最大相关性得分
        
        Parameters
        ----------
        correlation : float
            最大相关性（绝对值）
            
        Returns
        -------
        float
            最大相关性得分(0-100)
        """
        # 相关性越低越好
        if correlation <= self.correlation_thresholds['excellent']:
            return 100
        elif correlation <= self.correlation_thresholds['good']:
            return 80 + (self.correlation_thresholds['good'] - correlation) / \
                   (self.correlation_thresholds['good'] - self.correlation_thresholds['excellent']) * 20
        elif correlation <= self.correlation_thresholds['fair']:
            return 60 + (self.correlation_thresholds['fair'] - correlation) / \
                   (self.correlation_thresholds['fair'] - self.correlation_thresholds['good']) * 20
        elif correlation <= self.correlation_thresholds['poor']:
            return 40 + (self.correlation_thresholds['poor'] - correlation) / \
                   (self.correlation_thresholds['poor'] - self.correlation_thresholds['fair']) * 20
        else:
            return max(0, 40 * (1 - correlation))
    
    def _score_avg_correlation(self, correlation: float) -> float:
        """
        计算平均相关性得分
        
        Parameters
        ----------
        correlation : float
            平均相关性（绝对值）
            
        Returns
        -------
        float
            平均相关性得分(0-100)
        """
        # 平均相关性越低越好
        if correlation <= self.avg_correlation_thresholds['excellent']:
            return 100
        elif correlation <= self.avg_correlation_thresholds['good']:
            return 80 + (self.avg_correlation_thresholds['good'] - correlation) / \
                   (self.avg_correlation_thresholds['good'] - self.avg_correlation_thresholds['excellent']) * 20
        elif correlation <= self.avg_correlation_thresholds['fair']:
            return 60 + (self.avg_correlation_thresholds['fair'] - correlation) / \
                   (self.avg_correlation_thresholds['fair'] - self.avg_correlation_thresholds['good']) * 20
        elif correlation <= self.avg_correlation_thresholds['poor']:
            return 40 + (self.avg_correlation_thresholds['poor'] - correlation) / \
                   (self.avg_correlation_thresholds['poor'] - self.avg_correlation_thresholds['fair']) * 20
        else:
            return max(0, 40 * (1 - correlation))
    
    def _score_contribution(self, contribution: float) -> float:
        """
        计算独立贡献得分
        
        Parameters
        ----------
        contribution : float
            独立贡献（R²增量）
            
        Returns
        -------
        float
            贡献得分(0-100)
        """
        # 贡献越大越好
        if contribution >= self.contribution_thresholds['excellent']:
            return 100
        elif contribution >= self.contribution_thresholds['good']:
            return 80 + (contribution - self.contribution_thresholds['good']) / \
                   (self.contribution_thresholds['excellent'] - self.contribution_thresholds['good']) * 20
        elif contribution >= self.contribution_thresholds['fair']:
            return 60 + (contribution - self.contribution_thresholds['fair']) / \
                   (self.contribution_thresholds['good'] - self.contribution_thresholds['fair']) * 20
        elif contribution >= self.contribution_thresholds['poor']:
            return 40 + (contribution - self.contribution_thresholds['poor']) / \
                   (self.contribution_thresholds['fair'] - self.contribution_thresholds['poor']) * 20
        elif contribution > 0:
            return contribution / self.contribution_thresholds['poor'] * 40
        else:
            return 0
    
    def _score_orthogonality(self, orthogonality: float) -> float:
        """
        计算正交性得分
        
        Parameters
        ----------
        orthogonality : float
            正交性指标(0-1)
            
        Returns
        -------
        float
            正交性得分(0-100)
        """
        # 直接映射到0-100分
        return orthogonality * 100
    
    def _score_info_increment(self, increment: float) -> float:
        """
        计算信息增量得分
        
        Parameters
        ----------
        increment : float
            信息增量
            
        Returns
        -------
        float
            信息增量得分(0-100)
        """
        # 信息增量越大越好
        if increment >= self.ir_increment_thresholds['excellent']:
            return 100
        elif increment >= self.ir_increment_thresholds['good']:
            return 80 + (increment - self.ir_increment_thresholds['good']) / \
                   (self.ir_increment_thresholds['excellent'] - self.ir_increment_thresholds['good']) * 20
        elif increment >= self.ir_increment_thresholds['fair']:
            return 60 + (increment - self.ir_increment_thresholds['fair']) / \
                   (self.ir_increment_thresholds['good'] - self.ir_increment_thresholds['fair']) * 20
        elif increment >= self.ir_increment_thresholds['poor']:
            return 40 + (increment - self.ir_increment_thresholds['poor']) / \
                   (self.ir_increment_thresholds['fair'] - self.ir_increment_thresholds['poor']) * 20
        elif increment > 0:
            return increment / self.ir_increment_thresholds['poor'] * 40
        else:
            return 0
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """
        获取默认评分标准
        
        Returns
        -------
        Dict[str, Any]
            默认评分标准
        """
        return {
            'correlation_thresholds': self.correlation_thresholds,
            'avg_correlation_thresholds': self.avg_correlation_thresholds,
            'contribution_thresholds': self.contribution_thresholds,
            'ir_increment_thresholds': self.ir_increment_thresholds
        }
    
    def _get_default_weights(self) -> Dict[str, float]:
        """
        获取默认指标权重
        
        Returns
        -------
        Dict[str, float]
            默认指标权重
        """
        return {
            'max_correlation_score': 0.25,      # 最大相关性权重25%
            'avg_correlation_score': 0.25,      # 平均相关性权重25%
            'contribution_score': 0.20,         # 独立贡献权重20%
            'orthogonality_score': 0.15,        # 正交性权重15%
            'info_increment_score': 0.15        # 信息增量权重15%
        }
    
    def _generate_description(self, score: float, metrics: Dict[str, float]) -> str:
        """
        生成独特性评价描述
        
        Parameters
        ----------
        score : float
            总分
        metrics : Dict[str, float]
            详细指标
            
        Returns
        -------
        str
            评价描述
        """
        grade = self.get_grade(score)
        
        max_corr = metrics.get('max_correlation', 1.0)
        avg_corr = metrics.get('avg_correlation', 1.0)
        contribution = metrics.get('independent_contribution', 0)
        orthogonality = metrics.get('orthogonality', 0)
        
        description = f"独特性维度得分{score:.1f}分(等级{grade})。"
        
        # 相关性评价
        if max_corr < self.correlation_thresholds['excellent']:
            description += f"与其他因子的最大相关性({max_corr:.2f})很低，独特性强。"
        elif max_corr < self.correlation_thresholds['good']:
            description += f"与其他因子的最大相关性({max_corr:.2f})较低。"
        elif max_corr > self.correlation_thresholds['fair']:
            description += f"与其他因子的最大相关性({max_corr:.2f})较高，独特性不足。"
        
        # 平均相关性评价
        if avg_corr < self.avg_correlation_thresholds['excellent']:
            description += f"平均相关性({avg_corr:.2f})很低。"
        elif avg_corr > self.avg_correlation_thresholds['fair']:
            description += f"平均相关性({avg_corr:.2f})偏高。"
        
        # 独立贡献评价
        if contribution > self.contribution_thresholds['good']:
            description += f"独立贡献({contribution:.1%})显著。"
        elif contribution < self.contribution_thresholds['poor']:
            description += f"独立贡献({contribution:.1%})较小。"
        
        # 正交性评价
        if orthogonality > 0.7:
            description += "与现有因子正交性良好。"
        
        # 冗余度评价
        redundancy = metrics.get('redundancy', 0)
        if redundancy > 0.7:
            description += "信息冗余度较高，建议考虑因子组合优化。"
        
        return description
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        验证输入数据是否有效
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据
            
        Returns
        -------
        bool
            数据是否有效
        """
        if not super().validate_data(data):
            return False
        
        # 至少需要TestResult或相关性分析结果之一
        test_result = data.get('test_result')
        correlation_result = data.get('correlation_analysis')
        
        if not isinstance(test_result, TestResult) and not correlation_result:
            logger.warning("No TestResult or correlation analysis found in data")
            return False
        
        return True
    
    def _create_empty_score(self) -> DimensionScore:
        """创建空的评分结果"""
        return DimensionScore(
            dimension_name=self.name,
            score=0,
            weight=self.weight,
            metrics={},
            grade='F',
            description="无法计算独特性得分，数据不足"
        )