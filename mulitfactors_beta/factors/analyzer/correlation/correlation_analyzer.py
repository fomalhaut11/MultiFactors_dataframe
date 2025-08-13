"""
因子相关性分析器
计算因子间的相关性，识别冗余因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings

from ..base import AnalyzerBase, ComparativeAnalyzerMixin
from ...tester.base import TestResult

logger = logging.getLogger(__name__)


class CorrelationAnalyzer(AnalyzerBase, ComparativeAnalyzerMixin):
    """
    因子相关性分析器
    
    功能：
    1. 计算因子间相关性矩阵
    2. 支持多种相关性度量（Pearson、Spearman、Kendall）
    3. 识别高度相关的因子对
    4. 生成相关性热力图
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化相关性分析器
        
        Parameters
        ----------
        config : Dict, optional
            配置参数
        """
        super().__init__(name="CorrelationAnalyzer", config=config)
        
        # 相关性配置
        self.correlation_method = self.config.get('correlation_method', 'pearson')
        self.high_corr_threshold = self.config.get('high_corr_threshold', 0.7)
        self.min_periods = self.config.get('min_periods', 20)
        self.handle_missing = self.config.get('handle_missing', 'pairwise')
        
        # 结果缓存
        self.correlation_matrix = None
        self.high_corr_pairs = None
        
    def analyze(self, 
               data: Union[Dict[str, pd.Series], pd.DataFrame],
               methods: Optional[List[str]] = None,
               **kwargs) -> Dict[str, Any]:
        """
        执行相关性分析
        
        Parameters
        ----------
        data : Union[Dict[str, pd.Series], pd.DataFrame]
            因子数据，可以是：
            - Dict[factor_name, factor_series]: 因子字典
            - DataFrame: 列为因子的DataFrame
        methods : List[str], optional
            相关性计算方法列表，默认使用配置的方法
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            分析结果，包含：
            - correlation_matrix: 相关性矩阵
            - high_corr_pairs: 高相关因子对
            - redundant_factors: 冗余因子列表
            - statistics: 统计信息
        """
        self.analysis_time = datetime.now()
        
        # 预处理数据
        factor_df = self._prepare_factor_data(data)
        
        if factor_df.empty:
            logger.warning("No valid factor data for correlation analysis")
            return {}
        
        # 使用的相关性方法
        if methods is None:
            methods = [self.correlation_method]
        
        results = {
            'analysis_time': self.analysis_time,
            'factor_count': len(factor_df.columns),
            'sample_count': len(factor_df),
            'methods': methods
        }
        
        # 计算不同方法的相关性
        for method in methods:
            logger.info(f"Calculating {method} correlation")
            corr_matrix = self._calculate_correlation(factor_df, method)
            results[f'{method}_matrix'] = corr_matrix
            
            # 找出高相关因子对
            high_corr_pairs = self._find_high_correlation_pairs(corr_matrix)
            results[f'{method}_high_corr_pairs'] = high_corr_pairs
        
        # 使用主方法的结果
        self.correlation_matrix = results[f'{self.correlation_method}_matrix']
        self.high_corr_pairs = results[f'{self.correlation_method}_high_corr_pairs']
        
        # 识别冗余因子
        redundant_factors = self._identify_redundant_factors(
            self.correlation_matrix, 
            threshold=kwargs.get('redundancy_threshold', 0.9)
        )
        results['redundant_factors'] = redundant_factors
        
        # 计算统计信息
        results['statistics'] = self._calculate_statistics(self.correlation_matrix)
        
        # 生成摘要
        results['summary'] = self._generate_correlation_summary(results)
        
        logger.info(f"Correlation analysis completed: {len(factor_df.columns)} factors analyzed")
        
        return results
    
    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据
        
        Parameters
        ----------
        data : Any
            输入数据
            
        Returns
        -------
        bool
            数据是否有效
        """
        if isinstance(data, dict):
            # 检查是否为因子字典
            if not data:
                return False
            
            # 检查值是否为Series或DataFrame
            for key, value in data.items():
                if not isinstance(value, (pd.Series, pd.DataFrame)):
                    return False
            return True
            
        elif isinstance(data, pd.DataFrame):
            return not data.empty
        
        return False
    
    def _prepare_factor_data(self, 
                           data: Union[Dict[str, pd.Series], pd.DataFrame]) -> pd.DataFrame:
        """
        准备因子数据
        
        Parameters
        ----------
        data : Union[Dict[str, pd.Series], pd.DataFrame]
            原始数据
            
        Returns
        -------
        pd.DataFrame
            处理后的因子DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        
        # 从字典构建DataFrame
        factor_dict = {}
        
        for factor_name, factor_data in data.items():
            if isinstance(factor_data, pd.Series):
                factor_dict[factor_name] = factor_data
            elif isinstance(factor_data, TestResult):
                # 从TestResult提取处理后的因子
                if factor_data.processed_factor is not None:
                    factor_dict[factor_name] = factor_data.processed_factor
        
        if not factor_dict:
            return pd.DataFrame()
        
        # 对齐索引并合并
        factor_df = pd.DataFrame(factor_dict)
        
        # 处理缺失值
        if self.handle_missing == 'drop':
            factor_df = factor_df.dropna()
        elif self.handle_missing == 'fill':
            factor_df = factor_df.fillna(0)
        # pairwise会在相关性计算时处理
        
        return factor_df
    
    def _calculate_correlation(self, 
                              df: pd.DataFrame, 
                              method: str = 'pearson') -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Parameters
        ----------
        df : pd.DataFrame
            因子数据
        method : str
            相关性方法（pearson, spearman, kendall）
            
        Returns
        -------
        pd.DataFrame
            相关性矩阵
        """
        if method == 'pearson':
            corr_matrix = df.corr(method='pearson', min_periods=self.min_periods)
        elif method == 'spearman':
            corr_matrix = df.corr(method='spearman', min_periods=self.min_periods)
        elif method == 'kendall':
            corr_matrix = df.corr(method='kendall', min_periods=self.min_periods)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return corr_matrix
    
    def _find_high_correlation_pairs(self, 
                                    corr_matrix: pd.DataFrame,
                                    threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """
        找出高相关性的因子对
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关性矩阵
        threshold : float, optional
            相关性阈值
            
        Returns
        -------
        List[Tuple[str, str, float]]
            高相关因子对列表 [(factor1, factor2, correlation), ...]
        """
        if threshold is None:
            threshold = self.high_corr_threshold
        
        high_corr_pairs = []
        
        # 获取上三角矩阵的索引
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                factor1 = corr_matrix.columns[i]
                factor2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((factor1, factor2, corr_value))
        
        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def _identify_redundant_factors(self, 
                                   corr_matrix: pd.DataFrame,
                                   threshold: float = 0.9) -> List[str]:
        """
        识别冗余因子
        
        使用贪婪算法，保留相关性较低的因子集合
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关性矩阵
        threshold : float
            冗余阈值
            
        Returns
        -------
        List[str]
            建议移除的冗余因子列表
        """
        redundant = set()
        factors = list(corr_matrix.columns)
        
        for i, factor1 in enumerate(factors):
            if factor1 in redundant:
                continue
            
            for j, factor2 in enumerate(factors[i+1:], i+1):
                if factor2 in redundant:
                    continue
                
                if abs(corr_matrix.loc[factor1, factor2]) > threshold:
                    # 移除名称字母序较后的因子（或其他策略）
                    redundant.add(factor2)
        
        return list(redundant)
    
    def _calculate_statistics(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        计算相关性统计信息
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关性矩阵
            
        Returns
        -------
        Dict[str, float]
            统计信息
        """
        # 获取上三角矩阵的值（排除对角线）
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        corr_values = upper_triangle.stack().values
        
        statistics = {
            'mean_correlation': np.mean(corr_values),
            'median_correlation': np.median(corr_values),
            'std_correlation': np.std(corr_values),
            'max_correlation': np.max(corr_values),
            'min_correlation': np.min(corr_values),
            'high_corr_ratio': np.mean(np.abs(corr_values) > self.high_corr_threshold)
        }
        
        return statistics
    
    def _generate_correlation_summary(self, results: Dict[str, Any]) -> str:
        """
        生成相关性分析摘要
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析结果
            
        Returns
        -------
        str
            文本摘要
        """
        summary_lines = [
            f"Correlation Analysis Summary",
            f"=" * 50,
            f"Analysis Time: {results['analysis_time'].strftime('%Y-%m-%d %H:%M:%S')}",
            f"Factors Analyzed: {results['factor_count']}",
            f"Sample Size: {results['sample_count']}",
            f"Methods Used: {', '.join(results['methods'])}",
            f"",
            f"Statistics:",
        ]
        
        if 'statistics' in results:
            stats = results['statistics']
            summary_lines.extend([
                f"  Mean Correlation: {stats['mean_correlation']:.4f}",
                f"  Median Correlation: {stats['median_correlation']:.4f}",
                f"  Std Correlation: {stats['std_correlation']:.4f}",
                f"  Max Correlation: {stats['max_correlation']:.4f}",
                f"  Min Correlation: {stats['min_correlation']:.4f}",
                f"  High Correlation Ratio: {stats['high_corr_ratio']:.2%}",
            ])
        
        # 高相关因子对
        high_corr_pairs = results.get(f'{self.correlation_method}_high_corr_pairs', [])
        if high_corr_pairs:
            summary_lines.extend([
                f"",
                f"Top High Correlation Pairs (>{self.high_corr_threshold}):",
            ])
            for i, (f1, f2, corr) in enumerate(high_corr_pairs[:10], 1):
                summary_lines.append(f"  {i}. {f1} vs {f2}: {corr:.4f}")
        
        # 冗余因子
        redundant = results.get('redundant_factors', [])
        if redundant:
            summary_lines.extend([
                f"",
                f"Redundant Factors (suggested for removal):",
            ])
            for factor in redundant:
                summary_lines.append(f"  - {factor}")
        
        return "\n".join(summary_lines)
    
    def plot_correlation_heatmap(self, 
                                corr_matrix: Optional[pd.DataFrame] = None,
                                title: str = "Factor Correlation Heatmap",
                                figsize: Tuple[int, int] = (12, 10),
                                save_path: Optional[str] = None):
        """
        绘制相关性热力图
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame, optional
            相关性矩阵，默认使用最近的分析结果
        title : str
            图表标题
        figsize : Tuple[int, int]
            图表大小
        save_path : str, optional
            保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if corr_matrix is None:
                corr_matrix = self.correlation_matrix
            
            if corr_matrix is None:
                logger.warning("No correlation matrix available for plotting")
                return
            
            plt.figure(figsize=figsize)
            
            # 创建掩码，只显示下三角
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # 绘制热力图
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       fmt='.2f',
                       cmap='coolwarm',
                       center=0,
                       vmin=-1, 
                       vmax=1,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": 0.8})
            
            plt.title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation heatmap saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not installed, cannot plot heatmap")
    
    def get_factor_correlations(self, factor_name: str) -> pd.Series:
        """
        获取指定因子与其他因子的相关性
        
        Parameters
        ----------
        factor_name : str
            因子名称
            
        Returns
        -------
        pd.Series
            相关性序列
        """
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix available. Run analyze() first.")
        
        if factor_name not in self.correlation_matrix.columns:
            raise ValueError(f"Factor {factor_name} not found in correlation matrix")
        
        return self.correlation_matrix[factor_name].sort_values(ascending=False)