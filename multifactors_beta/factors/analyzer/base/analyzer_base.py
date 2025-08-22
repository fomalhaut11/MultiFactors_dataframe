"""
分析器基类
所有分析器的抽象基类，定义通用接口和功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from ...tester.base.test_result import TestResult
from ...utils.multiindex_helper import MultiIndexHelper

logger = logging.getLogger(__name__)


class AnalyzerBase(ABC):
    """
    分析器抽象基类
    
    定义所有分析器必须实现的接口
    提供通用的数据处理和验证功能
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        初始化分析器
        
        Parameters
        ----------
        name : str
            分析器名称
        config : Dict, optional
            配置参数
        """
        self.name = name
        self.config = config or {}
        self.results_cache = {}
        self.analysis_time = None
        
        # 配置参数
        self.min_samples = self.config.get('min_samples', 20)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.output_format = self.config.get('output_format', 'dataframe')
        
        logger.info(f"Initialized {self.name} analyzer")
    
    @abstractmethod
    def analyze(self, data: Union[Dict[str, TestResult], pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        执行分析
        
        Parameters
        ----------
        data : Union[Dict[str, TestResult], pd.DataFrame]
            输入数据，可以是测试结果字典或DataFrame
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            分析结果
        """
        pass
    
    @abstractmethod
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
        pass
    
    def preprocess_data(self, data: Union[Dict[str, TestResult], pd.DataFrame]) -> pd.DataFrame:
        """
        预处理数据
        
        Parameters
        ----------
        data : Union[Dict[str, TestResult], pd.DataFrame]
            原始数据
            
        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        if isinstance(data, dict):
            # 从TestResult字典提取数据
            processed_data = self._extract_from_test_results(data)
        elif isinstance(data, pd.DataFrame):
            processed_data = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # 确保数据格式正确
        if isinstance(processed_data, pd.Series) and not isinstance(processed_data.index, pd.MultiIndex):
            # 转换为MultiIndex格式
            processed_data = MultiIndexHelper.ensure_multiindex_format(processed_data)
        
        return processed_data
    
    def _extract_from_test_results(self, results_dict: Dict[str, TestResult]) -> pd.DataFrame:
        """
        从测试结果中提取数据
        
        Parameters
        ----------
        results_dict : Dict[str, TestResult]
            测试结果字典
            
        Returns
        -------
        pd.DataFrame
            提取的数据
        """
        data_list = []
        
        for factor_name, result in results_dict.items():
            if result is None or result.performance_metrics is None:
                continue
            
            row = {
                'factor': factor_name,
                'test_time': result.test_time,
                **result.performance_metrics
            }
            
            # 添加IC结果
            if result.ic_result:
                row.update({
                    'ic_mean': result.ic_result.ic_mean,
                    'ic_std': result.ic_result.ic_std,
                    'icir': result.ic_result.icir,
                    'rank_ic_mean': result.ic_result.rank_ic_mean
                })
            
            # 添加分组结果
            if result.group_result:
                row.update({
                    'monotonicity': result.group_result.monotonicity_score
                })
            
            # 添加换手率结果
            if result.turnover_result:
                row.update({
                    'avg_turnover': result.turnover_result.get('avg_turnover', 0),
                    'turnover_cost': result.turnover_result.get('avg_cost', 0)
                })
            
            data_list.append(row)
        
        return pd.DataFrame(data_list)
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None):
        """
        保存分析结果
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析结果
        output_path : str, optional
            输出路径
        """
        if output_path is None:
            output_path = f"analysis_results/{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据格式保存
        if output_path.suffix == '.csv' and 'dataframe' in results:
            results['dataframe'].to_csv(output_path)
        elif output_path.suffix == '.xlsx' and 'dataframe' in results:
            results['dataframe'].to_excel(output_path)
        else:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        
        logger.info(f"Saved analysis results to {output_path}")
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """
        生成分析摘要
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析结果
            
        Returns
        -------
        str
            文本摘要
        """
        summary = f"\n{'='*60}\n"
        summary += f"{self.name} Analysis Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Analysis Time: {self.analysis_time or datetime.now()}\n"
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                summary += f"{key}: {value:.4f}\n"
            elif isinstance(value, pd.DataFrame):
                summary += f"{key}: DataFrame with shape {value.shape}\n"
            elif isinstance(value, pd.Series):
                summary += f"{key}: Series with length {len(value)}\n"
        
        summary += f"{'='*60}\n"
        return summary
    
    def clear_cache(self):
        """清空缓存"""
        self.results_cache.clear()
        logger.info(f"Cleared cache for {self.name}")


class BatchAnalyzerMixin:
    """
    批量分析混入类
    提供批量处理多个因子的功能
    """
    
    def batch_analyze(self, 
                     data_dict: Dict[str, Any],
                     parallel: bool = False,
                     n_jobs: int = -1,
                     **kwargs) -> Dict[str, Dict]:
        """
        批量分析多个因子
        
        Parameters
        ----------
        data_dict : Dict[str, Any]
            因子数据字典
        parallel : bool
            是否并行处理
        n_jobs : int
            并行作业数，-1表示使用所有CPU
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Dict]
            分析结果字典
        """
        results = {}
        
        if parallel:
            from joblib import Parallel, delayed
            
            def analyze_single(factor_name, factor_data):
                return factor_name, self.analyze(factor_data, **kwargs)
            
            # 并行处理
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(analyze_single)(name, data) 
                for name, data in data_dict.items()
            )
            
            results = dict(parallel_results)
        else:
            # 串行处理
            for factor_name, factor_data in data_dict.items():
                try:
                    results[factor_name] = self.analyze(factor_data, **kwargs)
                    logger.info(f"Analyzed {factor_name}")
                except Exception as e:
                    logger.error(f"Failed to analyze {factor_name}: {e}")
                    results[factor_name] = None
        
        return results


class ComparativeAnalyzerMixin:
    """
    对比分析混入类
    提供因子间对比分析功能
    """
    
    def compare_factors(self,
                       factor_data: Dict[str, Any],
                       metrics: Optional[List[str]] = None,
                       sort_by: Optional[str] = None) -> pd.DataFrame:
        """
        对比多个因子
        
        Parameters
        ----------
        factor_data : Dict[str, Any]
            因子数据
        metrics : List[str], optional
            要对比的指标
        sort_by : str, optional
            排序依据
            
        Returns
        -------
        pd.DataFrame
            对比结果表
        """
        if metrics is None:
            metrics = ['ic_mean', 'icir', 'sharpe', 'turnover']
        
        comparison_data = []
        
        for factor_name, data in factor_data.items():
            if data is None:
                continue
            
            row = {'factor': factor_name}
            
            # 提取指标
            for metric in metrics:
                if isinstance(data, dict) and metric in data:
                    row[metric] = data[metric]
                elif hasattr(data, metric):
                    row[metric] = getattr(data, metric)
                else:
                    row[metric] = None
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 排序
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        return df
    
    def rank_factors(self,
                    comparison_df: pd.DataFrame,
                    weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        对因子进行排名
        
        Parameters
        ----------
        comparison_df : pd.DataFrame
            对比数据
        weights : Dict[str, float], optional
            指标权重
            
        Returns
        -------
        pd.DataFrame
            排名结果
        """
        if weights is None:
            weights = {
                'ic_mean': 0.3,
                'icir': 0.3,
                'sharpe': 0.2,
                'turnover': -0.2  # 换手率越低越好
            }
        
        # 计算综合得分
        score = pd.Series(0, index=comparison_df.index)
        
        for metric, weight in weights.items():
            if metric in comparison_df.columns:
                # 标准化
                values = comparison_df[metric].fillna(0)
                if values.std() > 0:
                    normalized = (values - values.mean()) / values.std()
                else:
                    normalized = values
                
                score += weight * normalized
        
        comparison_df['score'] = score
        comparison_df['rank'] = comparison_df['score'].rank(ascending=False)
        
        return comparison_df.sort_values('rank')