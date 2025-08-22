"""
因子筛选器
从磁盘加载历史测试结果进行分析和筛选
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

from factors.tester.core import ResultManager
from factors.tester.base import TestResult
from core.config_manager import get_config
from ..config import get_analyzer_config

logger = logging.getLogger(__name__)


class FactorScreener:
    """
    因子筛选器 - 从磁盘加载历史测试结果进行分析
    
    功能：
    1. 加载和管理历史测试结果
    2. 根据多维标准筛选因子
    3. 因子排名和评分
    4. 生成筛选报告
    """
    
    def __init__(self, test_data_path: Optional[str] = None, config_override: Optional[Dict] = None):
        """
        初始化筛选器
        
        Parameters:
        -----------
        test_data_path : str, optional
            测试结果路径，默认从config读取
        config_override : Dict, optional
            配置覆盖参数
        """
        # 初始化配置
        self.config = get_analyzer_config(config_override)
        
        # 初始化结果管理器
        self.result_manager = ResultManager(test_data_path)
        
        # 缓存
        self.test_results_cache = {}  # 缓存已加载的结果
        self.summary_df = None  # 汇总表缓存
        
        logger.info("FactorScreener initialized with path: %s", self.result_manager.base_path)
    
    def load_all_results(self, 
                        date_range: Optional[Tuple[str, str]] = None,
                        factor_names: Optional[List[str]] = None,
                        force_reload: bool = False) -> Dict[str, TestResult]:
        """
        从磁盘加载所有测试结果
        
        Parameters:
        -----------
        date_range : Tuple[str, str], optional
            日期范围过滤 (start_date, end_date)
        factor_names : List[str], optional
            因子名称过滤
        force_reload : bool
            是否强制重新加载
            
        Returns:
        --------
        Dict[factor_name, TestResult]
            因子测试结果字典
        """
        if not force_reload and self.test_results_cache:
            logger.info("Using cached results, %d factors loaded", len(self.test_results_cache))
            return self.test_results_cache
        
        logger.info("Loading test results from disk...")
        
        # 批量加载
        all_results = self.result_manager.load_batch()
        logger.info("Loaded %d test results", len(all_results))
        
        # 过滤和整理
        results_dict = {}
        
        for result in all_results:
            # 日期过滤
            if date_range:
                test_date = result.test_time.strftime('%Y%m%d')
                if test_date < date_range[0] or test_date > date_range[1]:
                    continue
            
            # 因子名称过滤
            if factor_names and result.factor_name not in factor_names:
                continue
            
            # 保留最新的测试结果
            if result.factor_name not in results_dict:
                results_dict[result.factor_name] = result
            elif result.test_time > results_dict[result.factor_name].test_time:
                results_dict[result.factor_name] = result
        
        self.test_results_cache = results_dict
        self.summary_df = None  # 清空汇总表缓存
        
        logger.info("Filtered to %d unique factors", len(results_dict))
        return results_dict
    
    def screen_factors(self, 
                      criteria: Optional[Dict[str, float]] = None,
                      preset: Optional[str] = None,
                      load_fresh: bool = False) -> List[str]:
        """
        根据标准筛选因子
        
        Parameters:
        -----------
        criteria : Dict[str, float], optional
            筛选标准，如：
            {
                'ic_mean_min': 0.02,
                'icir_min': 0.5,
                'monotonicity_min': 0.6,
                'sharpe_min': 1.0,
                't_value_min': 2.0
            }
        preset : str, optional
            预设标准 ('strict', 'normal', 'loose')
        load_fresh : bool
            是否重新加载数据
            
        Returns:
        --------
        List[str]
            符合标准的因子名称列表
        """
        # 加载或使用缓存
        if load_fresh or not self.test_results_cache:
            self.load_all_results()
        
        # 确定筛选标准
        if preset:
            criteria = self.config.get_screening_criteria(preset)
        elif criteria is None:
            criteria = self.config.screening
        
        logger.info("Screening with criteria: %s", criteria)
        
        selected_factors = []
        screening_details = []
        
        for factor_name, result in self.test_results_cache.items():
            # 检查是否有错误
            if result.errors:
                logger.debug("Factor %s has errors, skipping", factor_name)
                continue
            
            # 检查各项指标
            meets_criteria = True
            factor_metrics = {'factor_name': factor_name}
            
            # IC指标
            if 'ic_mean_min' in criteria and result.ic_result:
                ic_mean = result.ic_result.ic_mean
                factor_metrics['ic_mean'] = ic_mean
                if ic_mean < criteria['ic_mean_min']:
                    meets_criteria = False
            
            if 'icir_min' in criteria and result.ic_result:
                icir = result.ic_result.icir
                factor_metrics['icir'] = icir
                if icir < criteria['icir_min']:
                    meets_criteria = False
            
            # 单调性
            if 'monotonicity_min' in criteria and result.group_result:
                monotonicity = result.group_result.monotonicity_score
                factor_metrics['monotonicity'] = monotonicity
                if monotonicity < criteria['monotonicity_min']:
                    meets_criteria = False
            
            # 夏普比率
            if 'sharpe_min' in criteria:
                sharpe = result.performance_metrics.get('long_short_sharpe', 0)
                factor_metrics['sharpe'] = sharpe
                if sharpe < criteria['sharpe_min']:
                    meets_criteria = False
            
            # t值
            if 't_value_min' in criteria:
                t_value = abs(result.performance_metrics.get('t_value_mean', 0))
                factor_metrics['t_value'] = t_value
                if t_value < criteria['t_value_min']:
                    meets_criteria = False
            
            # 最大回撤
            if 'max_drawdown_limit' in criteria and result.group_result:
                # 计算最大回撤
                cumulative_returns = result.group_result.long_short_return.cumsum()
                drawdown = (cumulative_returns - cumulative_returns.expanding().max()).min()
                factor_metrics['max_drawdown'] = abs(drawdown)
                if abs(drawdown) > criteria['max_drawdown_limit']:
                    meets_criteria = False
            
            factor_metrics['selected'] = meets_criteria
            screening_details.append(factor_metrics)
            
            if meets_criteria:
                selected_factors.append(factor_name)
        
        # 保存筛选详情
        self.screening_details = pd.DataFrame(screening_details)
        
        logger.info("Selected %d factors out of %d", len(selected_factors), len(self.test_results_cache))
        return selected_factors
    
    def get_factor_ranking(self, 
                          metric: str = 'icir',
                          top_n: Optional[int] = None,
                          ascending: bool = False) -> pd.DataFrame:
        """
        获取因子排名
        
        Parameters:
        -----------
        metric : str
            排序指标 ('icir', 'ic_mean', 'sharpe', 'monotonicity')
        top_n : int, optional
            返回前N个
        ascending : bool
            是否升序排列
            
        Returns:
        --------
        pd.DataFrame
            排名DataFrame
        """
        if self.summary_df is None or self.summary_df.empty:
            # 生成汇总表
            self.summary_df = self.result_manager.get_summary_table(
                list(self.test_results_cache.values())
            )
        
        # 获取实际的列名
        metric_mapping = self.config.METRICS_MAPPING
        actual_metric = metric_mapping.get(metric, metric)
        
        # 检查列是否存在
        if actual_metric not in self.summary_df.columns:
            logger.warning("Metric %s not found in summary table", actual_metric)
            available_metrics = [col for col in self.summary_df.columns 
                               if col not in ['factor_name', 'test_id', 'test_time']]
            logger.info("Available metrics: %s", available_metrics)
            return self.summary_df
        
        # 排序
        df = self.summary_df.sort_values(actual_metric, ascending=ascending)
        
        if top_n:
            df = df.head(top_n)
        
        # 添加排名列
        df['rank'] = range(1, len(df) + 1)
        
        # 选择关键列显示
        display_columns = ['rank', 'factor_name', actual_metric]
        
        # 添加其他重要指标
        important_metrics = ['ic_mean', 'icir', 'monotonicity_score', 'long_short_sharpe']
        for col in important_metrics:
            if col in df.columns and col != actual_metric:
                display_columns.append(col)
        
        return df[display_columns]
    
    def analyze_factor_stability(self, 
                                factor_name: str,
                                lookback_days: Optional[int] = None) -> Dict[str, Any]:
        """
        分析因子在不同时期的稳定性
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        lookback_days : int, optional
            回望天数，默认从配置读取
            
        Returns:
        --------
        Dict
            稳定性分析结果
        """
        if lookback_days is None:
            lookback_days = self.config.stability.get('lookback_window', 30)
        
        # 加载该因子的所有历史测试结果
        all_results = []
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        for root, dirs, files in os.walk(self.result_manager.base_path):
            for file in files:
                if factor_name in file and file.endswith('.pkl') and 'config' not in file:
                    try:
                        filepath = os.path.join(root, file)
                        # 检查文件时间
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_time >= cutoff_date:
                            result = self.result_manager.load(filepath)
                            if result.factor_name == factor_name:
                                all_results.append(result)
                    except Exception as e:
                        logger.debug("Failed to load %s: %s", file, e)
                        continue
        
        if not all_results:
            logger.warning("No historical results found for factor %s", factor_name)
            return {'error': f'No results found for {factor_name}'}
        
        # 按时间排序
        all_results.sort(key=lambda x: x.test_time)
        
        # 提取关键指标
        ic_values = []
        icir_values = []
        sharpe_values = []
        test_dates = []
        
        for result in all_results:
            test_dates.append(result.test_time)
            
            if result.ic_result:
                ic_values.append(result.ic_result.ic_mean)
                icir_values.append(result.ic_result.icir)
            
            if result.performance_metrics:
                sharpe_values.append(result.performance_metrics.get('long_short_sharpe', np.nan))
        
        # 计算稳定性指标
        stability_metrics = {
            'factor_name': factor_name,
            'test_count': len(all_results),
            'date_range': [test_dates[0].strftime('%Y-%m-%d'), 
                          test_dates[-1].strftime('%Y-%m-%d')] if test_dates else None,
            'ic_mean': np.mean(ic_values) if ic_values else None,
            'ic_std': np.std(ic_values) if ic_values else None,
            'ic_stability': np.mean(ic_values) / np.std(ic_values) 
                           if ic_values and np.std(ic_values) > 0 else None,
            'icir_mean': np.mean(icir_values) if icir_values else None,
            'icir_std': np.std(icir_values) if icir_values else None,
            'sharpe_mean': np.nanmean(sharpe_values) if sharpe_values else None,
            'sharpe_std': np.nanstd(sharpe_values) if sharpe_values else None,
            'latest_test': test_dates[-1] if test_dates else None,
            'historical_ic': ic_values,
            'historical_dates': [d.strftime('%Y-%m-%d') for d in test_dates]
        }
        
        # 判断稳定性等级
        if stability_metrics['ic_stability']:
            if stability_metrics['ic_stability'] > 3:
                stability_metrics['stability_grade'] = 'Excellent'
            elif stability_metrics['ic_stability'] > 2:
                stability_metrics['stability_grade'] = 'Good'
            elif stability_metrics['ic_stability'] > 1:
                stability_metrics['stability_grade'] = 'Fair'
            else:
                stability_metrics['stability_grade'] = 'Poor'
        
        return stability_metrics
    
    def get_factor_score(self, factor_name: str) -> float:
        """
        计算因子综合评分
        
        Parameters:
        -----------
        factor_name : str
            因子名称
            
        Returns:
        --------
        float
            综合评分 (0-100)
        """
        if factor_name not in self.test_results_cache:
            logger.warning("Factor %s not in cache", factor_name)
            return 0.0
        
        result = self.test_results_cache[factor_name]
        weights = self.config.SCORING_WEIGHTS
        
        score = 0.0
        
        # IC得分
        if result.ic_result:
            ic_score = min(abs(result.ic_result.ic_mean) / 0.05 * 100, 100)
            score += ic_score * weights.get('ic_score', 0.25)
        
        # 稳定性得分
        stability = self.analyze_factor_stability(factor_name)
        if stability.get('ic_stability'):
            stability_score = min(stability['ic_stability'] / 3 * 100, 100)
            score += stability_score * weights.get('stability_score', 0.20)
        
        # 单调性得分
        if result.group_result:
            monotonicity_score = result.group_result.monotonicity_score * 100
            score += monotonicity_score * weights.get('monotonicity_score', 0.20)
        
        # 夏普得分
        if result.performance_metrics:
            sharpe = result.performance_metrics.get('long_short_sharpe', 0)
            sharpe_score = min(sharpe / 2 * 100, 100)
            score += sharpe_score * weights.get('sharpe_score', 0.20)
        
        return round(score, 2)
    
    def generate_screening_report(self, 
                                  output_path: Optional[str] = None,
                                  top_n: int = 20) -> pd.DataFrame:
        """
        生成筛选报告
        
        Parameters:
        -----------
        output_path : str, optional
            输出路径
        top_n : int
            显示前N个因子
            
        Returns:
        --------
        pd.DataFrame
            筛选报告表
        """
        if not self.test_results_cache:
            self.load_all_results()
        
        report_data = []
        
        for factor_name, result in self.test_results_cache.items():
            if result.errors:
                continue
            
            # 基本信息
            row = {
                'factor_name': factor_name,
                'test_date': result.test_time.strftime('%Y-%m-%d'),
                'sample_count': result.data_info.get('factor_count', 0),
            }
            
            # 性能指标
            if result.ic_result:
                row['ic_mean'] = result.ic_result.ic_mean
                row['icir'] = result.ic_result.icir
                row['rank_ic'] = result.ic_result.rank_ic_mean
            
            if result.group_result:
                row['monotonicity'] = result.group_result.monotonicity_score
            
            if result.performance_metrics:
                row['sharpe'] = result.performance_metrics.get('long_short_sharpe', 0)
                row['annual_return'] = result.performance_metrics.get('long_short_annual_return', 0)
            
            # 综合评分
            row['score'] = self.get_factor_score(factor_name)
            
            # 稳定性
            stability = self.analyze_factor_stability(factor_name)
            row['stability_grade'] = stability.get('stability_grade', 'Unknown')
            
            report_data.append(row)
        
        # 创建报告DataFrame
        report_df = pd.DataFrame(report_data)
        
        # 排序
        report_df = report_df.sort_values('score', ascending=False).head(top_n)
        
        # 保存报告
        if output_path:
            report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info("Report saved to %s", output_path)
        
        return report_df
    
    def compare_factors(self, factor_names: List[str]) -> pd.DataFrame:
        """
        比较多个因子
        
        Parameters:
        -----------
        factor_names : List[str]
            要比较的因子名称列表
            
        Returns:
        --------
        pd.DataFrame
            比较结果表
        """
        comparison_results = []
        
        for factor_name in factor_names:
            if factor_name not in self.test_results_cache:
                logger.warning("Factor %s not found", factor_name)
                continue
            
            result = self.test_results_cache[factor_name]
            
            row = {
                'factor_name': factor_name,
                'category': self.config.get_factor_category(factor_name),
                'score': self.get_factor_score(factor_name)
            }
            
            # 添加各项指标
            if result.ic_result:
                row['ic_mean'] = result.ic_result.ic_mean
                row['icir'] = result.ic_result.icir
            
            if result.group_result:
                row['monotonicity'] = result.group_result.monotonicity_score
            
            if result.performance_metrics:
                row['sharpe'] = result.performance_metrics.get('long_short_sharpe', 0)
                row['annual_return'] = result.performance_metrics.get('long_short_annual_return', 0)
            
            comparison_results.append(row)
        
        return pd.DataFrame(comparison_results).sort_values('score', ascending=False)