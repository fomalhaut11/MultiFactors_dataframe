"""
测试结果管理器
负责结果的保存、加载和分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import os
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime

from ..base.test_result import TestResult, BatchTestResult
from config import get_config

logger = logging.getLogger(__name__)


class ResultManager:
    """测试结果管理器"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        初始化结果管理器
        
        Parameters
        ----------
        base_path : str, optional
            结果保存的基础路径
        """
        self.base_path = base_path or get_config('main.paths.single_factor_test')
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def save(self, result: TestResult, subfolder: Optional[str] = None) -> str:
        """
        保存测试结果
        
        Parameters
        ----------
        result : TestResult
            测试结果
        subfolder : str, optional
            子文件夹名称
            
        Returns
        -------
        str
            保存的文件名
        """
        # 确定保存路径
        if subfolder:
            save_path = os.path.join(self.base_path, subfolder)
        else:
            # 按日期创建子文件夹
            date_folder = datetime.now().strftime('%Y%m%d')
            save_path = os.path.join(self.base_path, date_folder)
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        filename = result.save(save_path)
        logger.info(f"测试结果已保存: {filename}")
        
        return filename
    
    def load(self, filepath: str) -> TestResult:
        """
        加载测试结果
        
        Parameters
        ----------
        filepath : str
            文件路径
            
        Returns
        -------
        TestResult
            测试结果
        """
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.base_path, filepath)
        
        return TestResult.load(filepath)
    
    def load_batch(self, pattern: str = "*.pkl") -> List[TestResult]:
        """
        批量加载测试结果
        
        Parameters
        ----------
        pattern : str
            文件匹配模式
            
        Returns
        -------
        List[TestResult]
            测试结果列表
        """
        results = []
        
        # 查找所有匹配的文件
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.pkl') and 'config' not in file and 'summary' not in file:
                    try:
                        filepath = os.path.join(root, file)
                        result = self.load(filepath)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"加载文件失败 {file}: {e}")
        
        logger.info(f"加载了 {len(results)} 个测试结果")
        return results
    
    def get_summary_table(self, results: List[TestResult]) -> pd.DataFrame:
        """
        生成结果汇总表
        
        Parameters
        ----------
        results : List[TestResult]
            测试结果列表
            
        Returns
        -------
        pd.DataFrame
            汇总表
        """
        summaries = []
        
        for result in results:
            summary = {
                'factor_name': result.factor_name,
                'test_id': result.test_id,
                'test_time': result.test_time,
                'begin_date': result.config_snapshot.get('begin_date'),
                'end_date': result.config_snapshot.get('end_date'),
                'sample_count': result.data_info.get('factor_count', 0),
                'stock_count': result.data_info.get('stock_count', 0),
            }
            
            # 添加性能指标
            summary.update(result.performance_metrics)
            
            # 添加错误信息
            summary['has_error'] = len(result.errors) > 0
            summary['error_count'] = len(result.errors)
            
            summaries.append(summary)
        
        df = pd.DataFrame(summaries)
        
        # 排序
        if 'icir' in df.columns:
            df = df.sort_values('icir', ascending=False)
        
        return df
    
    def compare_results(self, results: List[TestResult]) -> pd.DataFrame:
        """
        比较多个测试结果
        
        Parameters
        ----------
        results : List[TestResult]
            测试结果列表
            
        Returns
        -------
        pd.DataFrame
            比较表
        """
        comparison = []
        
        for result in results:
            row = {
                'factor_name': result.factor_name,
                'test_id': result.test_id[:8],
            }
            
            # 回归结果
            if result.regression_result:
                row['t_value_mean'] = result.regression_result.tvalues.mean()
                row['t_value_abs_mean'] = result.regression_result.tvalues.abs().mean()
                row['factor_return_annual'] = result.regression_result.factor_return.mean() * 252
                row['factor_sharpe'] = (
                    result.regression_result.factor_return.mean() / 
                    result.regression_result.factor_return.std() * np.sqrt(252)
                ) if result.regression_result.factor_return.std() > 0 else 0
            
            # 分组结果
            if result.group_result:
                row['long_short_annual'] = result.group_result.long_short_return.mean() * 252
                row['long_short_sharpe'] = (
                    result.group_result.long_short_return.mean() / 
                    result.group_result.long_short_return.std() * np.sqrt(252)
                ) if result.group_result.long_short_return.std() > 0 else 0
                row['monotonicity'] = result.group_result.monotonicity_score
            
            # IC结果
            if result.ic_result:
                row['ic_mean'] = result.ic_result.ic_mean
                row['ic_std'] = result.ic_result.ic_std
                row['icir'] = result.ic_result.icir
                row['rank_ic_mean'] = result.ic_result.rank_ic_mean
                row['rank_icir'] = result.ic_result.rank_icir
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # 添加排名
        for col in ['t_value_abs_mean', 'factor_sharpe', 'long_short_sharpe', 
                   'monotonicity', 'ic_mean', 'icir']:
            if col in df.columns:
                df[f'{col}_rank'] = df[col].rank(ascending=False)
        
        return df
    
    def export_to_excel(
        self, 
        results: Union[TestResult, List[TestResult]], 
        filepath: str
    ):
        """
        导出结果到Excel
        
        Parameters
        ----------
        results : TestResult or List[TestResult]
            测试结果
        filepath : str
            Excel文件路径
        """
        if isinstance(results, TestResult):
            results = [results]
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 汇总表
            summary_df = self.get_summary_table(results)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 比较表
            if len(results) > 1:
                comparison_df = self.compare_results(results)
                comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
            
            # 每个因子的详细结果
            for i, result in enumerate(results[:10]):  # 最多10个
                sheet_name = f"{result.factor_name[:20]}_{i}"
                
                # 创建详细结果DataFrame
                detail_data = []
                
                # 回归结果
                if result.regression_result and not result.regression_result.factor_return.empty:
                    for date in result.regression_result.factor_return.index:
                        detail_data.append({
                            'date': date,
                            'factor_return': result.regression_result.factor_return.get(date, np.nan),
                            'cumulative_return': result.regression_result.cumulative_return.get(date, np.nan)
                        })
                
                if detail_data:
                    detail_df = pd.DataFrame(detail_data)
                    detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"结果已导出到Excel: {filepath}")
    
    def clean_old_results(self, days: int = 30):
        """
        清理旧的结果文件
        
        Parameters
        ----------
        days : int
            保留最近N天的结果
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                filepath = os.path.join(root, file)
                
                # 检查文件修改时间
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_date:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"删除文件失败 {filepath}: {e}")
        
        logger.info(f"清理了 {deleted_count} 个旧文件")