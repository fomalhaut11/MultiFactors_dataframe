"""
单因子测试结果数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import os
import pickle
import yaml
import json
from pathlib import Path
import uuid


@dataclass
class RegressionResult:
    """回归分析结果"""
    params: pd.Series  # 回归系数
    tvalues: pd.Series  # t值
    pvalues: pd.Series  # p值
    resid: pd.Series  # 残差
    rsquared_adj: float  # 调整R方
    factor_return: pd.Series  # 因子收益序列
    cumulative_return: pd.Series  # 累计收益
    
    @property
    def factor_return_mean(self) -> float:
        """平均因子收益"""
        if len(self.factor_return) > 0:
            return float(self.factor_return.mean())
        return 0.0
    
    @property
    def t_stat_mean(self) -> float:
        """平均t统计量"""
        if len(self.tvalues) > 0:
            return float(self.tvalues.mean())
        return 0.0
        
    @property
    def significance_ratio(self) -> float:
        """显著性比例(p<0.05)"""
        if len(self.pvalues) > 0:
            return float((self.pvalues < 0.05).mean())
        return 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        # 安全地转换Series为字典，处理Timestamp索引
        def safe_series_to_dict(series):
            if isinstance(series, pd.Series):
                return {str(k): v for k, v in series.to_dict().items()}
            return series
        
        return {
            'params': safe_series_to_dict(self.params),
            'tvalues': safe_series_to_dict(self.tvalues),
            'pvalues': safe_series_to_dict(self.pvalues),
            'rsquared_adj': self.rsquared_adj,
            'factor_return_mean': float(self.factor_return.mean()) if len(self.factor_return) > 0 else 0,
            'factor_return_std': float(self.factor_return.std()) if len(self.factor_return) > 0 else 0,
            'cumulative_return_final': float(self.cumulative_return.iloc[-1]) if len(self.cumulative_return) > 0 else 0
        }


@dataclass
class GroupResult:
    """分组测试结果"""
    group_returns: pd.DataFrame  # 各组收益率
    group_counts: pd.DataFrame  # 各组股票数量
    long_short_return: pd.Series  # 多空收益
    group_cumulative_returns: pd.DataFrame  # 各组累计收益
    monotonicity_score: float  # 单调性得分
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        # 安全地转换Series为字典，处理Timestamp索引
        def safe_series_to_dict(series):
            if isinstance(series, pd.Series):
                return {str(k): v for k, v in series.to_dict().items()}
            return series if series else {}
        
        return {
            'group_returns_mean': safe_series_to_dict(self.group_returns.mean() if not self.group_returns.empty else {}),
            'group_returns_std': safe_series_to_dict(self.group_returns.std() if not self.group_returns.empty else {}),
            'long_short_return_mean': float(self.long_short_return.mean()) if len(self.long_short_return) > 0 else 0,
            'long_short_return_std': float(self.long_short_return.std()) if len(self.long_short_return) > 0 else 0,
            'monotonicity_score': self.monotonicity_score
        }


@dataclass
class ICResult:
    """IC分析结果"""
    ic_series: pd.Series  # IC时间序列
    rank_ic_series: pd.Series  # Rank IC时间序列
    ic_mean: float  # IC均值
    ic_std: float  # IC标准差
    icir: float  # IC信息比率
    rank_ic_mean: float  # Rank IC均值
    rank_icir: float  # Rank IC信息比率
    ic_decay: pd.Series  # IC衰减
    
    @property
    def ic_win_rate(self) -> float:
        """IC胜率"""
        if len(self.ic_series) > 0:
            return float((self.ic_series > 0).mean())
        return 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'icir': self.icir,
            'rank_ic_mean': self.rank_ic_mean,
            'rank_icir': self.rank_icir,
            'ic_positive_ratio': float((self.ic_series > 0).mean()) if len(self.ic_series) > 0 else 0,
            'ic_decay_halflife': self._calculate_halflife() if len(self.ic_decay) > 1 else None
        }
    
    def _calculate_halflife(self) -> Optional[float]:
        """计算IC半衰期"""
        try:
            for i, val in enumerate(self.ic_decay):
                if abs(val) < abs(self.ic_decay.iloc[0]) / 2:
                    return float(i)
            return None
        except:
            return None


@dataclass
class TestResult:
    """单因子测试完整结果"""
    # 测试元数据
    test_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    test_time: datetime = field(default_factory=datetime.now)
    factor_name: str = ""
    
    @property
    def test_date(self) -> str:
        """测试日期"""
        return self.test_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 测试配置快照
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # 测试数据信息
    data_info: Dict[str, Any] = field(default_factory=dict)
    
    # 测试结果
    regression_result: Optional[RegressionResult] = None
    group_result: Optional[GroupResult] = None
    ic_result: Optional[ICResult] = None
    turnover_result: Optional[Dict[str, Any]] = None  # 换手率分析结果
    
    # 处理后的因子值
    processed_factor: Optional[pd.Series] = None
    
    # 性能指标
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_performance_metrics(self):
        """计算综合性能指标"""
        metrics = {}
        
        # 回归指标
        if self.regression_result:
            metrics['factor_return_annual'] = self.regression_result.factor_return.mean() * 252
            metrics['factor_return_sharpe'] = (
                self.regression_result.factor_return.mean() / 
                self.regression_result.factor_return.std() * np.sqrt(252)
                if self.regression_result.factor_return.std() > 0 else 0
            )
            metrics['t_value_mean'] = float(self.regression_result.tvalues.get('newfactor', 0))
            
        # 分组指标
        if self.group_result:
            metrics['long_short_annual_return'] = self.group_result.long_short_return.mean() * 252
            metrics['long_short_sharpe'] = (
                self.group_result.long_short_return.mean() / 
                self.group_result.long_short_return.std() * np.sqrt(252)
                if self.group_result.long_short_return.std() > 0 else 0
            )
            metrics['monotonicity_score'] = self.group_result.monotonicity_score
            
        # IC指标
        if self.ic_result:
            metrics['ic_mean'] = self.ic_result.ic_mean
            metrics['icir'] = self.ic_result.icir
            metrics['rank_ic_mean'] = self.ic_result.rank_ic_mean
            metrics['rank_icir'] = self.ic_result.rank_icir
            
        # 换手率指标
        if self.turnover_result:
            metrics['avg_turnover'] = self.turnover_result.get('avg_turnover', 0)
            metrics['max_turnover'] = self.turnover_result.get('max_turnover', 0)
            metrics['turnover_std'] = self.turnover_result.get('turnover_std', 0)
            metrics['total_turnover_cost'] = self.turnover_result.get('total_cost', 0)
            metrics['avg_turnover_cost'] = self.turnover_result.get('avg_cost', 0)
            
        self.performance_metrics = metrics
        return metrics
    
    def to_summary_dict(self) -> Dict:
        """生成摘要字典（用于快速查看）"""
        # 处理data_info中可能的Timestamp对象
        processed_data_info = {}
        if self.data_info:
            for key, value in self.data_info.items():
                if hasattr(value, 'isoformat'):
                    processed_data_info[key] = value.isoformat()
                elif hasattr(value, '__str__'):
                    processed_data_info[key] = str(value)
                else:
                    processed_data_info[key] = value
        
        summary = {
            'test_id': self.test_id,
            'test_time': self.test_time.isoformat(),
            'factor_name': self.factor_name,
            'config': {
                'begin_date': self.config_snapshot.get('begin_date'),
                'end_date': self.config_snapshot.get('end_date'),
                'group_nums': self.config_snapshot.get('group_nums'),
                'backtest_type': self.config_snapshot.get('backtest_type')
            },
            'data_info': processed_data_info,
            'performance_metrics': self.performance_metrics,
            'has_errors': len(self.errors) > 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
        
        # 添加详细结果摘要
        if self.regression_result:
            summary['regression'] = self.regression_result.to_dict()
        if self.group_result:
            summary['group'] = self.group_result.to_dict()
        if self.ic_result:
            summary['ic'] = self.ic_result.to_dict()
            
        return summary
    
    def save(self, path: str, save_format: str = 'pickle'):
        """
        保存测试结果
        
        Parameters
        ----------
        path : str
            保存路径
        save_format : str
            保存格式 ('pickle', 'parquet', 'both')
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        base_filename = f"{self.factor_name}_{self.test_time.strftime('%Y%m%d_%H%M%S')}_{self.test_id}"
        
        # 保存完整结果（pickle）
        if save_format in ['pickle', 'both']:
            pickle_file = os.path.join(path, f"{base_filename}.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(self, f)
        
        # 保存配置快照（yaml）
        config_file = os.path.join(path, f"{base_filename}_config.yaml")
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_snapshot, f, allow_unicode=True, default_flow_style=False)
        
        # 保存摘要（json）
        summary_file = os.path.join(path, f"{base_filename}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_summary_dict(), f, indent=2, ensure_ascii=False, default=str)
        
        # 保存关键数据表（parquet，可选）
        if save_format in ['parquet', 'both'] and self.processed_factor is not None:
            # 保存处理后的因子
            factor_file = os.path.join(path, f"{base_filename}_factor.parquet")
            self.processed_factor.to_frame('factor').to_parquet(factor_file)
            
            # 保存分组收益
            if self.group_result and not self.group_result.group_returns.empty:
                group_file = os.path.join(path, f"{base_filename}_group_returns.parquet")
                self.group_result.group_returns.to_parquet(group_file)
        
        return base_filename
    
    @classmethod
    def load(cls, filepath: str) -> 'TestResult':
        """
        加载测试结果
        
        Parameters
        ----------
        filepath : str
            文件路径
            
        Returns
        -------
        TestResult
            测试结果对象
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        return (f"TestResult(factor_name={self.factor_name}, "
                f"test_id={self.test_id}, "
                f"test_time={self.test_time.strftime('%Y-%m-%d %H:%M:%S')})")


@dataclass
class BatchTestResult:
    """批量测试结果"""
    test_results: List[TestResult] = field(default_factory=list)
    summary_df: Optional[pd.DataFrame] = None
    
    def add_result(self, result: TestResult):
        """添加测试结果"""
        self.test_results.append(result)
        
    def generate_summary(self) -> pd.DataFrame:
        """生成汇总表"""
        summaries = []
        for result in self.test_results:
            summary = {
                'factor_name': result.factor_name,
                'test_id': result.test_id,
                'test_time': result.test_time,
                **result.performance_metrics
            }
            summaries.append(summary)
        
        self.summary_df = pd.DataFrame(summaries)
        return self.summary_df
    
    def save(self, path: str):
        """保存批量测试结果"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # 保存汇总表
        if self.summary_df is not None:
            summary_file = os.path.join(path, f"batch_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存各个测试结果
        for result in self.test_results:
            result.save(path)