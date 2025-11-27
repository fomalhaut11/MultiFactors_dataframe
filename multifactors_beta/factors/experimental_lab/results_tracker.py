#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果跟踪器模块

跟踪和分析实验因子的完整生命周期结果
为factors.analyzer模块提供结构化的实验数据源
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict

from .factor_registry import ExperimentalFactorRegistry, FactorStatus
from factors.tester.base import TestResult
from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class FactorLifecycleRecord:
    """因子生命周期记录"""
    factor_name: str
    registration_time: datetime
    calculation_time: Optional[datetime] = None
    test_time: Optional[datetime] = None
    final_status: str = "registered"
    
    # 计算结果
    calculation_success: bool = False
    calculation_time_cost: float = 0.0
    data_quality_score: float = 0.0
    
    # 测试结果
    test_success: bool = False
    test_time_cost: float = 0.0
    performance_metrics: Dict[str, float] = None
    
    # 决策结果
    promotion_decision: Optional[str] = None  # promoted/archived/pending
    decision_time: Optional[datetime] = None
    decision_reason: str = ""
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class PerformanceAnalytics:
    """性能分析器"""
    
    @staticmethod
    def calculate_factor_score(performance_metrics: Dict[str, float], 
                             weights: Dict[str, float] = None) -> float:
        """
        计算因子综合评分
        
        Parameters:
        -----------
        performance_metrics : Dict[str, float]
            性能指标
        weights : Dict[str, float], optional
            权重配置
            
        Returns:
        --------
        float
            综合评分 (0-100)
        """
        if weights is None:
            weights = {
                'ic_mean': 0.3,
                'icir': 0.3,
                'sharpe': 0.2,
                'rank_ic_mean': 0.2
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in performance_metrics:
                value = performance_metrics[metric]
                # 标准化到0-100范围
                normalized_value = PerformanceAnalytics._normalize_metric(metric, value)
                score += normalized_value * weight
                total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight * 100
        
        return max(0, min(100, score))
    
    @staticmethod
    def _normalize_metric(metric_name: str, value: float) -> float:
        """将指标值标准化到0-1范围"""
        # 这里可以根据历史数据定义更精确的标准化方法
        normalization_rules = {
            'ic_mean': lambda x: max(0, min(1, (x + 0.1) / 0.2)),  # -0.1到0.1映射到0-1
            'icir': lambda x: max(0, min(1, (x + 1) / 3)),         # -1到2映射到0-1
            'sharpe': lambda x: max(0, min(1, (x + 1) / 4)),       # -1到3映射到0-1
            'rank_ic_mean': lambda x: max(0, min(1, (x + 0.1) / 0.2))
        }
        
        if metric_name in normalization_rules:
            return normalization_rules[metric_name](value)
        else:
            # 默认标准化方法
            return max(0, min(1, (value + 1) / 2))
    
    @staticmethod
    def analyze_factor_distribution(factors_data: List[FactorLifecycleRecord]) -> Dict[str, Any]:
        """分析因子分布特征"""
        if not factors_data:
            return {}
        
        # 状态分布
        status_counts = {}
        for record in factors_data:
            status = record.final_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 成功率统计
        total_factors = len(factors_data)
        calculation_success_rate = sum(1 for r in factors_data if r.calculation_success) / total_factors
        test_success_rate = sum(1 for r in factors_data if r.test_success) / total_factors
        
        # 性能指标分布
        ic_values = []
        icir_values = []
        for record in factors_data:
            if record.performance_metrics:
                if 'ic_mean' in record.performance_metrics:
                    ic_values.append(record.performance_metrics['ic_mean'])
                if 'icir' in record.performance_metrics:
                    icir_values.append(record.performance_metrics['icir'])
        
        analysis = {
            'total_factors': total_factors,
            'status_distribution': status_counts,
            'success_rates': {
                'calculation_success_rate': calculation_success_rate,
                'test_success_rate': test_success_rate
            },
            'performance_stats': {
                'ic_mean_stats': {
                    'count': len(ic_values),
                    'mean': np.mean(ic_values) if ic_values else 0,
                    'std': np.std(ic_values) if ic_values else 0,
                    'min': np.min(ic_values) if ic_values else 0,
                    'max': np.max(ic_values) if ic_values else 0
                },
                'icir_stats': {
                    'count': len(icir_values),
                    'mean': np.mean(icir_values) if icir_values else 0,
                    'std': np.std(icir_values) if icir_values else 0,
                    'min': np.min(icir_values) if icir_values else 0,
                    'max': np.max(icir_values) if icir_values else 0
                }
            }
        }
        
        return analysis


class ResultsTracker:
    """
    结果跟踪器
    
    功能：
    1. 跟踪因子完整生命周期
    2. 性能趋势分析
    3. 为筛选器提供结构化数据
    4. 实验效果评估和优化建议
    """
    
    def __init__(self, registry: Optional[ExperimentalFactorRegistry] = None,
                 storage_path: Optional[str] = None):
        """
        初始化结果跟踪器
        
        Parameters:
        -----------
        registry : ExperimentalFactorRegistry, optional
            因子注册表实例
        storage_path : str, optional
            存储路径
        """
        self.registry = registry or ExperimentalFactorRegistry()
        
        # 设置存储路径
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            try:
                base_path = Path(get_config('main.paths.factors_data'))
                self.storage_path = base_path / "experimental_lab" / "tracking"
            except:
                self.storage_path = Path("data/factors/experimental_lab/tracking")
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 跟踪数据文件
        self.lifecycle_file = self.storage_path / "lifecycle_records.json"
        self.analytics_file = self.storage_path / "analytics_cache.pkl"
        
        # 内存缓存
        self.lifecycle_records: Dict[str, FactorLifecycleRecord] = {}
        self.analytics_cache: Dict[str, Any] = {}
        
        # 加载现有数据
        self._load_tracking_data()
        
        logger.info(f"结果跟踪器初始化完成，存储路径: {self.storage_path}")
    
    def update_factor_lifecycle(self, factor_name: str, event_type: str, **event_data):
        """
        更新因子生命周期记录
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        event_type : str
            事件类型：registration/calculation/testing/decision
        **event_data : dict
            事件数据
        """
        # 获取或创建记录
        if factor_name not in self.lifecycle_records:
            factor = self.registry.get_factor(factor_name)
            if factor:
                self.lifecycle_records[factor_name] = FactorLifecycleRecord(
                    factor_name=factor_name,
                    registration_time=factor.created_time
                )
            else:
                self.lifecycle_records[factor_name] = FactorLifecycleRecord(
                    factor_name=factor_name,
                    registration_time=datetime.now()
                )
        
        record = self.lifecycle_records[factor_name]
        
        # 更新记录
        if event_type == "calculation":
            record.calculation_time = event_data.get('timestamp', datetime.now())
            record.calculation_success = event_data.get('success', False)
            record.calculation_time_cost = event_data.get('time_cost', 0.0)
            record.data_quality_score = event_data.get('quality_score', 0.0)
            
        elif event_type == "testing":
            record.test_time = event_data.get('timestamp', datetime.now())
            record.test_success = event_data.get('success', False)
            record.test_time_cost = event_data.get('time_cost', 0.0)
            if 'performance_metrics' in event_data:
                record.performance_metrics.update(event_data['performance_metrics'])
            
        elif event_type == "decision":
            record.promotion_decision = event_data.get('decision', 'pending')
            record.decision_time = event_data.get('timestamp', datetime.now())
            record.decision_reason = event_data.get('reason', '')
            record.final_status = event_data.get('status', record.final_status)
        
        # 保存更新
        self._save_tracking_data()
        
        logger.debug(f"更新因子 {factor_name} 生命周期记录: {event_type}")
    
    def get_factor_lifecycle(self, factor_name: str) -> Optional[FactorLifecycleRecord]:
        """获取因子生命周期记录"""
        return self.lifecycle_records.get(factor_name)
    
    def get_all_lifecycles(self, status_filter: List[str] = None) -> List[FactorLifecycleRecord]:
        """
        获取所有生命周期记录
        
        Parameters:
        -----------
        status_filter : List[str], optional
            状态筛选
            
        Returns:
        --------
        List[FactorLifecycleRecord]
            生命周期记录列表
        """
        records = list(self.lifecycle_records.values())
        
        if status_filter:
            records = [r for r in records if r.final_status in status_filter]
        
        # 按注册时间排序
        records.sort(key=lambda r: r.registration_time, reverse=True)
        return records
    
    def generate_lifecycle_summary(self) -> pd.DataFrame:
        """生成生命周期汇总表"""
        if not self.lifecycle_records:
            return pd.DataFrame()
        
        data = []
        for record in self.lifecycle_records.values():
            row = {
                'factor_name': record.factor_name,
                'registration_time': record.registration_time,
                'final_status': record.final_status,
                'calculation_success': record.calculation_success,
                'test_success': record.test_success,
                'calculation_time_cost': record.calculation_time_cost,
                'test_time_cost': record.test_time_cost,
                'data_quality_score': record.data_quality_score,
                'promotion_decision': record.promotion_decision or 'pending'
            }
            
            # 添加性能指标
            if record.performance_metrics:
                for metric, value in record.performance_metrics.items():
                    row[f'perf_{metric}'] = value
                
                # 计算综合评分
                row['overall_score'] = PerformanceAnalytics.calculate_factor_score(
                    record.performance_metrics
                )
            
            # 计算总耗时
            total_time = record.calculation_time_cost + record.test_time_cost
            row['total_time_cost'] = total_time
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('registration_time', ascending=False)
        return df
    
    def analyze_experimental_performance(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        分析实验性能
        
        Parameters:
        -----------
        time_window_days : int
            分析时间窗口（天）
            
        Returns:
        --------
        Dict[str, Any]
            分析结果
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        # 筛选时间窗口内的记录
        recent_records = [
            r for r in self.lifecycle_records.values()
            if r.registration_time >= cutoff_date
        ]
        
        if not recent_records:
            return {'message': f'过去{time_window_days}天内没有新的实验因子'}
        
        # 基本统计
        basic_stats = PerformanceAnalytics.analyze_factor_distribution(recent_records)
        
        # 时间趋势分析
        time_trend = self._analyze_time_trends(recent_records)
        
        # 成功因子特征分析
        successful_factors = self._analyze_successful_factors(recent_records)
        
        # 失败原因分析
        failure_analysis = self._analyze_failure_patterns(recent_records)
        
        analysis_result = {
            'analysis_period': f'过去{time_window_days}天',
            'basic_statistics': basic_stats,
            'time_trends': time_trend,
            'successful_factors_analysis': successful_factors,
            'failure_analysis': failure_analysis,
            'recommendations': self._generate_recommendations(recent_records)
        }
        
        # 缓存分析结果
        self.analytics_cache[f'performance_analysis_{time_window_days}d'] = {
            'result': analysis_result,
            'generated_at': datetime.now()
        }
        self._save_tracking_data()
        
        return analysis_result
    
    def export_for_screening(self, performance_threshold: float = 0.0,
                           include_pending: bool = False) -> Dict[str, Any]:
        """
        导出数据供筛选器使用
        
        Parameters:
        -----------
        performance_threshold : float
            性能阈值，只导出评分高于此值的因子
        include_pending : bool
            是否包含待评估的因子
            
        Returns:
        --------
        Dict[str, Any]
            筛选器格式的数据
        """
        # 筛选符合条件的因子
        qualified_records = []
        for record in self.lifecycle_records.values():
            if not record.test_success:
                continue
            
            # 计算综合评分
            if record.performance_metrics:
                score = PerformanceAnalytics.calculate_factor_score(record.performance_metrics)
                if score >= performance_threshold:
                    qualified_records.append(record)
            elif include_pending:
                qualified_records.append(record)
        
        # 构造导出数据
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_qualified_factors': len(qualified_records),
                'performance_threshold': performance_threshold,
                'include_pending': include_pending
            },
            'factors': {},
            'lifecycle_summary': []
        }
        
        for record in qualified_records:
            factor_name = record.factor_name
            
            # 因子详细信息
            export_data['factors'][factor_name] = {
                'basic_info': {
                    'name': factor_name,
                    'registration_time': record.registration_time.isoformat(),
                    'final_status': record.final_status
                },
                'performance_metrics': record.performance_metrics,
                'lifecycle_metrics': {
                    'calculation_success': record.calculation_success,
                    'test_success': record.test_success,
                    'total_time_cost': record.calculation_time_cost + record.test_time_cost,
                    'data_quality_score': record.data_quality_score
                },
                'decision_info': {
                    'promotion_decision': record.promotion_decision,
                    'decision_reason': record.decision_reason
                }
            }
            
            # 生命周期汇总
            export_data['lifecycle_summary'].append({
                'factor_name': factor_name,
                'overall_score': PerformanceAnalytics.calculate_factor_score(
                    record.performance_metrics
                ) if record.performance_metrics else 0,
                'status': record.final_status,
                'test_time': record.test_time.isoformat() if record.test_time else None
            })
        
        return export_data
    
    def get_top_performing_factors(self, top_n: int = 10, 
                                  metric: str = 'overall_score') -> List[Tuple[str, float]]:
        """
        获取表现最佳的因子
        
        Parameters:
        -----------
        top_n : int
            返回数量
        metric : str
            排序指标
            
        Returns:
        --------
        List[Tuple[str, float]]
            (因子名称, 得分) 元组列表
        """
        factor_scores = []
        
        for record in self.lifecycle_records.values():
            if not record.test_success or not record.performance_metrics:
                continue
            
            if metric == 'overall_score':
                score = PerformanceAnalytics.calculate_factor_score(record.performance_metrics)
            elif metric in record.performance_metrics:
                score = record.performance_metrics[metric]
            else:
                continue
            
            factor_scores.append((record.factor_name, score))
        
        # 排序并返回前N个
        factor_scores.sort(key=lambda x: x[1], reverse=True)
        return factor_scores[:top_n]
    
    def _analyze_time_trends(self, records: List[FactorLifecycleRecord]) -> Dict[str, Any]:
        """分析时间趋势"""
        if not records:
            return {}
        
        # 按日期分组统计
        daily_stats = {}
        for record in records:
            date_key = record.registration_time.date().isoformat()
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    'registered': 0,
                    'calculated': 0,
                    'tested': 0,
                    'successful': 0
                }
            
            daily_stats[date_key]['registered'] += 1
            if record.calculation_success:
                daily_stats[date_key]['calculated'] += 1
            if record.test_success:
                daily_stats[date_key]['tested'] += 1
            if record.promotion_decision == 'promoted':
                daily_stats[date_key]['successful'] += 1
        
        return daily_stats
    
    def _analyze_successful_factors(self, records: List[FactorLifecycleRecord]) -> Dict[str, Any]:
        """分析成功因子的特征"""
        successful_records = [
            r for r in records 
            if r.promotion_decision == 'promoted' or (
                r.test_success and r.performance_metrics and 
                PerformanceAnalytics.calculate_factor_score(r.performance_metrics) > 60
            )
        ]
        
        if not successful_records:
            return {'message': '暂无成功因子可分析'}
        
        # 计算成功因子的平均指标
        avg_metrics = {}
        metric_counts = {}
        
        for record in successful_records:
            if record.performance_metrics:
                for metric, value in record.performance_metrics.items():
                    if metric not in avg_metrics:
                        avg_metrics[metric] = 0
                        metric_counts[metric] = 0
                    avg_metrics[metric] += value
                    metric_counts[metric] += 1
        
        for metric in avg_metrics:
            if metric_counts[metric] > 0:
                avg_metrics[metric] /= metric_counts[metric]
        
        return {
            'successful_factor_count': len(successful_records),
            'average_performance_metrics': avg_metrics,
            'average_time_to_success': np.mean([
                (r.decision_time - r.registration_time).total_seconds() / 3600
                for r in successful_records if r.decision_time
            ]) if any(r.decision_time for r in successful_records) else 0
        }
    
    def _analyze_failure_patterns(self, records: List[FactorLifecycleRecord]) -> Dict[str, Any]:
        """分析失败模式"""
        failed_records = [
            r for r in records 
            if r.promotion_decision == 'archived' or r.final_status == 'failed'
        ]
        
        failure_reasons = {}
        for record in failed_records:
            reason = record.decision_reason or 'unknown'
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'failed_factor_count': len(failed_records),
            'failure_reasons': failure_reasons,
            'common_failure_stage': self._identify_common_failure_stage(failed_records)
        }
    
    def _identify_common_failure_stage(self, failed_records: List[FactorLifecycleRecord]) -> str:
        """识别常见失败阶段"""
        if not failed_records:
            return 'none'
        
        calculation_failures = sum(1 for r in failed_records if not r.calculation_success)
        test_failures = sum(1 for r in failed_records if not r.test_success)
        
        if calculation_failures > len(failed_records) * 0.5:
            return 'calculation'
        elif test_failures > len(failed_records) * 0.5:
            return 'testing'
        else:
            return 'evaluation'
    
    def _generate_recommendations(self, records: List[FactorLifecycleRecord]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not records:
            return recommendations
        
        # 计算成功率
        calc_success_rate = sum(1 for r in records if r.calculation_success) / len(records)
        test_success_rate = sum(1 for r in records if r.test_success) / len(records)
        
        if calc_success_rate < 0.7:
            recommendations.append("计算成功率较低，建议检查数据质量和计算逻辑")
        
        if test_success_rate < 0.6:
            recommendations.append("测试成功率偏低，建议优化因子设计和测试参数")
        
        # 分析平均耗时
        avg_calc_time = np.mean([r.calculation_time_cost for r in records if r.calculation_time_cost > 0])
        if avg_calc_time > 300:  # 5分钟
            recommendations.append("计算耗时较长，建议优化计算效率")
        
        return recommendations
    
    def _load_tracking_data(self):
        """加载跟踪数据"""
        try:
            # 加载生命周期记录
            if self.lifecycle_file.exists():
                with open(self.lifecycle_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for factor_name, record_data in data.items():
                    # 恢复datetime对象
                    record_data['registration_time'] = datetime.fromisoformat(record_data['registration_time'])
                    if record_data.get('calculation_time'):
                        record_data['calculation_time'] = datetime.fromisoformat(record_data['calculation_time'])
                    if record_data.get('test_time'):
                        record_data['test_time'] = datetime.fromisoformat(record_data['test_time'])
                    if record_data.get('decision_time'):
                        record_data['decision_time'] = datetime.fromisoformat(record_data['decision_time'])
                    
                    self.lifecycle_records[factor_name] = FactorLifecycleRecord(**record_data)
            
            # 加载分析缓存
            if self.analytics_file.exists():
                with open(self.analytics_file, 'rb') as f:
                    self.analytics_cache = pickle.load(f)
            
            logger.info(f"加载跟踪数据完成，共 {len(self.lifecycle_records)} 个因子记录")
            
        except Exception as e:
            logger.warning(f"加载跟踪数据失败: {e}，将创建新的跟踪数据")
    
    def _save_tracking_data(self):
        """保存跟踪数据"""
        try:
            # 保存生命周期记录
            serializable_data = {}
            for factor_name, record in self.lifecycle_records.items():
                data = asdict(record)
                # 转换datetime为字符串
                data['registration_time'] = record.registration_time.isoformat()
                if record.calculation_time:
                    data['calculation_time'] = record.calculation_time.isoformat()
                if record.test_time:
                    data['test_time'] = record.test_time.isoformat()
                if record.decision_time:
                    data['decision_time'] = record.decision_time.isoformat()
                serializable_data[factor_name] = data
            
            with open(self.lifecycle_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            # 保存分析缓存
            with open(self.analytics_file, 'wb') as f:
                pickle.dump(self.analytics_cache, f)
            
            logger.debug(f"保存跟踪数据完成，共 {len(self.lifecycle_records)} 个记录")
            
        except Exception as e:
            logger.error(f"保存跟踪数据失败: {e}")


if __name__ == "__main__":
    # 测试代码
    tracker = ResultsTracker()
    
    # 模拟添加一些生命周期记录
    tracker.update_factor_lifecycle("test_factor_1", "calculation", 
                                   success=True, time_cost=120.5, quality_score=0.85)
    
    tracker.update_factor_lifecycle("test_factor_1", "testing",
                                   success=True, time_cost=300.0,
                                   performance_metrics={'ic_mean': 0.06, 'icir': 0.8})
    
    tracker.update_factor_lifecycle("test_factor_1", "decision",
                                   decision='promoted', status='promoted',
                                   reason='表现优秀')
    
    print("结果跟踪器测试成功")
    
    # 生成汇总表
    summary = tracker.generate_lifecycle_summary()
    print(f"生命周期汇总表形状: {summary.shape}")
    
    # 分析实验性能
    analysis = tracker.analyze_experimental_performance(30)
    print(f"实验性能分析完成")