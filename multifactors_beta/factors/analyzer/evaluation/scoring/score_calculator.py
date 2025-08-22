"""
评分计算器
负责计算和管理各维度的得分
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """评分结果"""
    total_score: float  # 总分(0-100)
    dimension_scores: Dict[str, float]  # 各维度得分
    weighted_scores: Dict[str, float]  # 加权后的得分
    weights: Dict[str, float]  # 使用的权重
    grade: str  # 综合等级
    percentile: Optional[float] = None  # 百分位排名
    timestamp: datetime = field(default_factory=datetime.now)


class ScoreCalculator:
    """
    评分计算器
    
    负责：
    1. 管理评分权重
    2. 计算综合得分
    3. 等级映射
    4. 百分位排名
    """
    
    # 默认场景权重配置
    DEFAULT_SCENARIOS = {
        'balanced': {  # 均衡型
            'profitability': 0.35,
            'stability': 0.25,
            'tradability': 0.20,
            'uniqueness': 0.10,
            'timeliness': 0.10
        },
        'high_frequency': {  # 高频交易
            'profitability': 0.25,
            'stability': 0.15,
            'tradability': 0.40,
            'uniqueness': 0.10,
            'timeliness': 0.10
        },
        'value_investing': {  # 价值投资
            'profitability': 0.40,
            'stability': 0.35,
            'tradability': 0.10,
            'uniqueness': 0.10,
            'timeliness': 0.05
        },
        'risk_neutral': {  # 风险中性
            'profitability': 0.25,
            'stability': 0.45,
            'tradability': 0.15,
            'uniqueness': 0.10,
            'timeliness': 0.05
        },
        'momentum': {  # 动量策略
            'profitability': 0.35,
            'stability': 0.20,
            'tradability': 0.25,
            'uniqueness': 0.05,
            'timeliness': 0.15
        }
    }
    
    # 等级系统
    GRADE_SYSTEM = [
        (90, 'AAA', '卓越因子'),
        (85, 'AA', '优秀因子'),
        (80, 'A', '良好因子'),
        (70, 'BBB', '合格因子'),
        (60, 'BB', '一般因子'),
        (50, 'B', '较差因子'),
        (0, 'C', '不推荐使用')
    ]
    
    def __init__(self, scenario: str = 'balanced', config: Optional[Dict] = None):
        """
        初始化评分计算器
        
        Parameters
        ----------
        scenario : str
            评分场景
        config : Dict, optional
            配置参数
        """
        self.scenario = scenario
        self.config = config or {}
        
        # 设置权重
        if scenario in self.DEFAULT_SCENARIOS:
            self.weights = self.DEFAULT_SCENARIOS[scenario].copy()
        else:
            self.weights = self.DEFAULT_SCENARIOS['balanced'].copy()
            logger.warning(f"Unknown scenario '{scenario}', using 'balanced' weights")
        
        # 可以覆盖权重
        if 'custom_weights' in self.config:
            self.weights.update(self.config['custom_weights'])
        
        # 历史分数（用于计算百分位）
        self.historical_scores = []
        
        logger.info(f"ScoreCalculator initialized with scenario '{scenario}'")
    
    def calculate(self, dimension_scores: Dict[str, float]) -> ScoreResult:
        """
        计算综合得分
        
        Parameters
        ----------
        dimension_scores : Dict[str, float]
            各维度得分(0-100)
            
        Returns
        -------
        ScoreResult
            评分结果
        """
        # 验证输入
        if not dimension_scores:
            logger.warning("Empty dimension scores")
            return self._create_empty_result()
        
        # 计算加权得分
        weighted_scores = {}
        total_weighted_score = 0
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            weight = self.weights.get(dimension, 0)
            weighted_score = score * weight
            weighted_scores[dimension] = weighted_score
            total_weighted_score += weighted_score
            total_weight += weight
        
        # 计算总分
        if total_weight > 0:
            total_score = total_weighted_score / total_weight
        else:
            total_score = 0
        
        # 确保分数在0-100范围内
        total_score = max(0, min(100, total_score))
        
        # 获取等级
        grade = self.get_grade(total_score)
        
        # 计算百分位（如果有历史数据）
        percentile = self.calculate_percentile(total_score)
        
        # 保存到历史
        self.historical_scores.append(total_score)
        
        return ScoreResult(
            total_score=total_score,
            dimension_scores=dimension_scores,
            weighted_scores=weighted_scores,
            weights=self.weights.copy(),
            grade=grade,
            percentile=percentile
        )
    
    def batch_calculate(self, 
                       factors_dimensions: Dict[str, Dict[str, float]]) -> Dict[str, ScoreResult]:
        """
        批量计算多个因子的得分
        
        Parameters
        ----------
        factors_dimensions : Dict[str, Dict[str, float]]
            因子名称到维度得分的映射
            
        Returns
        -------
        Dict[str, ScoreResult]
            因子名称到评分结果的映射
        """
        results = {}
        
        for factor_name, dimension_scores in factors_dimensions.items():
            results[factor_name] = self.calculate(dimension_scores)
        
        # 更新百分位排名
        self._update_percentiles(results)
        
        return results
    
    def get_grade(self, score: float) -> str:
        """
        根据分数获取等级
        
        Parameters
        ----------
        score : float
            总分(0-100)
            
        Returns
        -------
        str
            等级
        """
        for threshold, grade, _ in self.GRADE_SYSTEM:
            if score >= threshold:
                return grade
        return 'C'
    
    def get_grade_description(self, grade: str) -> str:
        """
        获取等级描述
        
        Parameters
        ----------
        grade : str
            等级
            
        Returns
        -------
        str
            等级描述
        """
        for _, g, description in self.GRADE_SYSTEM:
            if g == grade:
                return description
        return '未知等级'
    
    def calculate_percentile(self, score: float) -> Optional[float]:
        """
        计算百分位排名
        
        Parameters
        ----------
        score : float
            分数
            
        Returns
        -------
        float or None
            百分位排名(0-100)
        """
        if len(self.historical_scores) < 10:  # 样本太少
            return None
        
        # 计算有多少历史分数低于当前分数
        lower_count = sum(1 for s in self.historical_scores if s < score)
        percentile = (lower_count / len(self.historical_scores)) * 100
        
        return percentile
    
    def _update_percentiles(self, results: Dict[str, ScoreResult]):
        """
        更新批量结果的百分位排名
        
        Parameters
        ----------
        results : Dict[str, ScoreResult]
            评分结果字典
        """
        if len(results) < 2:
            return
        
        # 获取所有分数
        scores = [(name, result.total_score) for name, result in results.items()]
        scores.sort(key=lambda x: x[1])
        
        # 计算每个因子的百分位
        n = len(scores)
        for i, (name, _) in enumerate(scores):
            percentile = (i / (n - 1)) * 100 if n > 1 else 50
            results[name].percentile = percentile
    
    def set_weights(self, weights: Dict[str, float]):
        """
        设置自定义权重
        
        Parameters
        ----------
        weights : Dict[str, float]
            维度权重字典
        """
        # 验证权重和是否为1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            # 归一化
            weights = {k: v/total_weight for k, v in weights.items()}
        
        self.weights = weights
        logger.info(f"Updated weights: {self.weights}")
    
    def set_scenario(self, scenario: str):
        """
        切换评分场景
        
        Parameters
        ----------
        scenario : str
            场景名称
        """
        if scenario not in self.DEFAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        self.scenario = scenario
        self.weights = self.DEFAULT_SCENARIOS[scenario].copy()
        logger.info(f"Switched to scenario '{scenario}'")
    
    def get_available_scenarios(self) -> List[str]:
        """
        获取可用的评分场景
        
        Returns
        -------
        List[str]
            场景名称列表
        """
        return list(self.DEFAULT_SCENARIOS.keys())
    
    def compare_scores(self, 
                      score1: ScoreResult, 
                      score2: ScoreResult) -> Dict[str, Any]:
        """
        比较两个评分结果
        
        Parameters
        ----------
        score1 : ScoreResult
            第一个评分结果
        score2 : ScoreResult
            第二个评分结果
            
        Returns
        -------
        Dict[str, Any]
            比较结果
        """
        comparison = {
            'total_score_diff': score1.total_score - score2.total_score,
            'grade_comparison': f"{score1.grade} vs {score2.grade}",
            'dimension_diffs': {},
            'better_dimensions': [],
            'worse_dimensions': []
        }
        
        # 比较各维度
        for dim in score1.dimension_scores:
            if dim in score2.dimension_scores:
                diff = score1.dimension_scores[dim] - score2.dimension_scores[dim]
                comparison['dimension_diffs'][dim] = diff
                
                if diff > 5:  # 显著更好
                    comparison['better_dimensions'].append(dim)
                elif diff < -5:  # 显著更差
                    comparison['worse_dimensions'].append(dim)
        
        return comparison
    
    def generate_score_report(self, result: ScoreResult) -> str:
        """
        生成评分报告
        
        Parameters
        ----------
        result : ScoreResult
            评分结果
            
        Returns
        -------
        str
            文本报告
        """
        lines = [
            "=" * 50,
            "因子评分报告",
            "=" * 50,
            f"评估时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"评分场景: {self.scenario}",
            "",
            f"总分: {result.total_score:.1f}/100",
            f"等级: {result.grade} ({self.get_grade_description(result.grade)})",
        ]
        
        if result.percentile is not None:
            lines.append(f"百分位排名: {result.percentile:.1f}%")
        
        lines.extend([
            "",
            "维度得分明细:",
            "-" * 30
        ])
        
        # 按得分排序维度
        sorted_dims = sorted(result.dimension_scores.items(), 
                           key=lambda x: x[1], reverse=True)
        
        for dim, score in sorted_dims:
            weight = result.weights.get(dim, 0)
            weighted = result.weighted_scores.get(dim, 0)
            lines.append(f"{dim:15s}: {score:6.1f} × {weight:.0%} = {weighted:6.1f}")
        
        lines.extend([
            "-" * 30,
            "",
            "评估建议:",
        ])
        
        # 生成建议
        if result.total_score >= 80:
            lines.append("• 该因子表现优秀，建议纳入核心因子池")
        elif result.total_score >= 60:
            lines.append("• 该因子表现良好，可以考虑使用")
        else:
            lines.append("• 该因子表现不佳，建议谨慎使用或进行优化")
        
        # 维度建议
        for dim, score in sorted_dims:
            if score < 50:
                lines.append(f"• 需要改进{dim}维度 (当前{score:.1f}分)")
        
        return "\n".join(lines)
    
    def _create_empty_result(self) -> ScoreResult:
        """创建空的评分结果"""
        return ScoreResult(
            total_score=0,
            dimension_scores={},
            weighted_scores={},
            weights=self.weights.copy(),
            grade='C',
            percentile=None
        )