"""
评估维度基类
定义所有评估维度的通用接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """维度评分结果"""
    dimension_name: str
    score: float  # 0-100
    weight: float  # 权重
    metrics: Dict[str, float]  # 详细指标
    grade: str  # 等级 (A/B/C/D/F)
    description: str  # 评分说明


class BaseDimension(ABC):
    """
    评估维度基类
    
    定义所有评估维度必须实现的接口
    提供通用的评分和标准化功能
    """
    
    def __init__(self, 
                 name: str,
                 weight: float = 0.2,
                 config: Optional[Dict] = None):
        """
        初始化维度
        
        Parameters
        ----------
        name : str
            维度名称
        weight : float
            维度权重
        config : Dict, optional
            配置参数
        """
        self.name = name
        self.weight = weight
        self.config = config or {}
        
        # 评分标准
        self.score_criteria = self.config.get('score_criteria', self._get_default_criteria())
        
        # 指标权重
        self.metric_weights = self.config.get('metric_weights', self._get_default_weights())
        
    @abstractmethod
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算维度得分
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据，包含TestResult等分析结果
            
        Returns
        -------
        DimensionScore
            维度评分结果
        """
        pass
    
    @abstractmethod
    def extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        从数据中提取所需指标
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据
            
        Returns
        -------
        Dict[str, float]
            提取的指标
        """
        pass
    
    @abstractmethod
    def _get_default_criteria(self) -> Dict[str, Any]:
        """
        获取默认评分标准
        
        Returns
        -------
        Dict[str, Any]
            默认评分标准
        """
        pass
    
    @abstractmethod
    def _get_default_weights(self) -> Dict[str, float]:
        """
        获取默认指标权重
        
        Returns
        -------
        Dict[str, float]
            默认指标权重
        """
        pass
    
    def normalize_score(self, value: float, 
                       min_val: float, 
                       max_val: float,
                       reverse: bool = False) -> float:
        """
        标准化分数到0-100
        
        Parameters
        ----------
        value : float
            原始值
        min_val : float
            最小值
        max_val : float
            最大值
        reverse : bool
            是否反向（值越小越好）
            
        Returns
        -------
        float
            标准化后的分数(0-100)
        """
        if max_val == min_val:
            return 50  # 默认中间值
        
        # 限制在范围内
        value = max(min_val, min(max_val, value))
        
        # 标准化到0-1
        normalized = (value - min_val) / (max_val - min_val)
        
        if reverse:
            normalized = 1 - normalized
        
        # 映射到0-100
        return normalized * 100
    
    def map_to_score(self, value: float, 
                    thresholds: List[Tuple[float, float]]) -> float:
        """
        根据阈值映射到分数
        
        Parameters
        ----------
        value : float
            原始值
        thresholds : List[Tuple[float, float]]
            阈值和对应分数列表 [(threshold, score), ...]
            
        Returns
        -------
        float
            映射后的分数
        """
        # 从高到低排序阈值
        thresholds = sorted(thresholds, key=lambda x: x[0], reverse=True)
        
        for threshold, score in thresholds:
            if value >= threshold:
                return score
        
        # 如果都不满足，返回最低分
        return thresholds[-1][1] if thresholds else 0
    
    def calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """
        计算加权得分
        
        Parameters
        ----------
        metrics : Dict[str, float]
            各指标得分
            
        Returns
        -------
        float
            加权总分
        """
        total_score = 0
        total_weight = 0
        
        for metric_name, metric_score in metrics.items():
            weight = self.metric_weights.get(metric_name, 0)
            total_score += metric_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0
    
    def get_grade(self, score: float) -> str:
        """
        根据分数获取等级
        
        Parameters
        ----------
        score : float
            分数(0-100)
            
        Returns
        -------
        str
            等级(A/B/C/D/F)
        """
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_description(self, score: float, metrics: Dict[str, float]) -> str:
        """
        生成评分说明
        
        Parameters
        ----------
        score : float
            总分
        metrics : Dict[str, float]
            详细指标
            
        Returns
        -------
        str
            评分说明
        """
        grade = self.get_grade(score)
        
        # 找出最强和最弱的指标
        if metrics:
            best_metric = max(metrics.items(), key=lambda x: x[1])
            worst_metric = min(metrics.items(), key=lambda x: x[1])
            
            description = f"{self.name}维度得分{score:.1f}分(等级{grade})。"
            description += f"最强项是{best_metric[0]}({best_metric[1]:.1f}分)，"
            description += f"最弱项是{worst_metric[0]}({worst_metric[1]:.1f}分)。"
        else:
            description = f"{self.name}维度得分{score:.1f}分(等级{grade})。"
        
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
        if not data:
            logger.warning(f"{self.name} dimension: Empty input data")
            return False
        
        # 子类可以覆盖此方法添加更多验证
        return True
    
    def get_diagnostic_info(self, score: float, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        获取诊断信息
        
        Parameters
        ----------
        score : float
            维度得分
        metrics : Dict[str, float]
            详细指标
            
        Returns
        -------
        Dict[str, Any]
            诊断信息
        """
        diagnostics = {
            'dimension': self.name,
            'score': score,
            'grade': self.get_grade(score),
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }
        
        # 识别优势和劣势
        for metric_name, metric_score in metrics.items():
            if metric_score >= 80:
                diagnostics['strengths'].append(f"{metric_name}表现优秀({metric_score:.1f}分)")
            elif metric_score < 50:
                diagnostics['weaknesses'].append(f"{metric_name}需要改进({metric_score:.1f}分)")
        
        # 生成建议
        if score < 60:
            diagnostics['suggestions'].append(f"重点提升{self.name}维度的表现")
        
        return diagnostics


class CompositeDimension(BaseDimension):
    """
    复合维度基类
    
    支持从多个子维度组合计算
    """
    
    def __init__(self, 
                 name: str,
                 sub_dimensions: List[BaseDimension],
                 weight: float = 0.2,
                 config: Optional[Dict] = None):
        """
        初始化复合维度
        
        Parameters
        ----------
        name : str
            维度名称
        sub_dimensions : List[BaseDimension]
            子维度列表
        weight : float
            维度权重
        config : Dict, optional
            配置参数
        """
        super().__init__(name, weight, config)
        self.sub_dimensions = sub_dimensions
    
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算复合维度得分
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据
            
        Returns
        -------
        DimensionScore
            维度评分结果
        """
        # 计算各子维度得分
        sub_scores = []
        all_metrics = {}
        
        for sub_dim in self.sub_dimensions:
            sub_result = sub_dim.calculate_score(data)
            sub_scores.append((sub_result.score, sub_dim.weight))
            all_metrics.update(sub_result.metrics)
        
        # 加权计算总分
        total_score = sum(score * weight for score, weight in sub_scores)
        total_weight = sum(weight for _, weight in sub_scores)
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0
        
        return DimensionScore(
            dimension_name=self.name,
            score=final_score,
            weight=self.weight,
            metrics=all_metrics,
            grade=self.get_grade(final_score),
            description=self.get_description(final_score, all_metrics)
        )
    
    def extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        从数据中提取所有子维度的指标
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据
            
        Returns
        -------
        Dict[str, float]
            提取的指标
        """
        all_metrics = {}
        for sub_dim in self.sub_dimensions:
            metrics = sub_dim.extract_metrics(data)
            all_metrics.update(metrics)
        return all_metrics
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """获取默认评分标准"""
        # 复合维度使用子维度的标准
        return {}
    
    def _get_default_weights(self) -> Dict[str, float]:
        """获取默认指标权重"""
        # 复合维度使用子维度的权重
        return {}