"""
因子综合评估器
整合各维度评估，提供因子的全面评价
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from ..base import AnalyzerBase
from ...tester.base import TestResult
from .scoring.score_calculator import ScoreCalculator, ScoreResult
from .dimensions.base_dimension import BaseDimension, DimensionScore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """综合评估结果"""
    # 基本信息
    factor_name: str
    evaluation_time: datetime
    scenario: str
    
    # 维度得分 (0-100)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    # 综合评估
    total_score: float = 0.0
    grade: str = 'C'
    rank: Optional[int] = None
    percentile: Optional[float] = None
    
    # 详细指标
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 诊断信息
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # 推荐信息
    recommendation: Dict[str, Any] = field(default_factory=dict)
    
    # 原始数据引用
    test_result: Optional[TestResult] = None
    correlation_result: Optional[Dict] = None
    stability_result: Optional[Dict] = None


class FactorEvaluator(AnalyzerBase):
    """
    因子综合评估器
    
    功能：
    1. 多维度综合评估
    2. 场景化评分策略
    3. 因子诊断和建议
    4. 批量评估和排名
    """
    
    def __init__(self, 
                 scenario: str = 'balanced',
                 config: Optional[Dict] = None):
        """
        初始化评估器
        
        Parameters
        ----------
        scenario : str
            评估场景 (balanced/high_frequency/value_investing/risk_neutral)
        config : Dict, optional
            配置参数
        """
        super().__init__(name="FactorEvaluator", config=config)
        
        self.scenario = scenario
        
        # 初始化评分计算器
        self.score_calculator = ScoreCalculator(scenario=scenario, config=config)
        
        # 初始化维度评估器
        self.dimensions = self._initialize_dimensions()
        
        # 评估结果缓存
        self.evaluation_results = {}
        
        # 评估标准
        self.min_ic = self.config.get('min_ic', 0.02)
        self.min_icir = self.config.get('min_icir', 0.3)
        self.max_turnover = self.config.get('max_turnover', 0.8)
        
        logger.info(f"FactorEvaluator initialized with scenario '{scenario}'")
    
    def _initialize_dimensions(self) -> Dict[str, BaseDimension]:
        """
        初始化评估维度
        
        Returns
        -------
        Dict[str, BaseDimension]
            维度评估器字典
        """
        dimensions = {}
        
        # 这里先创建占位符，实际维度实现在后续完成
        # Phase 2会实现具体的维度类
        
        logger.info("Dimensions initialized (placeholder for now)")
        return dimensions
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        执行分析（实现基类接口）
        
        这个方法是为了兼容AnalyzerBase，实际使用evaluate方法
        """
        if isinstance(data, TestResult):
            return self.evaluate(data, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
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
        if isinstance(data, TestResult):
            return data.performance_metrics is not None
        return False
    
    def evaluate(self,
                test_result: TestResult,
                correlation_result: Optional[Dict] = None,
                stability_result: Optional[Dict] = None,
                **kwargs) -> EvaluationResult:
        """
        评估单个因子
        
        Parameters
        ----------
        test_result : TestResult
            因子测试结果
        correlation_result : Dict, optional
            相关性分析结果
        stability_result : Dict, optional
            稳定性分析结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        EvaluationResult
            综合评估结果
        """
        self.analysis_time = datetime.now()
        
        # 提取因子名称
        factor_name = test_result.factor_name or 'unknown'
        
        logger.info(f"Evaluating factor: {factor_name}")
        
        # 准备数据
        eval_data = {
            'test_result': test_result,
            'correlation_result': correlation_result,
            'stability_result': stability_result
        }
        
        # 计算各维度得分
        dimension_scores = self._calculate_dimension_scores(eval_data)
        
        # 计算综合得分
        score_result = self.score_calculator.calculate(dimension_scores)
        
        # 生成诊断信息
        diagnostics = self._generate_diagnostics(
            test_result, 
            dimension_scores, 
            score_result
        )
        
        # 生成推荐信息
        recommendation = self._generate_recommendation(
            factor_name,
            score_result,
            diagnostics
        )
        
        # 创建评估结果
        evaluation_result = EvaluationResult(
            factor_name=factor_name,
            evaluation_time=self.analysis_time,
            scenario=self.scenario,
            dimension_scores=dimension_scores,
            total_score=score_result.total_score,
            grade=score_result.grade,
            percentile=score_result.percentile,
            metrics=self._extract_all_metrics(eval_data),
            strengths=diagnostics['strengths'],
            weaknesses=diagnostics['weaknesses'],
            warnings=diagnostics['warnings'],
            suggestions=diagnostics['suggestions'],
            recommendation=recommendation,
            test_result=test_result,
            correlation_result=correlation_result,
            stability_result=stability_result
        )
        
        # 缓存结果
        self.evaluation_results[factor_name] = evaluation_result
        
        logger.info(f"Evaluation completed: {factor_name} - Score: {score_result.total_score:.1f}, Grade: {score_result.grade}")
        
        return evaluation_result
    
    def _calculate_dimension_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        计算各维度得分
        
        Parameters
        ----------
        data : Dict[str, Any]
            评估数据
            
        Returns
        -------
        Dict[str, float]
            维度得分字典
        """
        dimension_scores = {}
        
        # 从测试结果中提取得分（临时实现，Phase 2会使用真正的维度评估器）
        test_result = data.get('test_result')
        
        if test_result and test_result.performance_metrics:
            metrics = test_result.performance_metrics
            
            # 收益能力维度
            ic_mean = metrics.get('ic_mean', 0)
            icir = metrics.get('icir', 0)
            sharpe = metrics.get('long_short_sharpe', 0)
            
            profitability_score = self._score_profitability(ic_mean, icir, sharpe)
            dimension_scores['profitability'] = profitability_score
            
            # 稳定性维度
            if data.get('stability_result'):
                stability_score = data['stability_result'].get('stability_score', 50)
            else:
                # 简单估算
                stability_score = 50 + min(30, icir * 30)
            dimension_scores['stability'] = stability_score
            
            # 可交易性维度
            turnover = metrics.get('avg_turnover', 0.5)
            tradability_score = self._score_tradability(turnover)
            dimension_scores['tradability'] = tradability_score
            
            # 独特性维度（需要相关性分析结果）
            if data.get('correlation_result'):
                high_corr_pairs = data['correlation_result'].get('high_corr_pairs', [])
                uniqueness_score = max(20, 80 - len(high_corr_pairs) * 10)
            else:
                uniqueness_score = 60  # 默认中等
            dimension_scores['uniqueness'] = uniqueness_score
            
            # 时效性维度
            timeliness_score = 60  # 默认中等，需要IC衰减分析
            dimension_scores['timeliness'] = timeliness_score
        
        return dimension_scores
    
    def _score_profitability(self, ic_mean: float, icir: float, sharpe: float) -> float:
        """
        计算收益能力得分
        
        Parameters
        ----------
        ic_mean : float
            IC均值
        icir : float
            ICIR
        sharpe : float
            夏普比率
            
        Returns
        -------
        float
            收益能力得分(0-100)
        """
        # IC得分
        if ic_mean >= 0.05:
            ic_score = 100
        elif ic_mean >= 0.04:
            ic_score = 80
        elif ic_mean >= 0.03:
            ic_score = 60
        elif ic_mean >= 0.02:
            ic_score = 40
        else:
            ic_score = max(0, ic_mean * 1000)
        
        # ICIR得分
        if icir >= 0.7:
            icir_score = 100
        elif icir >= 0.5:
            icir_score = 80
        elif icir >= 0.3:
            icir_score = 60
        else:
            icir_score = max(0, icir * 100)
        
        # 夏普得分
        if sharpe >= 1.5:
            sharpe_score = 100
        elif sharpe >= 1.0:
            sharpe_score = 80
        elif sharpe >= 0.5:
            sharpe_score = 60
        else:
            sharpe_score = max(0, sharpe * 40)
        
        # 加权平均
        return ic_score * 0.4 + icir_score * 0.4 + sharpe_score * 0.2
    
    def _score_tradability(self, turnover: float) -> float:
        """
        计算可交易性得分
        
        Parameters
        ----------
        turnover : float
            换手率
            
        Returns
        -------
        float
            可交易性得分(0-100)
        """
        if turnover <= 0.2:
            return 100
        elif turnover <= 0.4:
            return 80
        elif turnover <= 0.6:
            return 60
        elif turnover <= 0.8:
            return 40
        else:
            return max(0, 100 - turnover * 100)
    
    def _extract_all_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取所有指标
        
        Parameters
        ----------
        data : Dict[str, Any]
            评估数据
            
        Returns
        -------
        Dict[str, Any]
            所有指标
        """
        metrics = {}
        
        # 从测试结果提取
        if data.get('test_result'):
            test_result = data['test_result']
            if test_result.performance_metrics:
                metrics.update(test_result.performance_metrics)
        
        # 从相关性分析提取
        if data.get('correlation_result'):
            corr = data['correlation_result']
            if 'statistics' in corr:
                metrics['correlation_stats'] = corr['statistics']
        
        # 从稳定性分析提取
        if data.get('stability_result'):
            stab = data['stability_result']
            if 'basic_metrics' in stab:
                metrics['stability_metrics'] = stab['basic_metrics']
        
        return metrics
    
    def _generate_diagnostics(self,
                            test_result: TestResult,
                            dimension_scores: Dict[str, float],
                            score_result: ScoreResult) -> Dict[str, List[str]]:
        """
        生成诊断信息
        
        Parameters
        ----------
        test_result : TestResult
            测试结果
        dimension_scores : Dict[str, float]
            维度得分
        score_result : ScoreResult
            评分结果
            
        Returns
        -------
        Dict[str, List[str]]
            诊断信息
        """
        diagnostics = {
            'strengths': [],
            'weaknesses': [],
            'warnings': [],
            'suggestions': []
        }
        
        # 识别优势
        for dim, score in dimension_scores.items():
            if score >= 80:
                diagnostics['strengths'].append(f"{dim}维度表现优秀({score:.1f}分)")
        
        # 识别劣势
        for dim, score in dimension_scores.items():
            if score < 50:
                diagnostics['weaknesses'].append(f"{dim}维度需要改进({score:.1f}分)")
        
        # 生成预警
        if test_result.performance_metrics:
            metrics = test_result.performance_metrics
            
            if metrics.get('ic_mean', 0) < self.min_ic:
                diagnostics['warnings'].append(f"IC均值低于阈值({self.min_ic})")
            
            if metrics.get('icir', 0) < self.min_icir:
                diagnostics['warnings'].append(f"ICIR低于阈值({self.min_icir})")
            
            if metrics.get('avg_turnover', 0) > self.max_turnover:
                diagnostics['warnings'].append(f"换手率过高(>{self.max_turnover})")
        
        # 生成建议
        if score_result.total_score < 60:
            diagnostics['suggestions'].append("建议优化因子计算逻辑或参数")
        
        if dimension_scores.get('tradability', 100) < 50:
            diagnostics['suggestions'].append("考虑降低换手率，减少交易成本")
        
        if dimension_scores.get('stability', 100) < 50:
            diagnostics['suggestions'].append("增加样本期或优化数据处理方法")
        
        return diagnostics
    
    def _generate_recommendation(self,
                                factor_name: str,
                                score_result: ScoreResult,
                                diagnostics: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        生成推荐信息
        
        Parameters
        ----------
        factor_name : str
            因子名称
        score_result : ScoreResult
            评分结果
        diagnostics : Dict[str, List[str]]
            诊断信息
            
        Returns
        -------
        Dict[str, Any]
            推荐信息
        """
        recommendation = {
            'usage': '',
            'weight': 0,
            'priority': '',
            'combine_with': []
        }
        
        # 根据评分确定使用建议
        if score_result.total_score >= 80:
            recommendation['usage'] = '强烈推荐使用'
            recommendation['weight'] = 0.15
            recommendation['priority'] = 'high'
        elif score_result.total_score >= 60:
            recommendation['usage'] = '可以使用'
            recommendation['weight'] = 0.1
            recommendation['priority'] = 'medium'
        else:
            recommendation['usage'] = '谨慎使用'
            recommendation['weight'] = 0.05
            recommendation['priority'] = 'low'
        
        # 根据场景调整
        if self.scenario == 'high_frequency':
            if score_result.dimension_scores.get('tradability', 0) < 60:
                recommendation['usage'] = '不适合高频交易'
                recommendation['weight'] *= 0.5
        
        return recommendation
    
    def batch_evaluate(self,
                      factors: Dict[str, TestResult],
                      correlation_results: Optional[Dict] = None,
                      stability_results: Optional[Dict] = None,
                      **kwargs) -> Dict[str, EvaluationResult]:
        """
        批量评估因子
        
        Parameters
        ----------
        factors : Dict[str, TestResult]
            因子测试结果字典
        correlation_results : Dict, optional
            相关性分析结果
        stability_results : Dict, optional
            稳定性分析结果字典
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, EvaluationResult]
            评估结果字典
        """
        results = {}
        
        for factor_name, test_result in factors.items():
            # 获取对应的分析结果
            corr_result = None
            if correlation_results and factor_name in correlation_results:
                corr_result = correlation_results[factor_name]
            
            stab_result = None
            if stability_results and factor_name in stability_results:
                stab_result = stability_results[factor_name]
            
            # 评估
            results[factor_name] = self.evaluate(
                test_result,
                correlation_result=corr_result,
                stability_result=stab_result,
                **kwargs
            )
        
        # 更新排名
        self._update_rankings(results)
        
        return results
    
    def _update_rankings(self, results: Dict[str, EvaluationResult]):
        """
        更新因子排名
        
        Parameters
        ----------
        results : Dict[str, EvaluationResult]
            评估结果字典
        """
        # 按总分排序
        sorted_factors = sorted(
            results.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )
        
        # 更新排名和百分位
        n = len(sorted_factors)
        for i, (factor_name, result) in enumerate(sorted_factors):
            result.rank = i + 1
            result.percentile = ((n - i) / n) * 100 if n > 0 else 50
    
    def compare_factors(self,
                       evaluation_results: Dict[str, EvaluationResult],
                       dimensions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        因子对比分析
        
        Parameters
        ----------
        evaluation_results : Dict[str, EvaluationResult]
            评估结果字典
        dimensions : List[str], optional
            要对比的维度
            
        Returns
        -------
        pd.DataFrame
            对比结果表
        """
        if not evaluation_results:
            return pd.DataFrame()
        
        # 默认对比所有维度
        if dimensions is None:
            dimensions = ['profitability', 'stability', 'tradability', 
                         'uniqueness', 'timeliness']
        
        # 构建对比数据
        comparison_data = []
        
        for factor_name, result in evaluation_results.items():
            row = {
                'factor': factor_name,
                'total_score': result.total_score,
                'grade': result.grade,
                'rank': result.rank
            }
            
            # 添加维度得分
            for dim in dimensions:
                row[f'{dim}_score'] = result.dimension_scores.get(dim, 0)
            
            comparison_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        
        # 按总分排序
        df = df.sort_values('total_score', ascending=False)
        
        return df
    
    def rank_factors(self,
                    evaluation_results: Dict[str, EvaluationResult],
                    by: str = 'total_score',
                    ascending: bool = False) -> pd.DataFrame:
        """
        因子排名
        
        Parameters
        ----------
        evaluation_results : Dict[str, EvaluationResult]
            评估结果字典
        by : str
            排序依据
        ascending : bool
            是否升序
            
        Returns
        -------
        pd.DataFrame
            排名结果表
        """
        if not evaluation_results:
            return pd.DataFrame()
        
        # 构建排名数据
        ranking_data = []
        
        for factor_name, result in evaluation_results.items():
            row = {
                'rank': result.rank,
                'factor': factor_name,
                'total_score': result.total_score,
                'grade': result.grade,
                'percentile': result.percentile,
                'strengths_count': len(result.strengths),
                'weaknesses_count': len(result.weaknesses),
                'recommendation': result.recommendation.get('usage', '')
            }
            ranking_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(ranking_data)
        
        # 排序
        if by in df.columns:
            df = df.sort_values(by, ascending=ascending)
        
        return df
    
    def recommend_top_factors(self,
                            n: int = 10,
                            min_score: float = 60) -> List[str]:
        """
        推荐顶级因子
        
        Parameters
        ----------
        n : int
            推荐数量
        min_score : float
            最低分数要求
            
        Returns
        -------
        List[str]
            推荐的因子列表
        """
        # 筛选符合条件的因子
        qualified_factors = [
            (name, result.total_score)
            for name, result in self.evaluation_results.items()
            if result.total_score >= min_score
        ]
        
        # 按分数排序
        qualified_factors.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前n个
        return [name for name, _ in qualified_factors[:n]]
    
    def generate_evaluation_report(self, 
                                  evaluation_result: EvaluationResult) -> str:
        """
        生成评估报告
        
        Parameters
        ----------
        evaluation_result : EvaluationResult
            评估结果
            
        Returns
        -------
        str
            文本报告
        """
        lines = [
            "=" * 60,
            f"因子综合评估报告 - {evaluation_result.factor_name}",
            "=" * 60,
            f"评估时间: {evaluation_result.evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"评估场景: {evaluation_result.scenario}",
            "",
            "【综合评分】",
            f"总分: {evaluation_result.total_score:.1f}/100",
            f"等级: {evaluation_result.grade}",
        ]
        
        if evaluation_result.rank:
            lines.append(f"排名: 第{evaluation_result.rank}位")
        if evaluation_result.percentile:
            lines.append(f"百分位: {evaluation_result.percentile:.1f}%")
        
        lines.extend([
            "",
            "【维度得分】",
        ])
        
        for dim, score in evaluation_result.dimension_scores.items():
            lines.append(f"  {dim:15s}: {score:6.1f}分")
        
        lines.extend([
            "",
            "【优势】",
        ])
        for strength in evaluation_result.strengths:
            lines.append(f"  • {strength}")
        
        if evaluation_result.weaknesses:
            lines.extend([
                "",
                "【劣势】",
            ])
            for weakness in evaluation_result.weaknesses:
                lines.append(f"  • {weakness}")
        
        if evaluation_result.warnings:
            lines.extend([
                "",
                "【预警】",
            ])
            for warning in evaluation_result.warnings:
                lines.append(f"  ⚠ {warning}")
        
        if evaluation_result.suggestions:
            lines.extend([
                "",
                "【建议】",
            ])
            for suggestion in evaluation_result.suggestions:
                lines.append(f"  → {suggestion}")
        
        lines.extend([
            "",
            "【使用推荐】",
            f"  推荐度: {evaluation_result.recommendation.get('usage', 'N/A')}",
            f"  建议权重: {evaluation_result.recommendation.get('weight', 0):.1%}",
            f"  优先级: {evaluation_result.recommendation.get('priority', 'N/A')}",
        ])
        
        return "\n".join(lines)