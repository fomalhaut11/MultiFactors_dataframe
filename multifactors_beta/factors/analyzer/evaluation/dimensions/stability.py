"""
稳定性维度评估
评估因子的稳定性和一致性表现
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import logging
from scipy import stats

from .base_dimension import BaseDimension, DimensionScore
from ....tester.base import TestResult
from ...stability.stability_analyzer import StabilityAnalyzer

logger = logging.getLogger(__name__)


class StabilityDimension(BaseDimension):
    """
    稳定性维度
    
    评估指标：
    1. IC稳定性 - IC序列的波动性
    2. 收益稳定性 - 收益序列的稳定性
    3. 因子值稳定性 - 因子值的自相关性和持续性
    4. 市场状态稳定性 - 不同市场环境下的表现一致性
    5. 结构突变检测 - 检测因子是否存在结构性变化
    """
    
    def __init__(self, weight: float = 0.25, config: Optional[Dict] = None):
        """
        初始化稳定性维度
        
        Parameters
        ----------
        weight : float
            维度权重，默认25%
        config : Dict, optional
            配置参数
        """
        super().__init__(name="Stability", weight=weight, config=config)
        
        # 稳定性评估阈值
        self.ic_volatility_thresholds = self.config.get('ic_volatility_thresholds', {
            'excellent': 0.10,   # IC波动率小于10%
            'good': 0.15,
            'fair': 0.20,
            'poor': 0.30
        })
        
        self.autocorr_thresholds = self.config.get('autocorr_thresholds', {
            'excellent': 0.7,    # 自相关系数大于0.7
            'good': 0.5,
            'fair': 0.3,
            'poor': 0.1
        })
        
        self.consistency_thresholds = self.config.get('consistency_thresholds', {
            'excellent': 0.85,   # 一致性得分大于85%
            'good': 0.70,
            'fair': 0.55,
            'poor': 0.40
        })
        
        # 初始化稳定性分析器
        self.stability_analyzer = StabilityAnalyzer(config=config)
    
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算稳定性维度得分
        
        Parameters
        ----------
        data : Dict[str, Any]
            包含TestResult等数据
            
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
        
        # IC稳定性得分
        ic_volatility = metrics.get('ic_volatility', 1.0)
        scores['ic_stability_score'] = self._score_ic_stability(ic_volatility)
        
        # 收益稳定性得分
        return_volatility = metrics.get('return_volatility', 1.0)
        scores['return_stability_score'] = self._score_return_stability(return_volatility)
        
        # 因子值稳定性得分（自相关性）
        autocorrelation = metrics.get('factor_autocorrelation', 0)
        scores['factor_stability_score'] = self._score_factor_stability(autocorrelation)
        
        # 市场状态一致性得分
        market_consistency = metrics.get('market_state_consistency', 0)
        scores['market_consistency_score'] = self._score_market_consistency(market_consistency)
        
        # 结构稳定性得分（无结构突变）
        has_structural_break = metrics.get('has_structural_break', True)
        scores['structural_stability_score'] = 100 if not has_structural_break else 40
        
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
        从数据中提取稳定性相关指标
        
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
        
        # 从TestResult提取
        test_result = data.get('test_result')
        if isinstance(test_result, TestResult):
            # IC稳定性指标
            if test_result.ic_result:
                ic_result = test_result.ic_result
                if ic_result.ic_series is not None and len(ic_result.ic_series) > 0:
                    # IC波动率（标准差/均值的绝对值）
                    ic_mean = ic_result.ic_mean
                    ic_std = ic_result.ic_std
                    if abs(ic_mean) > 1e-6:
                        metrics['ic_volatility'] = ic_std / abs(ic_mean)
                    else:
                        metrics['ic_volatility'] = float('inf')
                    
                    # IC序列的自相关性
                    if len(ic_result.ic_series) > 10:
                        metrics['ic_autocorrelation'] = ic_result.ic_series.autocorr(lag=1)
                    
                    # IC正值比例
                    metrics['ic_positive_ratio'] = (ic_result.ic_series > 0).mean()
                    
                    # IC序列的偏度和峰度
                    metrics['ic_skewness'] = ic_result.ic_series.skew()
                    metrics['ic_kurtosis'] = ic_result.ic_series.kurtosis()
            
            # 分组收益稳定性
            if test_result.group_result:
                group_result = test_result.group_result
                if group_result.long_short_return is not None and len(group_result.long_short_return) > 0:
                    # 收益波动率
                    return_mean = group_result.long_short_return.mean()
                    return_std = group_result.long_short_return.std()
                    if abs(return_mean) > 1e-6:
                        metrics['return_volatility'] = return_std / abs(return_mean)
                    else:
                        metrics['return_volatility'] = float('inf')
                    
                    # 收益序列的自相关性
                    if len(group_result.long_short_return) > 10:
                        metrics['return_autocorrelation'] = group_result.long_short_return.autocorr(lag=1)
                    
                    # 最大回撤
                    cumulative_returns = (1 + group_result.long_short_return).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    metrics['max_drawdown'] = drawdown.min()
            
            # 因子值稳定性（需要原始因子数据）
            if test_result.processed_factor is not None:
                factor_data = test_result.processed_factor
                if isinstance(factor_data, pd.Series) and len(factor_data) > 0:
                    # 计算因子值的时间序列自相关性
                    if hasattr(factor_data.index, 'levels') and len(factor_data.index.levels) > 1:
                        # MultiIndex情况，按股票计算自相关性
                        autocorrs = []
                        for stock in factor_data.index.get_level_values(1).unique()[:100]:  # 采样100只股票
                            stock_data = factor_data.xs(stock, level=1)
                            if len(stock_data) > 10:
                                autocorr = stock_data.autocorr(lag=1)
                                if not np.isnan(autocorr):
                                    autocorrs.append(autocorr)
                        if autocorrs:
                            metrics['factor_autocorrelation'] = np.mean(autocorrs)
        
        # 从稳定性分析结果提取（如果有）
        stability_result = data.get('stability_analysis')
        if stability_result:
            # IC稳定性
            ic_stability = stability_result.get('ic_stability', {})
            if ic_stability:
                metrics['ic_rolling_std'] = ic_stability.get('rolling_std_mean', 0)
                metrics['ic_trend'] = ic_stability.get('trend_coefficient', 0)
            
            # 结构突变
            structural_breaks = stability_result.get('structural_breaks', {})
            if structural_breaks:
                metrics['has_structural_break'] = structural_breaks.get('has_break', False)
                metrics['break_confidence'] = structural_breaks.get('confidence', 0)
            
            # 市场状态一致性
            market_states = stability_result.get('market_state_analysis', {})
            if market_states:
                state_performance = market_states.get('state_performance', {})
                if state_performance:
                    # 计算不同市场状态下IC的一致性
                    ic_values = [perf.get('mean_ic', 0) for perf in state_performance.values()]
                    if ic_values:
                        # 一致性定义为：1 - 标准差/均值
                        ic_mean = np.mean(ic_values)
                        ic_std = np.std(ic_values)
                        if abs(ic_mean) > 1e-6:
                            metrics['market_state_consistency'] = max(0, 1 - ic_std / abs(ic_mean))
                        else:
                            metrics['market_state_consistency'] = 0
        
        return metrics
    
    def _score_ic_stability(self, volatility: float) -> float:
        """
        计算IC稳定性得分
        
        Parameters
        ----------
        volatility : float
            IC波动率（标准差/均值）
            
        Returns
        -------
        float
            IC稳定性得分(0-100)
        """
        if volatility == float('inf'):
            return 0
        
        # 波动率越小越好
        if volatility <= self.ic_volatility_thresholds['excellent']:
            return 100
        elif volatility <= self.ic_volatility_thresholds['good']:
            return 80 + (self.ic_volatility_thresholds['good'] - volatility) / \
                   (self.ic_volatility_thresholds['good'] - self.ic_volatility_thresholds['excellent']) * 20
        elif volatility <= self.ic_volatility_thresholds['fair']:
            return 60 + (self.ic_volatility_thresholds['fair'] - volatility) / \
                   (self.ic_volatility_thresholds['fair'] - self.ic_volatility_thresholds['good']) * 20
        elif volatility <= self.ic_volatility_thresholds['poor']:
            return 40 + (self.ic_volatility_thresholds['poor'] - volatility) / \
                   (self.ic_volatility_thresholds['poor'] - self.ic_volatility_thresholds['fair']) * 20
        else:
            return max(0, 40 * self.ic_volatility_thresholds['poor'] / volatility)
    
    def _score_return_stability(self, volatility: float) -> float:
        """
        计算收益稳定性得分
        
        Parameters
        ----------
        volatility : float
            收益波动率
            
        Returns
        -------
        float
            收益稳定性得分(0-100)
        """
        if volatility == float('inf'):
            return 0
        
        # 使用相似的逻辑，但阈值稍微宽松
        thresholds = {
            'excellent': 0.15,
            'good': 0.25,
            'fair': 0.35,
            'poor': 0.50
        }
        
        if volatility <= thresholds['excellent']:
            return 100
        elif volatility <= thresholds['good']:
            return 80 + (thresholds['good'] - volatility) / \
                   (thresholds['good'] - thresholds['excellent']) * 20
        elif volatility <= thresholds['fair']:
            return 60 + (thresholds['fair'] - volatility) / \
                   (thresholds['fair'] - thresholds['good']) * 20
        elif volatility <= thresholds['poor']:
            return 40 + (thresholds['poor'] - volatility) / \
                   (thresholds['poor'] - thresholds['fair']) * 20
        else:
            return max(0, 40 * thresholds['poor'] / volatility)
    
    def _score_factor_stability(self, autocorrelation: float) -> float:
        """
        计算因子值稳定性得分
        
        Parameters
        ----------
        autocorrelation : float
            因子值自相关系数
            
        Returns
        -------
        float
            因子稳定性得分(0-100)
        """
        # 自相关性越高越稳定
        if autocorrelation >= self.autocorr_thresholds['excellent']:
            return 100
        elif autocorrelation >= self.autocorr_thresholds['good']:
            return 80 + (autocorrelation - self.autocorr_thresholds['good']) / \
                   (self.autocorr_thresholds['excellent'] - self.autocorr_thresholds['good']) * 20
        elif autocorrelation >= self.autocorr_thresholds['fair']:
            return 60 + (autocorrelation - self.autocorr_thresholds['fair']) / \
                   (self.autocorr_thresholds['good'] - self.autocorr_thresholds['fair']) * 20
        elif autocorrelation >= self.autocorr_thresholds['poor']:
            return 40 + (autocorrelation - self.autocorr_thresholds['poor']) / \
                   (self.autocorr_thresholds['fair'] - self.autocorr_thresholds['poor']) * 20
        elif autocorrelation > 0:
            return autocorrelation / self.autocorr_thresholds['poor'] * 40
        else:
            return 0
    
    def _score_market_consistency(self, consistency: float) -> float:
        """
        计算市场状态一致性得分
        
        Parameters
        ----------
        consistency : float
            一致性指标(0-1)
            
        Returns
        -------
        float
            一致性得分(0-100)
        """
        if consistency >= self.consistency_thresholds['excellent']:
            return 100
        elif consistency >= self.consistency_thresholds['good']:
            return 80 + (consistency - self.consistency_thresholds['good']) / \
                   (self.consistency_thresholds['excellent'] - self.consistency_thresholds['good']) * 20
        elif consistency >= self.consistency_thresholds['fair']:
            return 60 + (consistency - self.consistency_thresholds['fair']) / \
                   (self.consistency_thresholds['good'] - self.consistency_thresholds['fair']) * 20
        elif consistency >= self.consistency_thresholds['poor']:
            return 40 + (consistency - self.consistency_thresholds['poor']) / \
                   (self.consistency_thresholds['fair'] - self.consistency_thresholds['poor']) * 20
        else:
            return consistency / self.consistency_thresholds['poor'] * 40
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """
        获取默认评分标准
        
        Returns
        -------
        Dict[str, Any]
            默认评分标准
        """
        return {
            'ic_volatility_thresholds': self.ic_volatility_thresholds,
            'autocorr_thresholds': self.autocorr_thresholds,
            'consistency_thresholds': self.consistency_thresholds
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
            'ic_stability_score': 0.30,           # IC稳定性权重30%
            'return_stability_score': 0.25,       # 收益稳定性权重25%
            'factor_stability_score': 0.20,       # 因子值稳定性权重20%
            'market_consistency_score': 0.15,     # 市场一致性权重15%
            'structural_stability_score': 0.10    # 结构稳定性权重10%
        }
    
    def _generate_description(self, score: float, metrics: Dict[str, float]) -> str:
        """
        生成稳定性评价描述
        
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
        
        ic_volatility = metrics.get('ic_volatility', float('inf'))
        autocorr = metrics.get('factor_autocorrelation', 0)
        has_break = metrics.get('has_structural_break', False)
        
        description = f"稳定性维度得分{score:.1f}分(等级{grade})。"
        
        # IC稳定性评价
        if ic_volatility < self.ic_volatility_thresholds['excellent']:
            description += f"IC波动率({ic_volatility:.2f})很低，预测稳定性极佳。"
        elif ic_volatility < self.ic_volatility_thresholds['good']:
            description += f"IC波动率({ic_volatility:.2f})较低，预测相对稳定。"
        elif ic_volatility < self.ic_volatility_thresholds['fair']:
            description += f"IC波动率({ic_volatility:.2f})中等。"
        else:
            description += f"IC波动率({ic_volatility:.2f})较高，稳定性欠佳。"
        
        # 因子值稳定性评价
        if autocorr > self.autocorr_thresholds['excellent']:
            description += f"因子值自相关性({autocorr:.2f})很强，持续性好。"
        elif autocorr > self.autocorr_thresholds['good']:
            description += f"因子值自相关性({autocorr:.2f})较好。"
        
        # 结构稳定性评价
        if has_break:
            description += "检测到结构性突变，需关注因子失效风险。"
        else:
            description += "未检测到结构性突变。"
        
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
        
        # 检查是否有TestResult或稳定性分析结果
        test_result = data.get('test_result')
        stability_result = data.get('stability_analysis')
        
        if not isinstance(test_result, TestResult) and not stability_result:
            logger.warning("No TestResult or stability analysis found in data")
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
            description="无法计算稳定性得分，数据不足"
        )