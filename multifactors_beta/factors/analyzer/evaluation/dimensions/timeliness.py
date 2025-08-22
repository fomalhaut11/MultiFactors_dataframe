"""
时效性维度评估
评估因子的预测时效性和最优持仓周期
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.optimize import curve_fit

from .base_dimension import BaseDimension, DimensionScore
from ....tester.base import TestResult

logger = logging.getLogger(__name__)


class TimelinesDimension(BaseDimension):
    """
    时效性维度
    
    评估指标：
    1. IC衰减速度 - IC随时间的衰减率
    2. 预测持续性 - 预测能力的持续时间
    3. 最优持仓周期 - 最佳的调仓频率
    4. 信号稳定性 - 因子信号的持续性
    5. 预测半衰期 - IC下降到一半的时间
    """
    
    def __init__(self, weight: float = 0.10, config: Optional[Dict] = None):
        """
        初始化时效性维度
        
        Parameters
        ----------
        weight : float
            维度权重，默认10%
        config : Dict, optional
            配置参数
        """
        super().__init__(name="Timeliness", weight=weight, config=config)
        
        # IC衰减率阈值（每期衰减率）
        self.decay_rate_thresholds = self.config.get('decay_rate_thresholds', {
            'excellent': 0.05,   # 每期衰减小于5%
            'good': 0.10,
            'fair': 0.20,
            'poor': 0.30
        })
        
        # 预测持续期阈值（期数）
        self.persistence_thresholds = self.config.get('persistence_thresholds', {
            'excellent': 20,     # IC保持显著性超过20期
            'good': 10,
            'fair': 5,
            'poor': 2
        })
        
        # 最优持仓周期阈值（天数）
        self.optimal_holding_thresholds = self.config.get('optimal_holding_thresholds', {
            'excellent': 20,     # 最优持仓期20天以上（月度调仓）
            'good': 10,          # 10-20天（双周调仓）
            'fair': 5,           # 5-10天（周度调仓）
            'poor': 1            # 1-5天（日度调仓）
        })
        
        # 半衰期阈值（期数）
        self.halflife_thresholds = self.config.get('halflife_thresholds', {
            'excellent': 30,     # 半衰期30期以上
            'good': 15,
            'fair': 7,
            'poor': 3
        })
        
        # 信号翻转率阈值
        self.flip_rate_thresholds = self.config.get('flip_rate_thresholds', {
            'excellent': 0.10,   # 信号翻转率小于10%
            'good': 0.20,
            'fair': 0.35,
            'poor': 0.50
        })
    
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算时效性维度得分
        
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
        
        # IC衰减速度得分
        decay_rate = metrics.get('ic_decay_rate', 1.0)
        scores['decay_rate_score'] = self._score_decay_rate(decay_rate)
        
        # 预测持续性得分
        persistence = metrics.get('prediction_persistence', 0)
        scores['persistence_score'] = self._score_persistence(persistence)
        
        # 最优持仓周期得分
        optimal_holding = metrics.get('optimal_holding_period', 1)
        scores['holding_period_score'] = self._score_holding_period(optimal_holding)
        
        # 信号稳定性得分
        signal_stability = metrics.get('signal_stability', 0)
        scores['signal_stability_score'] = self._score_signal_stability(signal_stability)
        
        # 预测半衰期得分
        halflife = metrics.get('ic_halflife', 0)
        scores['halflife_score'] = self._score_halflife(halflife)
        
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
        从数据中提取时效性相关指标
        
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
            # 从IC结果提取IC衰减分析
            if test_result.ic_result:
                ic_result = test_result.ic_result
                
                # 如果有滞后IC结果
                if hasattr(ic_result, 'lag_ic') and ic_result.lag_ic is not None:
                    lag_ic = ic_result.lag_ic
                    if isinstance(lag_ic, dict) and len(lag_ic) > 1:
                        # 计算IC衰减
                        lags = sorted(lag_ic.keys())
                        ic_values = [lag_ic[lag] for lag in lags]
                        
                        # 计算衰减率
                        decay_rate = self._calculate_decay_rate(lags, ic_values)
                        metrics['ic_decay_rate'] = decay_rate
                        
                        # 计算半衰期
                        halflife = self._calculate_halflife(lags, ic_values)
                        metrics['ic_halflife'] = halflife
                        
                        # 计算预测持续性（IC保持显著的期数）
                        significance_threshold = 0.02  # IC显著性阈值
                        persistence = 0
                        for lag in lags:
                            if abs(lag_ic[lag]) >= significance_threshold:
                                persistence = lag
                            else:
                                break
                        metrics['prediction_persistence'] = persistence
                        
                        # 找到最优持仓周期（IC最大的滞后期）
                        optimal_lag = max(lag_ic.items(), key=lambda x: abs(x[1]))[0]
                        metrics['optimal_holding_period'] = optimal_lag
                
                # 从IC时间序列计算稳定性
                if ic_result.ic_series is not None and len(ic_result.ic_series) > 10:
                    ic_series = ic_result.ic_series
                    
                    # 计算IC的自相关性（反映时间稳定性）
                    ic_autocorr = ic_series.autocorr(lag=1)
                    metrics['ic_autocorrelation'] = ic_autocorr
                    
                    # 计算IC符号变化频率
                    ic_signs = np.sign(ic_series)
                    sign_changes = (ic_signs.diff() != 0).sum()
                    metrics['ic_sign_change_rate'] = sign_changes / len(ic_series)
            
            # 从分组结果提取
            if test_result.group_result:
                group_result = test_result.group_result
                
                # 分析分组的时间稳定性
                if group_result.group_returns is not None and not group_result.group_returns.empty:
                    # 计算各组排名的稳定性
                    group_ranks = group_result.group_returns.rank(axis=1, ascending=False)
                    
                    # 计算排名变化率
                    rank_changes = group_ranks.diff()
                    avg_rank_change = rank_changes.abs().mean().mean()
                    metrics['group_rank_stability'] = 1 / (1 + avg_rank_change)
            
            # 从因子值提取信号稳定性
            if test_result.processed_factor is not None:
                factor_data = test_result.processed_factor
                if isinstance(factor_data, pd.Series) and len(factor_data) > 0:
                    # 计算因子值的时间稳定性
                    if hasattr(factor_data.index, 'levels') and len(factor_data.index.levels) > 1:
                        # MultiIndex情况，计算信号翻转率
                        flip_rates = []
                        for stock in factor_data.index.get_level_values(1).unique()[:100]:  # 采样100只股票
                            stock_data = factor_data.xs(stock, level=1)
                            if len(stock_data) > 10:
                                # 计算因子值符号变化率
                                signs = np.sign(stock_data - stock_data.median())
                                sign_changes = (signs.diff() != 0).sum()
                                flip_rate = sign_changes / len(stock_data)
                                flip_rates.append(flip_rate)
                        
                        if flip_rates:
                            metrics['signal_flip_rate'] = np.mean(flip_rates)
                            metrics['signal_stability'] = 1 - np.mean(flip_rates)
        
        # 从时效性分析结果提取（如果有）
        timeliness_result = data.get('timeliness_analysis')
        if timeliness_result:
            # IC衰减分析
            ic_decay = timeliness_result.get('ic_decay')
            if ic_decay:
                metrics['ic_decay_rate'] = ic_decay.get('decay_rate', 0)
                metrics['ic_halflife'] = ic_decay.get('halflife', 0)
                metrics['optimal_lag'] = ic_decay.get('optimal_lag', 1)
            
            # 预测窗口分析
            prediction_window = timeliness_result.get('prediction_window')
            if prediction_window:
                metrics['effective_window'] = prediction_window.get('effective_periods', 1)
                metrics['window_confidence'] = prediction_window.get('confidence', 0)
        
        # 估算缺失的指标
        if 'ic_decay_rate' not in metrics:
            # 基于IC自相关估算衰减率
            ic_autocorr = metrics.get('ic_autocorrelation', 0.5)
            metrics['ic_decay_rate'] = max(0, 1 - abs(ic_autocorr))
        
        if 'ic_halflife' not in metrics:
            # 基于衰减率估算半衰期
            decay_rate = metrics.get('ic_decay_rate', 0.1)
            if decay_rate > 0:
                metrics['ic_halflife'] = np.log(0.5) / np.log(1 - decay_rate)
            else:
                metrics['ic_halflife'] = float('inf')
        
        if 'prediction_persistence' not in metrics:
            # 基于半衰期估算持续性
            halflife = metrics.get('ic_halflife', 5)
            metrics['prediction_persistence'] = min(30, halflife * 1.5)
        
        if 'optimal_holding_period' not in metrics:
            # 基于持续性估算最优持仓期
            persistence = metrics.get('prediction_persistence', 5)
            metrics['optimal_holding_period'] = max(1, persistence / 2)
        
        if 'signal_stability' not in metrics:
            # 基于信号翻转率估算稳定性
            flip_rate = metrics.get('signal_flip_rate', 0.3)
            metrics['signal_stability'] = 1 - flip_rate
        
        return metrics
    
    def _calculate_decay_rate(self, lags: List[int], ic_values: List[float]) -> float:
        """
        计算IC衰减率
        
        Parameters
        ----------
        lags : List[int]
            滞后期列表
        ic_values : List[float]
            对应的IC值列表
            
        Returns
        -------
        float
            衰减率
        """
        if len(lags) < 2:
            return 0
        
        try:
            # 使用指数衰减模型拟合: IC(t) = IC0 * exp(-lambda * t)
            def exp_decay(t, ic0, decay_rate):
                return ic0 * np.exp(-decay_rate * t)
            
            # 使用IC绝对值进行拟合
            abs_ic = [abs(ic) for ic in ic_values]
            popt, _ = curve_fit(exp_decay, lags, abs_ic, p0=[abs_ic[0], 0.1], maxfev=1000)
            
            return abs(popt[1])  # 返回衰减率
        except:
            # 如果拟合失败，使用简单的线性估算
            if abs(ic_values[0]) > 1e-6:
                return 1 - abs(ic_values[-1] / ic_values[0]) ** (1 / len(lags))
            else:
                return 0.5  # 默认衰减率
    
    def _calculate_halflife(self, lags: List[int], ic_values: List[float]) -> float:
        """
        计算IC半衰期
        
        Parameters
        ----------
        lags : List[int]
            滞后期列表
        ic_values : List[float]
            对应的IC值列表
            
        Returns
        -------
        float
            半衰期
        """
        if len(lags) < 2:
            return 0
        
        # 找到IC下降到初始值一半的位置
        initial_ic = abs(ic_values[0])
        half_ic = initial_ic / 2
        
        for i, (lag, ic) in enumerate(zip(lags, ic_values)):
            if abs(ic) <= half_ic:
                # 线性插值估算精确的半衰期
                if i > 0:
                    prev_lag = lags[i-1]
                    prev_ic = abs(ic_values[i-1])
                    curr_ic = abs(ic)
                    
                    # 线性插值
                    if prev_ic != curr_ic:
                        halflife = prev_lag + (half_ic - prev_ic) / (curr_ic - prev_ic) * (lag - prev_lag)
                        return halflife
                return lag
        
        # 如果没有下降到一半，估算半衰期
        decay_rate = self._calculate_decay_rate(lags, ic_values)
        if decay_rate > 0:
            return np.log(2) / decay_rate
        else:
            return float('inf')
    
    def _score_decay_rate(self, decay_rate: float) -> float:
        """
        计算IC衰减率得分
        
        Parameters
        ----------
        decay_rate : float
            衰减率
            
        Returns
        -------
        float
            衰减率得分(0-100)
        """
        # 衰减率越低越好
        if decay_rate <= self.decay_rate_thresholds['excellent']:
            return 100
        elif decay_rate <= self.decay_rate_thresholds['good']:
            return 80 + (self.decay_rate_thresholds['good'] - decay_rate) / \
                   (self.decay_rate_thresholds['good'] - self.decay_rate_thresholds['excellent']) * 20
        elif decay_rate <= self.decay_rate_thresholds['fair']:
            return 60 + (self.decay_rate_thresholds['fair'] - decay_rate) / \
                   (self.decay_rate_thresholds['fair'] - self.decay_rate_thresholds['good']) * 20
        elif decay_rate <= self.decay_rate_thresholds['poor']:
            return 40 + (self.decay_rate_thresholds['poor'] - decay_rate) / \
                   (self.decay_rate_thresholds['poor'] - self.decay_rate_thresholds['fair']) * 20
        else:
            return max(0, 40 * self.decay_rate_thresholds['poor'] / decay_rate)
    
    def _score_persistence(self, persistence: float) -> float:
        """
        计算预测持续性得分
        
        Parameters
        ----------
        persistence : float
            持续期数
            
        Returns
        -------
        float
            持续性得分(0-100)
        """
        # 持续性越长越好
        if persistence >= self.persistence_thresholds['excellent']:
            return 100
        elif persistence >= self.persistence_thresholds['good']:
            return 80 + (persistence - self.persistence_thresholds['good']) / \
                   (self.persistence_thresholds['excellent'] - self.persistence_thresholds['good']) * 20
        elif persistence >= self.persistence_thresholds['fair']:
            return 60 + (persistence - self.persistence_thresholds['fair']) / \
                   (self.persistence_thresholds['good'] - self.persistence_thresholds['fair']) * 20
        elif persistence >= self.persistence_thresholds['poor']:
            return 40 + (persistence - self.persistence_thresholds['poor']) / \
                   (self.persistence_thresholds['fair'] - self.persistence_thresholds['poor']) * 20
        else:
            return persistence / self.persistence_thresholds['poor'] * 40
    
    def _score_holding_period(self, holding_period: float) -> float:
        """
        计算最优持仓周期得分
        
        Parameters
        ----------
        holding_period : float
            持仓周期（天）
            
        Returns
        -------
        float
            持仓周期得分(0-100)
        """
        # 适中的持仓周期最好（太短交易成本高，太长失去时效性）
        if holding_period >= self.optimal_holding_thresholds['good'] and \
           holding_period <= self.optimal_holding_thresholds['excellent']:
            # 10-20天最优
            return 100
        elif holding_period >= self.optimal_holding_thresholds['fair'] and \
             holding_period < self.optimal_holding_thresholds['good']:
            # 5-10天较好
            return 80
        elif holding_period > self.optimal_holding_thresholds['excellent'] and \
             holding_period <= 30:
            # 20-30天可接受
            return 70
        elif holding_period >= self.optimal_holding_thresholds['poor'] and \
             holding_period < self.optimal_holding_thresholds['fair']:
            # 1-5天，频繁交易
            return 50
        elif holding_period > 30:
            # 超过30天，时效性差
            return max(20, 50 - (holding_period - 30))
        else:
            # 小于1天，过于频繁
            return 30
    
    def _score_signal_stability(self, stability: float) -> float:
        """
        计算信号稳定性得分
        
        Parameters
        ----------
        stability : float
            稳定性指标(0-1)
            
        Returns
        -------
        float
            稳定性得分(0-100)
        """
        # 使用翻转率的反向指标
        flip_rate = 1 - stability
        
        if flip_rate <= self.flip_rate_thresholds['excellent']:
            return 100
        elif flip_rate <= self.flip_rate_thresholds['good']:
            return 80 + (self.flip_rate_thresholds['good'] - flip_rate) / \
                   (self.flip_rate_thresholds['good'] - self.flip_rate_thresholds['excellent']) * 20
        elif flip_rate <= self.flip_rate_thresholds['fair']:
            return 60 + (self.flip_rate_thresholds['fair'] - flip_rate) / \
                   (self.flip_rate_thresholds['fair'] - self.flip_rate_thresholds['good']) * 20
        elif flip_rate <= self.flip_rate_thresholds['poor']:
            return 40 + (self.flip_rate_thresholds['poor'] - flip_rate) / \
                   (self.flip_rate_thresholds['poor'] - self.flip_rate_thresholds['fair']) * 20
        else:
            return max(0, 40 * (1 - flip_rate))
    
    def _score_halflife(self, halflife: float) -> float:
        """
        计算IC半衰期得分
        
        Parameters
        ----------
        halflife : float
            半衰期（期数）
            
        Returns
        -------
        float
            半衰期得分(0-100)
        """
        if halflife == float('inf'):
            return 100
        
        # 半衰期越长越好
        if halflife >= self.halflife_thresholds['excellent']:
            return 100
        elif halflife >= self.halflife_thresholds['good']:
            return 80 + (halflife - self.halflife_thresholds['good']) / \
                   (self.halflife_thresholds['excellent'] - self.halflife_thresholds['good']) * 20
        elif halflife >= self.halflife_thresholds['fair']:
            return 60 + (halflife - self.halflife_thresholds['fair']) / \
                   (self.halflife_thresholds['good'] - self.halflife_thresholds['fair']) * 20
        elif halflife >= self.halflife_thresholds['poor']:
            return 40 + (halflife - self.halflife_thresholds['poor']) / \
                   (self.halflife_thresholds['fair'] - self.halflife_thresholds['poor']) * 20
        else:
            return halflife / self.halflife_thresholds['poor'] * 40
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """
        获取默认评分标准
        
        Returns
        -------
        Dict[str, Any]
            默认评分标准
        """
        return {
            'decay_rate_thresholds': self.decay_rate_thresholds,
            'persistence_thresholds': self.persistence_thresholds,
            'optimal_holding_thresholds': self.optimal_holding_thresholds,
            'halflife_thresholds': self.halflife_thresholds,
            'flip_rate_thresholds': self.flip_rate_thresholds
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
            'decay_rate_score': 0.25,        # IC衰减率权重25%
            'persistence_score': 0.25,       # 预测持续性权重25%
            'holding_period_score': 0.20,    # 最优持仓期权重20%
            'halflife_score': 0.15,          # 半衰期权重15%
            'signal_stability_score': 0.15   # 信号稳定性权重15%
        }
    
    def _generate_description(self, score: float, metrics: Dict[str, float]) -> str:
        """
        生成时效性评价描述
        
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
        
        decay_rate = metrics.get('ic_decay_rate', 1.0)
        persistence = metrics.get('prediction_persistence', 0)
        optimal_holding = metrics.get('optimal_holding_period', 1)
        halflife = metrics.get('ic_halflife', 0)
        
        description = f"时效性维度得分{score:.1f}分(等级{grade})。"
        
        # IC衰减评价
        if decay_rate < self.decay_rate_thresholds['excellent']:
            description += f"IC衰减率({decay_rate:.2f})很低，预测效力持久。"
        elif decay_rate < self.decay_rate_thresholds['good']:
            description += f"IC衰减率({decay_rate:.2f})较低。"
        elif decay_rate > self.decay_rate_thresholds['fair']:
            description += f"IC衰减率({decay_rate:.2f})较高，预测效力衰减快。"
        
        # 持续性评价
        if persistence > self.persistence_thresholds['good']:
            description += f"预测持续性({persistence:.0f}期)良好。"
        elif persistence < self.persistence_thresholds['poor']:
            description += f"预测持续性({persistence:.0f}期)较短。"
        
        # 最优持仓期评价
        if optimal_holding >= 10 and optimal_holding <= 20:
            description += f"最优持仓期({optimal_holding:.0f}天)适中。"
        elif optimal_holding < 5:
            description += f"最优持仓期({optimal_holding:.0f}天)较短，需频繁调仓。"
        elif optimal_holding > 30:
            description += f"最优持仓期({optimal_holding:.0f}天)较长。"
        
        # 半衰期评价
        if halflife > self.halflife_thresholds['good']:
            description += f"IC半衰期({halflife:.0f}期)较长。"
        
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
        
        # 检查是否有TestResult或时效性分析结果
        test_result = data.get('test_result')
        timeliness_result = data.get('timeliness_analysis')
        
        if not isinstance(test_result, TestResult) and not timeliness_result:
            logger.warning("No TestResult or timeliness analysis found in data")
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
            description="无法计算时效性得分，数据不足"
        )