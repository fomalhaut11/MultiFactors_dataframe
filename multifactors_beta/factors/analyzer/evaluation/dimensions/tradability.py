"""
可交易性维度评估
评估因子的实际可交易性和实施成本
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

from .base_dimension import BaseDimension, DimensionScore
from ....tester.base import TestResult

logger = logging.getLogger(__name__)


class TradabilityDimension(BaseDimension):
    """
    可交易性维度
    
    评估指标：
    1. 换手率 - 因子调仓频率
    2. 交易成本 - 预估交易成本占比
    3. 流动性 - 可投资股票的流动性
    4. 容量 - 策略容量限制
    5. 股票覆盖度 - 可投资股票数量
    """
    
    def __init__(self, weight: float = 0.20, config: Optional[Dict] = None):
        """
        初始化可交易性维度
        
        Parameters
        ----------
        weight : float
            维度权重，默认20%
        config : Dict, optional
            配置参数
        """
        super().__init__(name="Tradability", weight=weight, config=config)
        
        # 换手率阈值（年化）
        self.turnover_thresholds = self.config.get('turnover_thresholds', {
            'excellent': 2.0,    # 年换手率小于200%
            'good': 4.0,         # 年换手率小于400%
            'fair': 8.0,         # 年换手率小于800%
            'poor': 12.0         # 年换手率小于1200%
        })
        
        # 交易成本阈值（占收益比例）
        self.cost_ratio_thresholds = self.config.get('cost_ratio_thresholds', {
            'excellent': 0.10,   # 交易成本占收益10%以下
            'good': 0.20,
            'fair': 0.35,
            'poor': 0.50
        })
        
        # 覆盖度阈值（可投资股票占比）
        self.coverage_thresholds = self.config.get('coverage_thresholds', {
            'excellent': 0.80,   # 覆盖80%以上股票
            'good': 0.60,
            'fair': 0.40,
            'poor': 0.20
        })
        
        # 流动性阈值（平均日成交额，亿元）
        self.liquidity_thresholds = self.config.get('liquidity_thresholds', {
            'excellent': 5.0,    # 平均日成交额5亿以上
            'good': 2.0,
            'fair': 1.0,
            'poor': 0.5
        })
        
        # 默认交易成本参数
        self.transaction_cost = self.config.get('transaction_cost', {
            'commission': 0.0005,    # 佣金率
            'slippage': 0.001,       # 滑点
            'stamp_tax': 0.001       # 印花税（卖出）
        })
    
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算可交易性维度得分
        
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
        
        # 换手率得分
        turnover = metrics.get('annual_turnover', float('inf'))
        scores['turnover_score'] = self._score_turnover(turnover)
        
        # 交易成本得分
        cost_ratio = metrics.get('cost_to_return_ratio', 1.0)
        scores['cost_score'] = self._score_cost_ratio(cost_ratio)
        
        # 股票覆盖度得分
        coverage = metrics.get('stock_coverage', 0)
        scores['coverage_score'] = self._score_coverage(coverage)
        
        # 流动性得分
        avg_liquidity = metrics.get('avg_liquidity', 0)
        scores['liquidity_score'] = self._score_liquidity(avg_liquidity)
        
        # 容量得分
        capacity_limit = metrics.get('capacity_limit', 0)
        scores['capacity_score'] = self._score_capacity(capacity_limit)
        
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
        从数据中提取可交易性相关指标
        
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
            # 从分组结果提取换手率
            if test_result.group_result:
                group_result = test_result.group_result
                
                # 换手率（如果有）
                if hasattr(group_result, 'turnover') and group_result.turnover is not None:
                    # 假设是日度换手率，年化
                    daily_turnover = group_result.turnover
                    if isinstance(daily_turnover, (pd.Series, pd.DataFrame)):
                        daily_turnover = daily_turnover.mean()
                    metrics['annual_turnover'] = daily_turnover * 252
                else:
                    # 估算换手率（基于分组重新平衡）
                    # 假设每期完全换手（最坏情况）
                    if hasattr(group_result, 'rebalance_freq'):
                        freq = group_result.rebalance_freq
                        freq_map = {'D': 252, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}
                        periods_per_year = freq_map.get(freq, 252)
                        # 每次换手2倍（买入+卖出），但实际会有重叠
                        metrics['annual_turnover'] = periods_per_year * 1.5
                    else:
                        # 默认月度调仓
                        metrics['annual_turnover'] = 12 * 1.5
                
                # 股票覆盖度
                if test_result.processed_factor is not None:
                    factor_data = test_result.processed_factor
                    if isinstance(factor_data, pd.Series):
                        # 计算平均每期有效因子值的股票数量
                        if hasattr(factor_data.index, 'levels'):
                            # MultiIndex: (date, stock)
                            dates = factor_data.index.get_level_values(0).unique()
                            valid_counts = []
                            total_counts = []
                            for date in dates[:100]:  # 采样100个时间点
                                date_data = factor_data.xs(date, level=0)
                                valid_counts.append((~date_data.isna()).sum())
                                total_counts.append(len(date_data))
                            
                            if total_counts and np.mean(total_counts) > 0:
                                metrics['stock_coverage'] = np.mean(valid_counts) / np.mean(total_counts)
                                metrics['avg_valid_stocks'] = np.mean(valid_counts)
                
                # 从收益计算交易成本影响
                if group_result.long_short_return is not None and len(group_result.long_short_return) > 0:
                    gross_return = group_result.long_short_return.mean() * 252
                    
                    # 估算交易成本
                    annual_turnover = metrics.get('annual_turnover', 6.0)  # 默认年换手6倍
                    total_cost_rate = self.transaction_cost['commission'] * 2 + \
                                      self.transaction_cost['slippage'] * 2 + \
                                      self.transaction_cost['stamp_tax']
                    annual_cost = annual_turnover * total_cost_rate
                    
                    # 交易成本占收益比例
                    if abs(gross_return) > 1e-6:
                        metrics['cost_to_return_ratio'] = annual_cost / abs(gross_return)
                    else:
                        metrics['cost_to_return_ratio'] = float('inf')
                    
                    # 扣除成本后的净收益
                    metrics['net_return'] = gross_return - annual_cost
                    metrics['estimated_trading_cost'] = annual_cost
            
            # 从性能指标提取
            if test_result.performance_metrics:
                perf = test_result.performance_metrics
                
                # 如果有直接的换手率指标
                if 'turnover' in perf:
                    metrics['annual_turnover'] = perf['turnover'] * 252
                
                # 如果有直接的容量指标
                if 'capacity' in perf:
                    metrics['capacity_limit'] = perf['capacity']
        
        # 从市场数据提取流动性（如果有）
        market_data = data.get('market_data')
        if market_data:
            liquidity_data = market_data.get('liquidity')
            if liquidity_data is not None:
                if isinstance(liquidity_data, (pd.Series, pd.DataFrame)):
                    # 平均日成交额（假设单位是元，转换为亿元）
                    metrics['avg_liquidity'] = liquidity_data.mean() / 1e8
                elif isinstance(liquidity_data, (int, float)):
                    metrics['avg_liquidity'] = liquidity_data / 1e8
        
        # 默认流动性估算（基于覆盖度）
        if 'avg_liquidity' not in metrics:
            coverage = metrics.get('stock_coverage', 0.5)
            # 假设覆盖度高的因子选择的是流动性好的股票
            metrics['avg_liquidity'] = coverage * 3.0  # 粗略估算
        
        # 容量估算（基于流动性和换手率）
        if 'capacity_limit' not in metrics:
            avg_liquidity = metrics.get('avg_liquidity', 1.0)
            annual_turnover = metrics.get('annual_turnover', 6.0)
            avg_valid_stocks = metrics.get('avg_valid_stocks', 100)
            
            # 假设单笔交易不超过日成交额的10%
            daily_capacity = avg_liquidity * 1e8 * 0.1 * avg_valid_stocks
            # 考虑换手率的影响
            if annual_turnover > 0:
                metrics['capacity_limit'] = daily_capacity / (annual_turnover / 252)
            else:
                metrics['capacity_limit'] = daily_capacity
        
        return metrics
    
    def _score_turnover(self, turnover: float) -> float:
        """
        计算换手率得分
        
        Parameters
        ----------
        turnover : float
            年化换手率
            
        Returns
        -------
        float
            换手率得分(0-100)
        """
        if turnover == float('inf'):
            return 0
        
        # 换手率越低越好
        if turnover <= self.turnover_thresholds['excellent']:
            return 100
        elif turnover <= self.turnover_thresholds['good']:
            return 80 + (self.turnover_thresholds['good'] - turnover) / \
                   (self.turnover_thresholds['good'] - self.turnover_thresholds['excellent']) * 20
        elif turnover <= self.turnover_thresholds['fair']:
            return 60 + (self.turnover_thresholds['fair'] - turnover) / \
                   (self.turnover_thresholds['fair'] - self.turnover_thresholds['good']) * 20
        elif turnover <= self.turnover_thresholds['poor']:
            return 40 + (self.turnover_thresholds['poor'] - turnover) / \
                   (self.turnover_thresholds['poor'] - self.turnover_thresholds['fair']) * 20
        else:
            return max(0, 40 * self.turnover_thresholds['poor'] / turnover)
    
    def _score_cost_ratio(self, ratio: float) -> float:
        """
        计算交易成本比例得分
        
        Parameters
        ----------
        ratio : float
            交易成本占收益比例
            
        Returns
        -------
        float
            成本得分(0-100)
        """
        if ratio == float('inf'):
            return 0
        
        # 成本占比越低越好
        if ratio <= self.cost_ratio_thresholds['excellent']:
            return 100
        elif ratio <= self.cost_ratio_thresholds['good']:
            return 80 + (self.cost_ratio_thresholds['good'] - ratio) / \
                   (self.cost_ratio_thresholds['good'] - self.cost_ratio_thresholds['excellent']) * 20
        elif ratio <= self.cost_ratio_thresholds['fair']:
            return 60 + (self.cost_ratio_thresholds['fair'] - ratio) / \
                   (self.cost_ratio_thresholds['fair'] - self.cost_ratio_thresholds['good']) * 20
        elif ratio <= self.cost_ratio_thresholds['poor']:
            return 40 + (self.cost_ratio_thresholds['poor'] - ratio) / \
                   (self.cost_ratio_thresholds['poor'] - self.cost_ratio_thresholds['fair']) * 20
        else:
            return max(0, 40 * self.cost_ratio_thresholds['poor'] / ratio)
    
    def _score_coverage(self, coverage: float) -> float:
        """
        计算股票覆盖度得分
        
        Parameters
        ----------
        coverage : float
            股票覆盖比例(0-1)
            
        Returns
        -------
        float
            覆盖度得分(0-100)
        """
        # 覆盖度越高越好
        if coverage >= self.coverage_thresholds['excellent']:
            return 100
        elif coverage >= self.coverage_thresholds['good']:
            return 80 + (coverage - self.coverage_thresholds['good']) / \
                   (self.coverage_thresholds['excellent'] - self.coverage_thresholds['good']) * 20
        elif coverage >= self.coverage_thresholds['fair']:
            return 60 + (coverage - self.coverage_thresholds['fair']) / \
                   (self.coverage_thresholds['good'] - self.coverage_thresholds['fair']) * 20
        elif coverage >= self.coverage_thresholds['poor']:
            return 40 + (coverage - self.coverage_thresholds['poor']) / \
                   (self.coverage_thresholds['fair'] - self.coverage_thresholds['poor']) * 20
        else:
            return coverage / self.coverage_thresholds['poor'] * 40
    
    def _score_liquidity(self, liquidity: float) -> float:
        """
        计算流动性得分
        
        Parameters
        ----------
        liquidity : float
            平均日成交额（亿元）
            
        Returns
        -------
        float
            流动性得分(0-100)
        """
        # 流动性越高越好
        if liquidity >= self.liquidity_thresholds['excellent']:
            return 100
        elif liquidity >= self.liquidity_thresholds['good']:
            return 80 + (liquidity - self.liquidity_thresholds['good']) / \
                   (self.liquidity_thresholds['excellent'] - self.liquidity_thresholds['good']) * 20
        elif liquidity >= self.liquidity_thresholds['fair']:
            return 60 + (liquidity - self.liquidity_thresholds['fair']) / \
                   (self.liquidity_thresholds['good'] - self.liquidity_thresholds['fair']) * 20
        elif liquidity >= self.liquidity_thresholds['poor']:
            return 40 + (liquidity - self.liquidity_thresholds['poor']) / \
                   (self.liquidity_thresholds['fair'] - self.liquidity_thresholds['poor']) * 20
        else:
            return liquidity / self.liquidity_thresholds['poor'] * 40
    
    def _score_capacity(self, capacity: float) -> float:
        """
        计算策略容量得分
        
        Parameters
        ----------
        capacity : float
            策略容量（元）
            
        Returns
        -------
        float
            容量得分(0-100)
        """
        # 容量阈值（亿元）
        thresholds = {
            'excellent': 10.0,   # 10亿以上
            'good': 5.0,         # 5亿以上
            'fair': 1.0,         # 1亿以上
            'poor': 0.1          # 1000万以上
        }
        
        capacity_billion = capacity / 1e8  # 转换为亿元
        
        if capacity_billion >= thresholds['excellent']:
            return 100
        elif capacity_billion >= thresholds['good']:
            return 80 + (capacity_billion - thresholds['good']) / \
                   (thresholds['excellent'] - thresholds['good']) * 20
        elif capacity_billion >= thresholds['fair']:
            return 60 + (capacity_billion - thresholds['fair']) / \
                   (thresholds['good'] - thresholds['fair']) * 20
        elif capacity_billion >= thresholds['poor']:
            return 40 + (capacity_billion - thresholds['poor']) / \
                   (thresholds['fair'] - thresholds['poor']) * 20
        else:
            return capacity_billion / thresholds['poor'] * 40
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """
        获取默认评分标准
        
        Returns
        -------
        Dict[str, Any]
            默认评分标准
        """
        return {
            'turnover_thresholds': self.turnover_thresholds,
            'cost_ratio_thresholds': self.cost_ratio_thresholds,
            'coverage_thresholds': self.coverage_thresholds,
            'liquidity_thresholds': self.liquidity_thresholds
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
            'turnover_score': 0.30,      # 换手率权重30%
            'cost_score': 0.25,          # 交易成本权重25%
            'liquidity_score': 0.20,     # 流动性权重20%
            'coverage_score': 0.15,      # 覆盖度权重15%
            'capacity_score': 0.10       # 容量权重10%
        }
    
    def _generate_description(self, score: float, metrics: Dict[str, float]) -> str:
        """
        生成可交易性评价描述
        
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
        
        turnover = metrics.get('annual_turnover', float('inf'))
        cost_ratio = metrics.get('cost_to_return_ratio', 1.0)
        coverage = metrics.get('stock_coverage', 0)
        liquidity = metrics.get('avg_liquidity', 0)
        
        description = f"可交易性维度得分{score:.1f}分(等级{grade})。"
        
        # 换手率评价
        if turnover < self.turnover_thresholds['excellent']:
            description += f"年换手率({turnover:.1f}倍)很低，交易频率合理。"
        elif turnover < self.turnover_thresholds['good']:
            description += f"年换手率({turnover:.1f}倍)适中。"
        elif turnover < self.turnover_thresholds['fair']:
            description += f"年换手率({turnover:.1f}倍)偏高。"
        else:
            description += f"年换手率({turnover:.1f}倍)过高，交易成本压力大。"
        
        # 成本评价
        if cost_ratio < self.cost_ratio_thresholds['excellent']:
            description += f"交易成本占比({cost_ratio:.1%})很低。"
        elif cost_ratio < self.cost_ratio_thresholds['good']:
            description += f"交易成本占比({cost_ratio:.1%})可接受。"
        else:
            description += f"交易成本占比({cost_ratio:.1%})偏高，侵蚀收益。"
        
        # 覆盖度评价
        if coverage > self.coverage_thresholds['good']:
            description += f"股票覆盖度({coverage:.1%})良好。"
        elif coverage < self.coverage_thresholds['poor']:
            description += f"股票覆盖度({coverage:.1%})较低。"
        
        # 流动性评价
        if liquidity > self.liquidity_thresholds['good']:
            description += f"平均流动性({liquidity:.1f}亿)充足。"
        
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
        
        # 检查是否有TestResult
        test_result = data.get('test_result')
        if not isinstance(test_result, TestResult):
            logger.warning("No TestResult found in data")
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
            description="无法计算可交易性得分，数据不足"
        )