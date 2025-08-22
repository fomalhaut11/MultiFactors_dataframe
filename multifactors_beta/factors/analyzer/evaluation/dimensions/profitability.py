"""
收益能力维度评估
评估因子的收益预测能力和盈利潜力
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

from .base_dimension import BaseDimension, DimensionScore
from ....tester.base import TestResult

logger = logging.getLogger(__name__)


class ProfitabilityDimension(BaseDimension):
    """
    收益能力维度
    
    评估指标：
    1. IC均值 - 因子与收益的相关性
    2. ICIR - IC的稳定性
    3. 夏普比率 - 风险调整后收益
    4. 多空组合收益 - 多空策略表现
    5. 最大组收益 - 最优组表现
    """
    
    def __init__(self, weight: float = 0.35, config: Optional[Dict] = None):
        """
        初始化收益能力维度
        
        Parameters
        ----------
        weight : float
            维度权重，默认35%
        config : Dict, optional
            配置参数
        """
        super().__init__(name="Profitability", weight=weight, config=config)
        
        # 收益能力评估阈值
        self.ic_thresholds = self.config.get('ic_thresholds', {
            'excellent': 0.05,
            'good': 0.03,
            'fair': 0.02,
            'poor': 0.01
        })
        
        self.icir_thresholds = self.config.get('icir_thresholds', {
            'excellent': 0.7,
            'good': 0.5,
            'fair': 0.3,
            'poor': 0.1
        })
        
        self.sharpe_thresholds = self.config.get('sharpe_thresholds', {
            'excellent': 1.5,
            'good': 1.0,
            'fair': 0.5,
            'poor': 0
        })
    
    def calculate_score(self, data: Dict[str, Any]) -> DimensionScore:
        """
        计算收益能力维度得分
        
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
        
        # IC均值得分
        ic_mean = metrics.get('ic_mean', 0)
        scores['ic_score'] = self._score_ic(ic_mean)
        
        # ICIR得分
        icir = metrics.get('icir', 0)
        scores['icir_score'] = self._score_icir(icir)
        
        # 夏普比率得分
        sharpe = metrics.get('sharpe_ratio', 0)
        scores['sharpe_score'] = self._score_sharpe(sharpe)
        
        # 多空收益得分
        long_short_return = metrics.get('long_short_return', 0)
        scores['long_short_score'] = self._score_long_short(long_short_return)
        
        # 最大组收益得分
        top_group_return = metrics.get('top_group_return', 0)
        scores['top_group_score'] = self._score_top_group(top_group_return)
        
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
        从数据中提取收益能力相关指标
        
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
            # 从性能指标中提取
            if test_result.performance_metrics:
                perf = test_result.performance_metrics
                metrics['ic_mean'] = perf.get('ic_mean', 0)
                metrics['icir'] = perf.get('icir', 0)
                metrics['rank_ic_mean'] = perf.get('rank_ic_mean', 0)
                metrics['rank_icir'] = perf.get('rank_icir', 0)
                metrics['sharpe_ratio'] = perf.get('long_short_sharpe', 0)
                metrics['annual_return'] = perf.get('long_short_annual_return', 0)
            
            # 从IC结果中提取
            if test_result.ic_result:
                ic_result = test_result.ic_result
                metrics['ic_mean'] = ic_result.ic_mean
                metrics['ic_std'] = ic_result.ic_std
                metrics['icir'] = ic_result.icir
                metrics['rank_ic_mean'] = ic_result.rank_ic_mean
                metrics['rank_icir'] = ic_result.rank_icir
                
                # IC正值比例
                if ic_result.ic_series is not None and len(ic_result.ic_series) > 0:
                    metrics['ic_positive_ratio'] = (ic_result.ic_series > 0).mean()
            
            # 从分组结果中提取
            if test_result.group_result:
                group_result = test_result.group_result
                # 多空收益
                if group_result.long_short_return is not None and len(group_result.long_short_return) > 0:
                    metrics['long_short_return'] = group_result.long_short_return.mean() * 252  # 年化
                    metrics['long_short_std'] = group_result.long_short_return.std() * np.sqrt(252)
                
                # 最大组收益
                if group_result.group_returns is not None and not group_result.group_returns.empty:
                    # 获取最高组（通常是最后一组）的平均收益
                    top_group_idx = group_result.group_returns.columns[-1]
                    metrics['top_group_return'] = group_result.group_returns[top_group_idx].mean() * 252
                
                # 单调性得分
                metrics['monotonicity'] = group_result.monotonicity_score
            
            # 从回归结果中提取
            if test_result.regression_result:
                reg_result = test_result.regression_result
                if reg_result.factor_return is not None and len(reg_result.factor_return) > 0:
                    metrics['factor_return_mean'] = reg_result.factor_return.mean() * 252
                    metrics['factor_return_std'] = reg_result.factor_return.std() * np.sqrt(252)
                    # t值的均值
                    if reg_result.tvalues is not None and len(reg_result.tvalues) > 0:
                        metrics['avg_t_value'] = reg_result.tvalues.mean()
        
        return metrics
    
    def _score_ic(self, ic_mean: float) -> float:
        """
        计算IC均值得分
        
        Parameters
        ----------
        ic_mean : float
            IC均值
            
        Returns
        -------
        float
            IC得分(0-100)
        """
        if ic_mean >= self.ic_thresholds['excellent']:
            return 100
        elif ic_mean >= self.ic_thresholds['good']:
            return 80 + (ic_mean - self.ic_thresholds['good']) / (self.ic_thresholds['excellent'] - self.ic_thresholds['good']) * 20
        elif ic_mean >= self.ic_thresholds['fair']:
            return 60 + (ic_mean - self.ic_thresholds['fair']) / (self.ic_thresholds['good'] - self.ic_thresholds['fair']) * 20
        elif ic_mean >= self.ic_thresholds['poor']:
            return 40 + (ic_mean - self.ic_thresholds['poor']) / (self.ic_thresholds['fair'] - self.ic_thresholds['poor']) * 20
        elif ic_mean > 0:
            return ic_mean / self.ic_thresholds['poor'] * 40
        else:
            return 0
    
    def _score_icir(self, icir: float) -> float:
        """
        计算ICIR得分
        
        Parameters
        ----------
        icir : float
            ICIR值
            
        Returns
        -------
        float
            ICIR得分(0-100)
        """
        if icir >= self.icir_thresholds['excellent']:
            return 100
        elif icir >= self.icir_thresholds['good']:
            return 80 + (icir - self.icir_thresholds['good']) / (self.icir_thresholds['excellent'] - self.icir_thresholds['good']) * 20
        elif icir >= self.icir_thresholds['fair']:
            return 60 + (icir - self.icir_thresholds['fair']) / (self.icir_thresholds['good'] - self.icir_thresholds['fair']) * 20
        elif icir >= self.icir_thresholds['poor']:
            return 40 + (icir - self.icir_thresholds['poor']) / (self.icir_thresholds['fair'] - self.icir_thresholds['poor']) * 20
        elif icir > 0:
            return icir / self.icir_thresholds['poor'] * 40
        else:
            return 0
    
    def _score_sharpe(self, sharpe: float) -> float:
        """
        计算夏普比率得分
        
        Parameters
        ----------
        sharpe : float
            夏普比率
            
        Returns
        -------
        float
            夏普得分(0-100)
        """
        if sharpe >= self.sharpe_thresholds['excellent']:
            return 100
        elif sharpe >= self.sharpe_thresholds['good']:
            return 80 + (sharpe - self.sharpe_thresholds['good']) / (self.sharpe_thresholds['excellent'] - self.sharpe_thresholds['good']) * 20
        elif sharpe >= self.sharpe_thresholds['fair']:
            return 60 + (sharpe - self.sharpe_thresholds['fair']) / (self.sharpe_thresholds['good'] - self.sharpe_thresholds['fair']) * 20
        elif sharpe >= self.sharpe_thresholds['poor']:
            return 40 + (sharpe - self.sharpe_thresholds['poor']) / (self.sharpe_thresholds['fair'] - self.sharpe_thresholds['poor']) * 20
        elif sharpe > 0:
            return sharpe / self.sharpe_thresholds['fair'] * 40
        else:
            return max(0, 20 + sharpe * 10)  # 负夏普也给一点分
    
    def _score_long_short(self, annual_return: float) -> float:
        """
        计算多空组合收益得分
        
        Parameters
        ----------
        annual_return : float
            年化收益率
            
        Returns
        -------
        float
            多空收益得分(0-100)
        """
        # 年化收益率评分标准
        if annual_return >= 0.30:  # 30%以上
            return 100
        elif annual_return >= 0.20:  # 20-30%
            return 80 + (annual_return - 0.20) / 0.10 * 20
        elif annual_return >= 0.10:  # 10-20%
            return 60 + (annual_return - 0.10) / 0.10 * 20
        elif annual_return >= 0.05:  # 5-10%
            return 40 + (annual_return - 0.05) / 0.05 * 20
        elif annual_return > 0:  # 0-5%
            return annual_return / 0.05 * 40
        else:
            return max(0, 20 + annual_return * 100)  # 负收益也给一点分
    
    def _score_top_group(self, top_return: float) -> float:
        """
        计算最大组收益得分
        
        Parameters
        ----------
        top_return : float
            最大组年化收益率
            
        Returns
        -------
        float
            最大组得分(0-100)
        """
        # 最大组收益评分标准（比多空收益标准稍低）
        if top_return >= 0.25:  # 25%以上
            return 100
        elif top_return >= 0.15:  # 15-25%
            return 80 + (top_return - 0.15) / 0.10 * 20
        elif top_return >= 0.08:  # 8-15%
            return 60 + (top_return - 0.08) / 0.07 * 20
        elif top_return >= 0.03:  # 3-8%
            return 40 + (top_return - 0.03) / 0.05 * 20
        elif top_return > 0:  # 0-3%
            return top_return / 0.03 * 40
        else:
            return max(0, 20 + top_return * 100)
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """
        获取默认评分标准
        
        Returns
        -------
        Dict[str, Any]
            默认评分标准
        """
        return {
            'ic_thresholds': self.ic_thresholds,
            'icir_thresholds': self.icir_thresholds,
            'sharpe_thresholds': self.sharpe_thresholds
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
            'ic_score': 0.30,        # IC均值权重30%
            'icir_score': 0.30,      # ICIR权重30%
            'sharpe_score': 0.20,    # 夏普比率权重20%
            'long_short_score': 0.10, # 多空收益权重10%
            'top_group_score': 0.10  # 最大组收益权重10%
        }
    
    def _generate_description(self, score: float, metrics: Dict[str, float]) -> str:
        """
        生成收益能力评价描述
        
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
        
        ic_mean = metrics.get('ic_mean', 0)
        icir = metrics.get('icir', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        description = f"收益能力维度得分{score:.1f}分(等级{grade})。"
        
        # IC评价
        if ic_mean >= self.ic_thresholds['excellent']:
            description += f"IC均值({ic_mean:.4f})表现卓越，预测能力强。"
        elif ic_mean >= self.ic_thresholds['good']:
            description += f"IC均值({ic_mean:.4f})表现良好。"
        elif ic_mean >= self.ic_thresholds['fair']:
            description += f"IC均值({ic_mean:.4f})表现一般。"
        else:
            description += f"IC均值({ic_mean:.4f})偏低，预测能力有限。"
        
        # ICIR评价
        if icir >= self.icir_thresholds['excellent']:
            description += f"ICIR({icir:.2f})很高，信号稳定性优秀。"
        elif icir >= self.icir_thresholds['good']:
            description += f"ICIR({icir:.2f})较好，信号相对稳定。"
        else:
            description += f"ICIR({icir:.2f})偏低，稳定性有待提高。"
        
        # 夏普评价
        if sharpe >= self.sharpe_thresholds['excellent']:
            description += f"夏普比率({sharpe:.2f})优秀。"
        elif sharpe >= self.sharpe_thresholds['good']:
            description += f"夏普比率({sharpe:.2f})良好。"
        
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
        
        # 检查是否有必要的结果
        if test_result.ic_result is None and test_result.performance_metrics is None:
            logger.warning("No IC result or performance metrics in TestResult")
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
            description="无法计算收益能力得分，数据不足"
        )