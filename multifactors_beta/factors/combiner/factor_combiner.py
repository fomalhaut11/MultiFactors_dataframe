"""
因子组合器主类

整合各种组合方法和权重计算策略
"""

from typing import Dict, Optional, Any, Union, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .base.combiner_base import CombinerBase
from .weighting import (
    EqualWeightCalculator,
    ICWeightCalculator,
    IRWeightCalculator,
    RiskParityCalculator,
    OptimalWeightCalculator
)
from .methods import LinearCombiner

logger = logging.getLogger(__name__)


class FactorCombiner(CombinerBase):
    """
    因子组合器
    
    提供多种组合方法和权重计算策略
    """
    
    def __init__(self, 
                method: str = 'equal_weight',
                config: Optional[Dict[str, Any]] = None):
        """
        初始化因子组合器
        
        Parameters
        ----------
        method : str
            组合方法：'equal_weight', 'ic_weight', 'ir_weight', 'custom'
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        self.method = method
        self._init_components()
        
        logger.info(f"Initialized FactorCombiner with method: {method}")
    
    def _init_components(self):
        """初始化组件"""
        # 初始化权重计算器
        self.weight_calculators = {
            'equal_weight': EqualWeightCalculator(self.config),
            'ic_weight': ICWeightCalculator(self.config),
            'ir_weight': IRWeightCalculator(self.config),
            'risk_parity': RiskParityCalculator(self.config),
            'optimal_weight': OptimalWeightCalculator(self.config),
        }
        
        # 初始化组合方法
        self.linear_combiner = LinearCombiner(self.config)
    
    def combine(self,
               factors: Dict[str, pd.Series],
               evaluation_results: Optional[Dict] = None,
               custom_weights: Optional[Dict[str, float]] = None,
               **kwargs) -> pd.Series:
        """
        组合多个因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子字典，key为因子名，value为MultiIndex Series
        evaluation_results : Dict, optional
            评估结果，用于计算权重
        custom_weights : Dict[str, float], optional
            自定义权重，优先级最高
        **kwargs : dict
            其他参数，如returns用于计算IC
            
        Returns
        -------
        pd.Series
            组合后的因子
        """
        # 验证输入
        self.validate_factors(factors)
        
        # 对齐因子
        aligned_factors = self.align_factors(factors)
        
        # 计算权重
        if custom_weights:
            weights = self.normalize_weights_dict(custom_weights)
            logger.info("Using custom weights")
        else:
            weights = self.calculate_weights(
                aligned_factors,
                evaluation_results,
                **kwargs
            )
        
        # 组合因子
        composite = self.linear_combiner.combine(aligned_factors, weights)
        
        # 记录组合历史
        self.save_combination_history(
            factors=list(factors.keys()),
            weights=weights,
            result_stats={
                'n_factors': len(factors),
                'n_observations': len(composite),
                'method': self.method
            }
        )
        
        logger.info(
            f"Combined {len(factors)} factors into composite factor "
            f"with {len(composite)} observations"
        )
        
        return composite
    
    def calculate_weights(self,
                         factors: Dict[str, pd.Series],
                         evaluation_results: Optional[Dict] = None,
                         **kwargs) -> Dict[str, float]:
        """
        计算因子权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            因子权重
        """
        if self.method in self.weight_calculators:
            calculator = self.weight_calculators[self.method]
            weights = calculator.calculate(factors, evaluation_results, **kwargs)
        elif self.method == 'custom':
            # 自定义权重应该通过custom_weights参数传入
            raise ValueError(
                "Custom weights should be provided via custom_weights parameter"
            )
        else:
            raise ValueError(f"Unknown weight calculation method: {self.method}")
        
        logger.info(f"Calculated weights using {self.method}: {weights}")
        return weights
    
    def combine_with_evaluation(self,
                              test_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any]) -> pd.Series:
        """
        基于测试和评估结果组合因子
        
        Parameters
        ----------
        test_results : Dict[str, Any]
            因子测试结果，包含因子数据
        evaluation_results : Dict[str, Any]
            因子评估结果
            
        Returns
        -------
        pd.Series
            组合后的因子
        """
        # 提取因子数据
        factors = {}
        for factor_name, test_result in test_results.items():
            if hasattr(test_result, 'processed_factor'):
                factors[factor_name] = test_result.processed_factor
            elif isinstance(test_result, dict) and 'factor' in test_result:
                factors[factor_name] = test_result['factor']
        
        if not factors:
            raise ValueError("No factor data found in test results")
        
        # 组合
        return self.combine(factors, evaluation_results)
    
    def orthogonalize(self,
                     factors: Dict[str, pd.Series],
                     method: str = 'gram_schmidt',
                     base_factor: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        正交化因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子
        method : str
            正交化方法：'gram_schmidt', 'residual'
        base_factor : str, optional
            基准因子（保持不变）
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        # 验证和对齐
        self.validate_factors(factors)
        aligned_factors = self.align_factors(factors)
        
        if method == 'gram_schmidt':
            return self._gram_schmidt_orthogonalize(aligned_factors, base_factor)
        elif method == 'residual':
            return self._residual_orthogonalize(aligned_factors, base_factor)
        else:
            raise ValueError(f"Unknown orthogonalization method: {method}")
    
    def _gram_schmidt_orthogonalize(self,
                                   factors: Dict[str, pd.Series],
                                   base_factor: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Gram-Schmidt正交化
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
        base_factor : str, optional
            基准因子
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        factor_names = list(factors.keys())
        
        # 调整顺序，基准因子放在第一位
        if base_factor and base_factor in factor_names:
            factor_names.remove(base_factor)
            factor_names.insert(0, base_factor)
        
        # 转换为DataFrame便于处理
        factor_df = pd.DataFrame(factors)
        
        # 按日期分组进行正交化
        def orthogonalize_group(group):
            # 转换为numpy数组
            data = group[factor_names].values
            n_obs, n_factors = data.shape
            
            # Gram-Schmidt过程
            orthogonal = np.zeros_like(data)
            
            for i in range(n_factors):
                orthogonal[:, i] = data[:, i]
                
                # 减去之前向量的投影
                for j in range(i):
                    if np.linalg.norm(orthogonal[:, j]) > 0:
                        projection = np.dot(data[:, i], orthogonal[:, j]) / \
                                   np.dot(orthogonal[:, j], orthogonal[:, j])
                        orthogonal[:, i] -= projection * orthogonal[:, j]
                
                # 标准化
                norm = np.linalg.norm(orthogonal[:, i])
                if norm > 0:
                    orthogonal[:, i] /= norm
            
            # 转回DataFrame
            result = pd.DataFrame(
                orthogonal,
                index=group.index,
                columns=factor_names
            )
            return result
        
        # 应用正交化
        orthogonal_df = factor_df.groupby(level=0).apply(orthogonalize_group)
        
        # 转回字典格式
        orthogonal_factors = {
            name: orthogonal_df[name]
            for name in factor_names
        }
        
        logger.info(f"Orthogonalized {len(factors)} factors using Gram-Schmidt")
        return orthogonal_factors
    
    def _residual_orthogonalize(self,
                               factors: Dict[str, pd.Series],
                               base_factor: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        残差正交化
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            对齐后的因子
        base_factor : str, optional
            基准因子
            
        Returns
        -------
        Dict[str, pd.Series]
            正交化后的因子
        """
        if not base_factor or base_factor not in factors:
            # 如果没有指定基准因子，使用第一个
            base_factor = next(iter(factors.keys()))
        
        orthogonal_factors = {}
        base_series = factors[base_factor]
        
        # 基准因子保持不变
        orthogonal_factors[base_factor] = base_series
        
        # 其他因子取残差
        for name, factor in factors.items():
            if name == base_factor:
                continue
            
            # 按日期回归
            residuals = []
            dates = factor.index.get_level_values(0).unique()
            
            for date in dates:
                # 获取当日数据
                base_day = base_series.xs(date, level=0)
                factor_day = factor.xs(date, level=0)
                
                # 对齐
                common_stocks = base_day.index.intersection(factor_day.index)
                if len(common_stocks) < 3:  # 数据太少
                    continue
                
                X = base_day.loc[common_stocks].values.reshape(-1, 1)
                y = factor_day.loc[common_stocks].values
                
                # 线性回归
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(X, y)
                
                # 计算残差
                residual = y - reg.predict(X)
                
                # 保存残差
                for i, stock in enumerate(common_stocks):
                    residuals.append((date, stock, residual[i]))
            
            # 构建残差Series
            if residuals:
                residual_index = pd.MultiIndex.from_tuples(
                    [(d, s) for d, s, _ in residuals]
                )
                residual_values = [r for _, _, r in residuals]
                orthogonal_factors[name] = pd.Series(
                    residual_values,
                    index=residual_index,
                    name=name
                )
        
        logger.info(f"Orthogonalized {len(factors)} factors using residual method")
        return orthogonal_factors
    
    def rolling_combine(self,
                       factors: Dict[str, pd.Series],
                       window: int = 60,
                       min_periods: int = 30,
                       evaluation_func: Optional[Any] = None,
                       **kwargs) -> pd.Series:
        """
        滚动窗口组合
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        window : int
            窗口大小
        min_periods : int
            最小期数
        evaluation_func : callable, optional
            评估函数，用于计算动态权重
        **kwargs : dict
            其他参数
            
        Returns
        -------
        pd.Series
            滚动组合的因子
        """
        # 验证和对齐
        self.validate_factors(factors)
        aligned_factors = self.align_factors(factors)
        
        # 获取所有日期
        all_dates = aligned_factors[next(iter(aligned_factors))].index.get_level_values(0).unique()
        all_dates = sorted(all_dates)
        
        # 滚动组合
        results = []
        
        for i in range(len(all_dates)):
            if i < min_periods - 1:
                continue
            
            # 获取窗口数据
            start_idx = max(0, i - window + 1)
            window_dates = all_dates[start_idx:i+1]
            
            # 提取窗口内的因子数据
            window_factors = {}
            for name, factor in aligned_factors.items():
                window_factor = factor[
                    factor.index.get_level_values(0).isin(window_dates)
                ]
                window_factors[name] = window_factor
            
            # 计算权重（如果提供了评估函数）
            if evaluation_func:
                weights = evaluation_func(window_factors, **kwargs)
            else:
                # 使用默认方法计算权重
                weights = self.calculate_weights(window_factors, **kwargs)
            
            # 组合当期因子
            current_date = all_dates[i]
            current_factors = {}
            for name, factor in aligned_factors.items():
                current_factors[name] = factor.xs(current_date, level=0)
            
            # 计算组合值
            combined_values = sum(
                current_factors[name] * weights.get(name, 0)
                for name in current_factors.keys()
            )
            
            # 添加日期索引
            for stock, value in combined_values.items():
                results.append((current_date, stock, value))
        
        # 构建结果Series
        if results:
            result_index = pd.MultiIndex.from_tuples(
                [(d, s) for d, s, _ in results]
            )
            result_values = [v for _, _, v in results]
            composite = pd.Series(
                result_values,
                index=result_index,
                name='rolling_composite'
            )
        else:
            # 返回空Series
            first_factor = next(iter(aligned_factors.values()))
            composite = pd.Series(
                [],
                index=pd.MultiIndex.from_tuples([]),
                name='rolling_composite'
            )
        
        logger.info(
            f"Completed rolling combination with window={window}, "
            f"resulting in {len(composite)} observations"
        )
        
        return composite