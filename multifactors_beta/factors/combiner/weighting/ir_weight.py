"""
IR加权计算器
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

from .base_weight import BaseWeightCalculator

logger = logging.getLogger(__name__)


class IRWeightCalculator(BaseWeightCalculator):
    """
    IR（信息比率）加权计算器
    
    根据因子的IR（IC/IC_std）计算权重，考虑稳定性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化IR加权计算器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        super().__init__(config)
        self.min_ir = self.config.get('min_ir', 0.0)  # 最小IR阈值
        self.use_abs_ir = self.config.get('use_abs_ir', True)  # 是否使用绝对值
        self.ir_lookback = self.config.get('ir_lookback', 12)  # IR回看期数
    
    def calculate(self,
                 factors: Dict[str, pd.Series],
                 evaluation_results: Optional[Dict] = None,
                 **kwargs) -> Dict[str, float]:
        """
        根据IR计算权重
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            因子数据
        evaluation_results : Dict, optional
            评估结果，包含IR信息
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            IR加权的权重字典
        """
        self.validate_inputs(factors, evaluation_results)
        
        # 获取IR值
        ir_values = self._extract_ir_values(factors, evaluation_results, **kwargs)
        
        if not ir_values:
            logger.warning("No IR values available, using equal weights")
            n_factors = len(factors)
            return {name: 1.0/n_factors for name in factors.keys()}
        
        # 计算权重
        weights = self._calculate_ir_weights(ir_values)
        
        # 应用约束
        weights = self.apply_constraints(weights)
        
        logger.info(f"Calculated IR weights for {len(weights)} factors")
        return weights
    
    def _extract_ir_values(self,
                          factors: Dict[str, pd.Series],
                          evaluation_results: Optional[Dict] = None,
                          **kwargs) -> Dict[str, float]:
        """
        提取IR值
        
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
            因子IR值字典
        """
        ir_values = {}
        
        # 从评估结果中获取IR
        if evaluation_results:
            for factor_name in factors.keys():
                if factor_name in evaluation_results:
                    eval_result = evaluation_results[factor_name]
                    
                    # 尝试从不同位置获取IR
                    ir = None
                    if hasattr(eval_result, 'metrics'):
                        # 优先使用ICIR
                        ir = eval_result.metrics.get('icir')
                        if ir is None:
                            # 如果没有ICIR，尝试计算
                            ic_mean = eval_result.metrics.get('ic_mean')
                            ic_std = eval_result.metrics.get('ic_std')
                            if ic_mean is not None and ic_std is not None and ic_std > 0:
                                ir = ic_mean / ic_std
                    elif hasattr(eval_result, 'ic_result'):
                        if eval_result.ic_result:
                            ir = eval_result.ic_result.icir
                    elif isinstance(eval_result, dict):
                        ir = eval_result.get('icir')
                        if ir is None:
                            ic_mean = eval_result.get('ic_mean')
                            ic_std = eval_result.get('ic_std')
                            if ic_mean is not None and ic_std is not None and ic_std > 0:
                                ir = ic_mean / ic_std
                    
                    if ir is not None:
                        ir_values[factor_name] = ir
        
        # 如果提供了收益数据，计算IR
        if 'returns' in kwargs and len(ir_values) < len(factors):
            returns = kwargs['returns']
            for factor_name, factor in factors.items():
                if factor_name not in ir_values:
                    ir = self._calculate_ir(factor, returns)
                    if ir is not None:
                        ir_values[factor_name] = ir
        
        return ir_values
    
    def _calculate_ir(self, factor: pd.Series, returns: pd.Series) -> Optional[float]:
        """
        计算因子IR
        
        Parameters
        ----------
        factor : pd.Series
            因子值
        returns : pd.Series
            收益率
            
        Returns
        -------
        float or None
            IR值
        """
        try:
            # 对齐数据
            aligned = pd.DataFrame({'factor': factor, 'return': returns}).dropna()
            if len(aligned) < 10:  # 数据太少
                return None
            
            # 按日期计算IC
            ic_series = aligned.groupby(level=0).apply(
                lambda x: x['factor'].corr(x['return'], method='spearman')
            )
            
            # 只使用最近的数据
            if self.ir_lookback > 0:
                ic_series = ic_series.tail(self.ir_lookback)
            
            # 计算IR
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            
            if ic_std > 0:
                ir = ic_mean / ic_std
            else:
                ir = 0
            
            return ir
            
        except Exception as e:
            logger.warning(f"Failed to calculate IR: {e}")
            return None
    
    def _calculate_ir_weights(self, ir_values: Dict[str, float]) -> Dict[str, float]:
        """
        根据IR值计算权重
        
        Parameters
        ----------
        ir_values : Dict[str, float]
            IR值字典
            
        Returns
        -------
        Dict[str, float]
            权重字典
        """
        weights = {}
        
        # 处理IR值
        processed_ir = {}
        for name, ir in ir_values.items():
            # 使用绝对值或原值
            if self.use_abs_ir:
                ir_value = abs(ir)
            else:
                ir_value = ir
            
            # 应用最小IR阈值
            if abs(ir) < self.min_ir:
                ir_value = 0
            
            processed_ir[name] = max(0, ir_value)  # 确保非负
        
        # 计算权重（IR的平方，强调稳定性）
        total_ir_squared = sum(ir**2 for ir in processed_ir.values())
        
        if total_ir_squared > 0:
            weights = {name: ir**2/total_ir_squared for name, ir in processed_ir.items()}
        else:
            # 如果所有IR都是0，使用等权
            n = len(processed_ir)
            weights = {name: 1.0/n for name in processed_ir.keys()}
        
        return weights
    
    def _requires_evaluation(self) -> bool:
        """
        是否需要评估结果
        
        Returns
        -------
        bool
            True，IR加权需要评估结果
        """
        return True