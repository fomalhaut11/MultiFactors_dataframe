#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合因子管理器

统一管理需要多种数据源的混合因子计算
支持灵活的因子注册和动态路由
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any, Type
import logging

from ...base.factor_base import FactorBase
from .valuation_factors import ValuationFactorCalculator

logger = logging.getLogger(__name__)


class MixedFactorManager:
    """
    混合因子管理器
    
    负责统一管理各种混合因子计算器，提供统一的调用接口
    """
    
    def __init__(self):
        self.name = 'MixedFactorManager'
        self.category = 'mixed'
        
        # 注册各种混合因子计算器
        self.calculators = {}
        self.factor_registry = {}
        
        # 初始化已实现的计算器
        self._register_calculators()
    
    def _register_calculators(self):
        """注册混合因子计算器"""
        
        # 估值因子计算器
        try:
            valuation_calc = ValuationFactorCalculator()
            self.calculators['valuation'] = valuation_calc
            
            # 注册估值因子
            for factor_name in valuation_calc.get_available_factors():
                self.factor_registry[factor_name] = {
                    'calculator_type': 'valuation',
                    'calculator': valuation_calc,
                    'data_requirements': ['financial_data', 'market_cap'],
                    'category': 'valuation'
                }
            
            logger.info(f"✅ 注册估值因子计算器，包含因子: {valuation_calc.get_available_factors()}")
            
        except Exception as e:
            logger.error(f"❌ 估值因子计算器注册失败: {e}")
        
        # 注册自定义混合因子
        try:
            from .custom_mixed_factors import create_cashflow_efficiency_ratio
            
            # 创建自定义因子实例
            cashflow_factor = create_cashflow_efficiency_ratio()
            
            # 注册到管理器
            self.factor_registry['CashflowEfficiencyRatio'] = {
                'calculator_type': 'custom',
                'calculator': cashflow_factor,
                'data_requirements': ['financial_data', 'bp_data'],
                'category': 'mixed_custom'
            }
            
            logger.info("✅ 注册自定义混合因子: CashflowEfficiencyRatio")
            
        except Exception as e:
            logger.error(f"❌ 自定义混合因子注册失败: {e}")
        
        # TODO: 注册其他混合因子计算器
        # self._register_size_factors()
        # self._register_liquidity_factors()
        # self._register_quality_factors()
    
    def get_available_factors(self) -> List[str]:
        """获取所有可用的混合因子列表"""
        return list(self.factor_registry.keys())
    
    def get_factor_info(self, factor_name: str = None) -> Dict[str, Any]:
        """获取因子信息"""
        if factor_name:
            if factor_name in self.factor_registry:
                return self.factor_registry[factor_name]
            else:
                return {'error': f'未找到因子: {factor_name}'}
        else:
            return self.factor_registry
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """根据类别获取因子列表"""
        return [name for name, info in self.factor_registry.items() 
                if info.get('category') == category]
    
    def calculate_factor(self, 
                        factor_name: str,
                        data: Dict[str, Any],
                        **kwargs) -> pd.Series:
        """
        计算单个混合因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        data : dict
            数据字典，包含各种数据源
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        pd.Series
            因子计算结果
        """
        if factor_name not in self.factor_registry:
            raise ValueError(f"未知的混合因子: {factor_name}")
        
        factor_info = self.factor_registry[factor_name]
        calculator = factor_info['calculator']
        data_requirements = factor_info['data_requirements']
        
        # 检查数据可用性
        missing_data = [req for req in data_requirements if req not in data or data[req] is None]
        if missing_data:
            raise ValueError(f"因子 {factor_name} 缺少必需数据: {missing_data}")
        
        # 根据因子类型调用相应计算方法
        try:
            if factor_info['calculator_type'] == 'valuation':
                return calculator.calculate(
                    data['financial_data'], 
                    data['market_cap'], 
                    factor_names=factor_name, 
                    **kwargs
                )
            elif factor_info['calculator_type'] == 'custom':
                # 自定义因子使用calculate方法
                return calculator.calculate(data, **kwargs)
            else:
                # 其他类型的混合因子计算逻辑
                method_name = f'calculate_{factor_name}'
                if hasattr(calculator, method_name):
                    method = getattr(calculator, method_name)
                    return method(data, **kwargs)
                else:
                    raise AttributeError(f"计算器 {calculator.__class__.__name__} 没有方法 {method_name}")
                    
        except Exception as e:
            logger.error(f"计算混合因子 {factor_name} 失败: {e}")
            return pd.Series(dtype=float, name=factor_name)
    
    def calculate_multiple_factors(self,
                                 factor_names: List[str],
                                 data: Dict[str, Any],
                                 **kwargs) -> pd.DataFrame:
        """
        批量计算多个混合因子
        
        Parameters:
        -----------
        factor_names : List[str]
            因子名称列表
        data : dict
            数据字典
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        pd.DataFrame
            包含所有因子的数据框
        """
        results = {}
        
        # 按计算器类型分组，提高计算效率
        calculator_groups = {}
        for factor_name in factor_names:
            if factor_name not in self.factor_registry:
                logger.warning(f"跳过未知因子: {factor_name}")
                continue
            
            calc_type = self.factor_registry[factor_name]['calculator_type']
            if calc_type not in calculator_groups:
                calculator_groups[calc_type] = []
            calculator_groups[calc_type].append(factor_name)
        
        # 按组批量计算
        for calc_type, factors in calculator_groups.items():
            try:
                logger.info(f"批量计算 {calc_type} 因子: {factors}")
                
                if calc_type == 'valuation':
                    calculator = self.calculators['valuation']
                    batch_result = calculator.calculate_multiple_factors(
                        factors, data['financial_data'], data['market_cap'], **kwargs
                    )
                    results.update(batch_result.to_dict('series'))
                else:
                    # 其他类型的批量计算逻辑
                    for factor_name in factors:
                        result = self.calculate_factor(factor_name, data, **kwargs)
                        results[factor_name] = result
                        
            except Exception as e:
                logger.error(f"批量计算 {calc_type} 因子失败: {e}")
                # 逐个计算失败的因子
                for factor_name in factors:
                    try:
                        result = self.calculate_factor(factor_name, data, **kwargs)
                        results[factor_name] = result
                    except Exception as ee:
                        logger.error(f"单独计算因子 {factor_name} 也失败: {ee}")
        
        return pd.DataFrame(results)
    
    def get_data_requirements(self, factor_names: Union[str, List[str]]) -> List[str]:
        """
        获取指定因子的数据需求
        
        Parameters:
        -----------
        factor_names : str or List[str]
            因子名称或因子名称列表
            
        Returns:
        --------
        List[str]
            所需的数据类型列表
        """
        if isinstance(factor_names, str):
            factor_names = [factor_names]
        
        all_requirements = set()
        for factor_name in factor_names:
            if factor_name in self.factor_registry:
                requirements = self.factor_registry[factor_name]['data_requirements']
                all_requirements.update(requirements)
        
        return list(all_requirements)


# =====================================================
# 便捷函数
# =====================================================

# 全局管理器实例
_mixed_factor_manager = None

def get_mixed_factor_manager() -> MixedFactorManager:
    """获取混合因子管理器单例"""
    global _mixed_factor_manager
    if _mixed_factor_manager is None:
        _mixed_factor_manager = MixedFactorManager()
    return _mixed_factor_manager


def calculate_mixed_factor(factor_name: str, data: Dict[str, Any], **kwargs) -> pd.Series:
    """计算单个混合因子的便捷函数"""
    manager = get_mixed_factor_manager()
    return manager.calculate_factor(factor_name, data, **kwargs)


def calculate_multiple_mixed_factors(factor_names: List[str], 
                                    data: Dict[str, Any], 
                                    **kwargs) -> pd.DataFrame:
    """计算多个混合因子的便捷函数"""
    manager = get_mixed_factor_manager()
    return manager.calculate_multiple_factors(factor_names, data, **kwargs)


def get_all_mixed_factors() -> List[str]:
    """获取所有可用混合因子列表"""
    manager = get_mixed_factor_manager()
    return manager.get_available_factors()


def get_mixed_factor_data_requirements(factor_names: Union[str, List[str]]) -> List[str]:
    """获取混合因子数据需求"""
    manager = get_mixed_factor_manager()
    return manager.get_data_requirements(factor_names)