#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子生成器基类

提供统一的因子生成接口和基础功能
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Any
import logging
from pathlib import Path
import pickle

from ..base import DataProcessingMixin
from core.config_manager import get_path
from ..utils.multiindex_helper import (
    validate_factor_format,
    ensure_multiindex_format,
    MultiIndexHelper
)

logger = logging.getLogger(__name__)


class FactorGenerator(DataProcessingMixin, ABC):
    """
    因子生成器基类
    
    所有因子生成器的抽象基类，定义了统一的接口
    """
    
    def __init__(self, 
                 factor_type: str,
                 **kwargs):
        """
        初始化因子生成器
        
        Parameters
        ----------
        factor_type : str
            因子类型（financial, technical, risk等）
        """
        self.name = kwargs.pop('name', f'{factor_type}_generator')
        self.category = kwargs.pop('category', factor_type)
        self.factor_type = factor_type
        self.generated_factors = {}
        # 调用DataProcessingMixin的初始化（如果有的话）
        super().__init__()
        
    @abstractmethod
    def generate(self, 
                factor_name: str,
                data: pd.Series,
                **kwargs) -> pd.Series:
        """
        生成指定的因子
        
        Parameters
        ----------
        factor_name : str
            因子名称
        data : pd.Series
            输入数据，MultiIndex格式[TradingDates, StockCodes]
        **kwargs
            其他参数
            
        Returns
        -------
        pd.Series
            生成的因子数据，MultiIndex格式
        """
        pass
        
    def batch_generate(self,
                      factor_names: List[str],
                      data: pd.Series,
                      **kwargs) -> Dict[str, pd.Series]:
        """
        批量生成多个因子
        
        Parameters
        ----------
        factor_names : List[str]
            因子名称列表
        data : pd.Series
            输入数据，MultiIndex格式
        **kwargs
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            因子名称到因子数据的映射
        """
        # 确保数据格式
        data = ensure_multiindex_format(data)
        results = {}
        for factor_name in factor_names:
            try:
                logger.info(f"生成因子: {factor_name}")
                factor_data = self.generate(factor_name, data, **kwargs)
                results[factor_name] = factor_data
                self.generated_factors[factor_name] = factor_data
            except Exception as e:
                logger.error(f"生成因子 {factor_name} 失败: {e}")
                results[factor_name] = None
                
        return results
        
    def save_factor(self,
                   factor_name: str,
                   factor_data: pd.Series = None,
                   format: str = 'pkl') -> Path:
        """
        保存因子到本地
        
        Parameters
        ----------
        factor_name : str
            因子名称
        factor_data : pd.Series, optional
            因子数据（MultiIndex格式），如果不提供则从generated_factors中获取
        format : str
            保存格式（pkl或csv）
            
        Returns
        -------
        Path
            保存的文件路径
        """
        if factor_data is None:
            if factor_name not in self.generated_factors:
                raise ValueError(f"因子 {factor_name} 未生成")
            factor_data = self.generated_factors[factor_name]
            
        # 获取保存路径
        factor_dir = Path(get_path('factors'))
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        if format == 'pkl':
            filepath = factor_dir / f"{factor_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(factor_data, f)
        elif format == 'csv':
            filepath = factor_dir / f"{factor_name}.csv"
            factor_data.to_csv(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        logger.info(f"因子 {factor_name} 已保存到: {filepath}")
        return filepath
        
    def load_factor(self,
                   factor_name: str,
                   format: str = 'pkl') -> pd.Series:
        """
        从本地加载因子
        
        Parameters
        ----------
        factor_name : str
            因子名称
        format : str
            文件格式（pkl或csv）
            
        Returns
        -------
        pd.Series
            因子数据（MultiIndex格式）
        """
        factor_dir = Path(get_path('factors'))
        
        if format == 'pkl':
            filepath = factor_dir / f"{factor_name}.pkl"
            with open(filepath, 'rb') as f:
                factor_data = pickle.load(f)
        elif format == 'csv':
            filepath = factor_dir / f"{factor_name}.csv"
            factor_data = pd.read_csv(filepath, index_col=[0, 1])
            # 如果读取的是DataFrame，转换为Series
            if isinstance(factor_data, pd.DataFrame):
                if factor_data.shape[1] == 1:
                    factor_data = factor_data.iloc[:, 0]
                else:
                    # 如果有多列，尝试stack
                    factor_data = factor_data.stack()
            # 确保索引名称正确
            if isinstance(factor_data.index, pd.MultiIndex):
                factor_data.index.names = ['TradingDates', 'StockCodes']
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        logger.info(f"从 {filepath} 加载因子 {factor_name}")
        return factor_data
        
    @abstractmethod
    def get_available_factors(self) -> List[str]:
        """
        获取可用的因子列表
        
        Returns
        -------
        List[str]
            可用的因子名称列表
        """
        pass
        
    def get_factor_info(self, factor_name: str) -> Dict[str, Any]:
        """
        获取因子信息
        
        Parameters
        ----------
        factor_name : str
            因子名称
            
        Returns
        -------
        Dict[str, Any]
            因子信息
        """
        return {
            'name': factor_name,
            'type': self.factor_type,
            'generator': self.__class__.__name__,
            'generated': factor_name in self.generated_factors
        }


class FinancialFactorGenerator(FactorGenerator):
    """
    财务因子生成器
    
    生成基于财务报表数据的因子
    """
    
    def __init__(self, **kwargs):
        super().__init__(factor_type='financial', **kwargs)
        # 导入财务因子计算器
        from .financial.pure_financial_factors import PureFinancialFactorCalculator
        from .financial.earnings_surprise_factors import SUE
        self.calculator = PureFinancialFactorCalculator()
        self.sue_calculator = SUE()
        
    def generate(self,
                factor_name: str,
                data: pd.Series,
                **kwargs) -> pd.Series:
        """
        生成财务因子
        """
        # 确保数据格式
        data = ensure_multiindex_format(data)
        # 检查是否是SUE相关因子
        if factor_name in ['SUE', 'EarningsRevision', 'EarningsMomentum']:
            if factor_name == 'SUE':
                return self.sue_calculator.calculate(data, **kwargs)
            else:
                raise NotImplementedError(f"因子 {factor_name} 尚未实现")
                
        # 使用纯财务因子计算器
        method_name = f"calculate_{factor_name}"
        if hasattr(self.calculator, method_name):
            method = getattr(self.calculator, method_name)
            return method(data, **kwargs)
        else:
            raise ValueError(f"不支持的财务因子: {factor_name}")
            
    def get_available_factors(self) -> List[str]:
        """
        获取可用的财务因子列表
        """
        factors = self.calculator.get_available_factors()
        factors.extend(['SUE', 'EarningsRevision', 'EarningsMomentum'])
        return factors


class TechnicalFactorGenerator(FactorGenerator):
    """
    技术因子生成器
    
    生成基于价格和成交量数据的技术因子
    """
    
    def __init__(self, **kwargs):
        super().__init__(factor_type='technical', **kwargs)
        
    def generate(self,
                factor_name: str,
                data: pd.Series,
                **kwargs) -> pd.Series:
        """
        生成技术因子
        """
        # 确保数据格式
        data = ensure_multiindex_format(data)
        
        # TODO: 实现具体的技术因子生成逻辑
        raise NotImplementedError(f"技术因子 {factor_name} 尚未实现")
        
    def get_available_factors(self) -> List[str]:
        """
        获取可用的技术因子列表
        """
        return [
            'MA', 'EMA', 'RSI', 'MACD',
            'Volatility', 'Momentum', 'Reversal'
        ]


class RiskFactorGenerator(FactorGenerator):
    """
    风险因子生成器
    
    生成风险相关的因子
    """
    
    def __init__(self, **kwargs):
        super().__init__(factor_type='risk', **kwargs)
        
    def generate(self,
                factor_name: str,
                data: pd.Series,
                **kwargs) -> pd.Series:
        """
        生成风险因子
        """
        # 确保数据格式
        data = ensure_multiindex_format(data)
        
        # TODO: 实现具体的风险因子生成逻辑
        if factor_name == 'Beta':
            # 使用现有的Beta因子计算
            from .risk.beta_factors import BetaFactor
            beta_calculator = BetaFactor()
            return beta_calculator.calculate(data, **kwargs)
        else:
            raise NotImplementedError(f"风险因子 {factor_name} 尚未实现")
            
    def get_available_factors(self) -> List[str]:
        """
        获取可用的风险因子列表
        """
        return ['Beta', 'VaR', 'CVaR', 'ResidualVolatility']


# 工厂函数
def create_generator(factor_type: str, **kwargs) -> FactorGenerator:
    """
    创建因子生成器
    
    Parameters
    ----------
    factor_type : str
        因子类型（financial, technical, risk）
        
    Returns
    -------
    FactorGenerator
        对应的因子生成器实例
    """
    generators = {
        'financial': FinancialFactorGenerator,
        'technical': TechnicalFactorGenerator,
        'risk': RiskFactorGenerator
    }
    
    if factor_type not in generators:
        raise ValueError(f"不支持的因子类型: {factor_type}")
        
    return generators[factor_type](**kwargs)


# 导出接口
__all__ = [
    'FactorGenerator',
    'FinancialFactorGenerator',
    'TechnicalFactorGenerator',
    'RiskFactorGenerator',
    'create_generator'
]