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
from config import get_config
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
                      data: Union[pd.Series, Dict[str, Any]],
                      **kwargs) -> Dict[str, pd.Series]:
        """
        批量生成多个因子
        
        Parameters
        ----------
        factor_names : List[str]
            因子名称列表
        data : Union[pd.Series, Dict[str, Any]]
            输入数据，对于普通因子是MultiIndex格式的Series，对于混合因子是字典
        **kwargs
            其他参数
            
        Returns
        -------
        Dict[str, pd.Series]
            因子名称到因子数据的映射
        """
        # 数据验证
        if not isinstance(factor_names, (list, tuple)):
            raise ValueError(f"factor_names 必须是列表或元组，当前类型: {type(factor_names)}")
        
        if not factor_names:
            logger.warning("factor_names 为空，返回空结果")
            return {}
            
        # 对于非字典数据，确保MultiIndex格式
        if not isinstance(data, dict):
            data = ensure_multiindex_format(data)
            
        results = {}
        successful_count = 0
        
        for factor_name in factor_names:
            try:
                logger.info(f"生成因子: {factor_name}")
                factor_data = self.generate(factor_name, data, **kwargs)
                
                # 验证生成结果
                if factor_data is not None and not (isinstance(factor_data, pd.Series) and factor_data.empty):
                    results[factor_name] = factor_data
                    self.generated_factors[factor_name] = factor_data
                    successful_count += 1
                    logger.info(f"✅ 因子 {factor_name} 生成成功，数据点数: {len(factor_data) if factor_data is not None else 0}")
                else:
                    logger.warning(f"⚠️ 因子 {factor_name} 生成结果为空")
                    results[factor_name] = None
                    
            except Exception as e:
                logger.error(f"❌ 生成因子 {factor_name} 失败: {e}")
                results[factor_name] = None
                
        logger.info(f"批量生成完成: {successful_count}/{len(factor_names)} 个因子成功生成")
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
        # 输入验证
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError("factor_name 必须是非空字符串")
            
        if factor_data is None:
            if factor_name not in self.generated_factors:
                raise ValueError(f"因子 {factor_name} 未生成，请先生成因子或提供 factor_data 参数")
            factor_data = self.generated_factors[factor_name]
            
        # 验证因子数据
        if factor_data is None:
            raise ValueError(f"因子 {factor_name} 的数据为空")
            
        if not isinstance(factor_data, pd.Series):
            raise ValueError(f"因子数据必须是 pandas.Series，当前类型: {type(factor_data)}")
            
        if factor_data.empty:
            logger.warning(f"因子 {factor_name} 的数据为空，仍然保存")
            
        # 获取保存路径
        try:
            factor_dir = Path(get_config('main.paths.factors'))
        except:
            # 如果配置管理器不可用，使用默认路径
            factor_dir = Path.cwd() / 'factors'
            logger.warning(f"配置管理器不可用，使用默认路径: {factor_dir}")
            
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证格式
        supported_formats = ['pkl', 'csv']
        if format not in supported_formats:
            raise ValueError(f"不支持的格式: {format}，支持的格式: {supported_formats}")
        
        # 保存文件
        try:
            if format == 'pkl':
                filepath = factor_dir / f"{factor_name}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(factor_data, f)
            elif format == 'csv':
                filepath = factor_dir / f"{factor_name}.csv"
                factor_data.to_csv(filepath)
                
            logger.info(f"✅ 因子 {factor_name} 已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 保存因子 {factor_name} 失败: {e}")
            raise
        
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
        # 输入验证
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError("factor_name 必须是非空字符串")
            
        supported_formats = ['pkl', 'csv']
        if format not in supported_formats:
            raise ValueError(f"不支持的格式: {format}，支持的格式: {supported_formats}")
        
        # 获取文件路径
        try:
            factor_dir = Path(get_config('main.paths.factors'))
        except:
            # 如果配置管理器不可用，使用默认路径
            factor_dir = Path.cwd() / 'factors'
            logger.warning(f"配置管理器不可用，使用默认路径: {factor_dir}")
        
        filepath = factor_dir / f"{factor_name}.{format}"
        
        # 检查文件是否存在
        if not filepath.exists():
            raise FileNotFoundError(f"因子文件不存在: {filepath}")
        
        # 加载因子数据
        try:
            if format == 'pkl':
                with open(filepath, 'rb') as f:
                    factor_data = pickle.load(f)
            elif format == 'csv':
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
            
            # 验证加载的数据
            if not isinstance(factor_data, pd.Series):
                raise ValueError(f"加载的数据类型不正确，期望 pandas.Series，实际: {type(factor_data)}")
            
            logger.info(f"✅ 从 {filepath} 加载因子 {factor_name}，数据点数: {len(factor_data)}")
            return factor_data
            
        except Exception as e:
            logger.error(f"❌ 加载因子 {factor_name} 失败: {e}")
            raise
        
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
        
        # 定义因子到字段的映射关系
        self.factor_field_mapping = {
            'SUE': 'NET_PROFIT_EXCL_MIN_INT_INC',  # SUE使用扣非净利润
            'ROE_ttm': None,  # ROE需要多个字段，由计算器内部处理
            'CurrentRatio': None,  # 流动比率需要多个字段
            # 可以继续添加其他因子的字段映射
        }
    
    def _extract_required_field(self, factor_name: str, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        为指定因子提取所需字段
        
        Parameters
        ----------
        factor_name : str
            因子名称
        data : Union[pd.Series, pd.DataFrame]
            输入数据
            
        Returns
        -------
        pd.Series
            提取的单列数据
        """
        if isinstance(data, pd.Series):
            return data
        
        if isinstance(data, pd.DataFrame):
            # 检查是否有预定义的字段映射
            if factor_name in self.factor_field_mapping:
                required_field = self.factor_field_mapping[factor_name]
                if required_field and required_field in data.columns:
                    logger.info(f"为因子 {factor_name} 自动提取字段: {required_field}")
                    return data[required_field]
            
            # 对于特殊因子，使用因子自身的字段验证
            if factor_name == 'SUE':
                field_name = self.sue_calculator.validate_data_format(data)
                return data[field_name]
            
            # 如果没有映射且是单字段因子，抛出明确错误
            raise ValueError(
                f"因子 {factor_name} 需要单列数据，但收到了{len(data.columns)}列的DataFrame。\n"
                f"请手动指定字段，例如: generator.generate('{factor_name}', data['FIELD_NAME'])\n"
                f"可用字段: {list(data.columns[:5])}..."
            )
        
        raise TypeError(f"不支持的数据类型: {type(data)}")
        
    def generate(self,
                factor_name: str,
                data: Union[pd.Series, pd.DataFrame],
                **kwargs) -> pd.Series:
        """
        生成财务因子
        """
        # 首先提取因子所需的字段（如果输入是DataFrame）
        if isinstance(data, pd.DataFrame):
            logger.info(f"为因子 {factor_name} 处理 {data.shape[1]} 列的DataFrame")
            data = self._extract_required_field(factor_name, data)
        
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


class MixedFactorGenerator(FactorGenerator):
    """
    混合因子生成器
    
    生成需要多种数据源的复合因子
    """
    
    def __init__(self, **kwargs):
        super().__init__(factor_type='mixed', **kwargs)
        # 导入混合因子管理器
        from .mixed import MixedFactorManager
        self.manager = MixedFactorManager()
        
    def generate(self,
                factor_name: str,
                data: Union[pd.Series, Dict[str, Any]],
                **kwargs) -> pd.Series:
        """
        生成混合因子
        
        Parameters
        ----------
        factor_name : str
            因子名称
        data : Union[pd.Series, Dict[str, Any]]
            输入数据，混合因子需要字典格式，包含多种数据源
        **kwargs
            其他参数
            
        Returns
        -------
        pd.Series
            生成的因子数据
        """
        # 混合因子需要字典格式的数据
        if isinstance(data, pd.Series):
            raise ValueError(
                f"混合因子 {factor_name} 需要多种数据源，"
                f"请提供字典格式的数据，包含: {self.manager.get_data_requirements(factor_name)}"
            )
        
        if not isinstance(data, dict):
            raise ValueError(f"混合因子数据必须是字典格式，当前类型: {type(data)}")
        
        # 检查因子是否可用
        if factor_name not in self.manager.get_available_factors():
            raise ValueError(f"不支持的混合因子: {factor_name}")
        
        # 使用混合因子管理器计算
        try:
            result = self.manager.calculate_factor(factor_name, data, **kwargs)
            return result
        except Exception as e:
            logger.error(f"混合因子 {factor_name} 生成失败: {e}")
            raise
            
    def batch_generate(self,
                      factor_names: List[str],
                      data: Dict[str, Any],
                      **kwargs) -> Dict[str, pd.Series]:
        """
        批量生成多个混合因子
        
        重写基类方法以优化混合因子的批量计算
        """
        if not isinstance(data, dict):
            raise ValueError("混合因子批量计算需要字典格式的数据")
        
        try:
            # 使用管理器的批量计算功能
            result_df = self.manager.calculate_multiple_factors(factor_names, data, **kwargs)
            
            # 转换为字典格式
            results = {}
            for factor_name in factor_names:
                if factor_name in result_df.columns:
                    results[factor_name] = result_df[factor_name]
                    self.generated_factors[factor_name] = result_df[factor_name]
                else:
                    logger.warning(f"因子 {factor_name} 未能生成")
                    results[factor_name] = None
                    
            return results
        except Exception as e:
            logger.error(f"混合因子批量生成失败: {e}")
            # 回退到单个计算
            return super().batch_generate(factor_names, data, **kwargs)
            
    def get_available_factors(self) -> List[str]:
        """
        获取可用的混合因子列表
        """
        return self.manager.get_available_factors()
        
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """
        获取混合因子分类
        """
        all_factors = self.get_available_factors()
        categories = {}
        
        # 根据因子管理器获取分类信息
        for factor_name in all_factors:
            factor_info = self.manager.get_factor_info(factor_name)
            category = factor_info.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(factor_name)
            
        return categories
        
    def get_data_requirements(self, factor_names: Union[str, List[str]]) -> List[str]:
        """
        获取因子数据需求
        """
        return self.manager.get_data_requirements(factor_names)


# 工厂函数
def create_generator(factor_type: str, **kwargs) -> FactorGenerator:
    """
    创建因子生成器
    
    Parameters
    ----------
    factor_type : str
        因子类型（financial, technical, risk, mixed）
        
    Returns
    -------
    FactorGenerator
        对应的因子生成器实例
    """
    generators = {
        'financial': FinancialFactorGenerator,
        'technical': TechnicalFactorGenerator,
        'risk': RiskFactorGenerator,
        'mixed': MixedFactorGenerator
    }
    
    if factor_type not in generators:
        raise ValueError(f"不支持的因子类型: {factor_type}，支持的类型: {list(generators.keys())}")
        
    return generators[factor_type](**kwargs)


# 导出接口
__all__ = [
    'FactorGenerator',
    'FinancialFactorGenerator',
    'TechnicalFactorGenerator',
    'RiskFactorGenerator',
    'MixedFactorGenerator',
    'create_generator'
]