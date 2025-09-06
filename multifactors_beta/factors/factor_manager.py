#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子管理器
统一管理所有已实现的因子，提供计算、存储、加载等功能
"""

import pandas as pd
import numpy as np
import os
import pickle
import gzip
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# 导入项目模块
from .meta.factor_registry import get_factor_registry, FactorType
from .config.factor_configuration import get_factor_config_manager
from .register_factors import register_all_factors

# 导入所有因子类
from .generator.financial.profitability_factors import (
    ROE_ttm_Factor,
    ROA_ttm_Factor, 
    GrossProfitMargin_ttm_Factor,
    ProfitCost_ttm_Factor
)

from .generator.financial.solvency_factors import (
    CurrentRatio_Factor,
    DebtToAssets_Factor
)

from .generator.financial.legacy_financial_factors import (
    SUE_ttm_120d_Factor
)

from .generator.financial.value_factors import (
    EPRatioFactor,
    BPRatioFactor,
    SPRatioFactor,
    CFPRatioFactor,
    EarningsYieldFactor
)

from .generator.financial.quality_factors import (
    ROEQualityFactor,
    EarningsQualityFactor,
    DebtQualityFactor,
    ProfitabilityStabilityFactor,
    AssetQualityFactor
)

from .generator.technical.momentum_factors import (
    MomentumFactor,
    ShortTermReversalFactor,
    LongTermReversalFactor,
    TrendStrengthFactor,
    PriceMomentumFactor,
    VolatilityAdjustedMomentumFactor
)

logger = logging.getLogger(__name__)


class FactorManager:
    """
    因子管理器
    统一管理所有因子的计算、存储、加载等功能
    """
    
    def __init__(self, auto_register: bool = True):
        """
        初始化因子管理器
        
        Parameters:
        -----------
        auto_register : bool
            是否自动注册所有因子
        """
        self.registry = get_factor_registry()
        self.config_manager = get_factor_config_manager()
        self.factor_instances = {}  # 缓存因子实例
        
        # 因子类映射表
        self.factor_classes = {
            # 盈利能力因子
            'ROE_ttm': ROE_ttm_Factor,
            'ROA_ttm': ROA_ttm_Factor,
            'GrossProfitMargin_ttm': GrossProfitMargin_ttm_Factor,
            'ProfitCost_ttm': ProfitCost_ttm_Factor,
            
            # 偿债能力因子
            'CurrentRatio': CurrentRatio_Factor,
            'DebtToAssets': DebtToAssets_Factor,
            
            # 盈余惊喜因子
            'SUE_ttm_120d': SUE_ttm_120d_Factor,
            
            # 价值因子
            'EP_Ratio': EPRatioFactor,
            'BP_Ratio': BPRatioFactor,
            'SP_Ratio': SPRatioFactor,
            'CFP_Ratio': CFPRatioFactor,
            'EarningsYield': EarningsYieldFactor,
            
            # 质量因子
            'ROE_Quality_12': ROEQualityFactor,
            'EarningsQuality': EarningsQualityFactor,
            'DebtQuality': DebtQualityFactor,
            'ProfitabilityStability_20': ProfitabilityStabilityFactor,
            'AssetQuality': AssetQualityFactor,
            
            # 技术因子
            'Momentum_252_22': MomentumFactor,
            'ShortReversal_22': ShortTermReversalFactor,
            'LongReversal_504': LongTermReversalFactor,
            'TrendStrength_20_60': TrendStrengthFactor,
            'PriceMomentum_12_1': PriceMomentumFactor,
            'VolAdjMom_252_60': VolatilityAdjustedMomentumFactor
        }
        
        if auto_register:
            self._ensure_factors_registered()
    
    def _ensure_factors_registered(self):
        """确保所有因子都已注册"""
        try:
            # 检查是否已有因子注册
            stats = self.registry.get_factor_statistics()
            if stats['total_factors'] == 0:
                logger.info("未发现已注册因子，开始自动注册...")
                register_all_factors()
            else:
                logger.info(f"发现 {stats['total_factors']} 个已注册因子")
        except Exception as e:
            logger.error(f"确保因子注册时出错: {e}")
    
    def get_factor_instance(self, factor_name: str):
        """获取因子实例（带缓存）"""
        if factor_name not in self.factor_instances:
            if factor_name in self.factor_classes:
                factor_class = self.factor_classes[factor_name]
                self.factor_instances[factor_name] = factor_class()
            else:
                raise ValueError(f"未找到因子类: {factor_name}")
        
        return self.factor_instances[factor_name]
    
    def list_available_factors(self, factor_type: Optional[FactorType] = None) -> List[Dict[str, Any]]:
        """
        列出所有可用因子
        
        Parameters:
        -----------
        factor_type : FactorType, optional
            筛选特定类型的因子
            
        Returns:
        --------
        List[Dict] : 因子信息列表
        """
        factors = self.registry.list_factors(factor_type=factor_type)
        
        result = []
        for factor_metadata in factors:
            factor_info = {
                'name': factor_metadata.name,
                'type': factor_metadata.type.value if factor_metadata.type else 'unknown',
                'category': factor_metadata.category,
                'description': factor_metadata.description,
                'tags': factor_metadata.tags,
                'formula': factor_metadata.formula,
                'available': factor_metadata.name in self.factor_classes
            }
            result.append(factor_info)
        
        return result
    
    def calculate_factor(self, factor_name: str, data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
        """
        计算单个因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        data : pd.DataFrame or pd.Series
            输入数据
        **kwargs : dict
            额外参数
            
        Returns:
        --------
        pd.Series : 因子值
        """
        try:
            factor_instance = self.get_factor_instance(factor_name)
            
            # 获取因子特定配置
            factor_config = self.config_manager.get_factor_config(factor_name)
            kwargs.update(factor_config.get('custom_parameters', {}))
            
            logger.info(f"开始计算因子: {factor_name}")
            start_time = datetime.now()
            
            result = factor_instance.calculate(data, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"因子 {factor_name} 计算完成，耗时 {duration:.2f} 秒")
            
            return result
            
        except Exception as e:
            logger.error(f"计算因子 {factor_name} 失败: {e}")
            raise
    
    def calculate_multiple_factors(self, 
                                 factor_names: List[str], 
                                 data: Union[pd.DataFrame, pd.Series],
                                 parallel: bool = True,
                                 **kwargs) -> pd.DataFrame:
        """
        批量计算多个因子
        
        Parameters:
        -----------
        factor_names : List[str]
            因子名称列表
        data : pd.DataFrame or pd.Series
            输入数据
        parallel : bool
            是否并行计算
        **kwargs : dict
            额外参数
            
        Returns:
        --------
        pd.DataFrame : 所有因子值
        """
        results = {}
        
        if parallel and len(factor_names) > 1:
            # 并行计算
            compute_config = self.config_manager.get_compute_config()
            max_workers = min(compute_config.parallel_workers, len(factor_names))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_factor = {
                    executor.submit(self.calculate_factor, factor_name, data, **kwargs): factor_name
                    for factor_name in factor_names
                }
                
                # 收集结果
                for future in as_completed(future_to_factor):
                    factor_name = future_to_factor[future]
                    try:
                        result = future.result()
                        results[factor_name] = result
                    except Exception as e:
                        logger.error(f"并行计算因子 {factor_name} 失败: {e}")
                        results[factor_name] = pd.Series(dtype=float)
        else:
            # 串行计算
            for factor_name in factor_names:
                try:
                    result = self.calculate_factor(factor_name, data, **kwargs)
                    results[factor_name] = result
                except Exception as e:
                    logger.error(f"计算因子 {factor_name} 失败: {e}")
                    results[factor_name] = pd.Series(dtype=float)
        
        # 合并结果
        if results:
            factor_df = pd.DataFrame(results)
            logger.info(f"批量计算完成，共 {len(results)} 个因子")
            return factor_df
        else:
            logger.warning("没有成功计算的因子")
            return pd.DataFrame()
    
    def calculate_factors_by_category(self, 
                                    category: str, 
                                    data: Union[pd.DataFrame, pd.Series],
                                    **kwargs) -> pd.DataFrame:
        """
        按类别计算因子
        
        Parameters:
        -----------
        category : str
            因子类别
        data : pd.DataFrame or pd.Series
            输入数据
        **kwargs : dict
            额外参数
            
        Returns:
        --------
        pd.DataFrame : 该类别下所有因子值
        """
        # 获取该类别下的所有因子
        all_factors = self.list_available_factors()
        category_factors = [
            f['name'] for f in all_factors 
            if f['category'] == category and f['available']
        ]
        
        if not category_factors:
            logger.warning(f"类别 {category} 下没有可用因子")
            return pd.DataFrame()
        
        logger.info(f"计算类别 {category} 下的 {len(category_factors)} 个因子")
        return self.calculate_multiple_factors(category_factors, data, **kwargs)
    
    def save_factor(self, factor_name: str, factor_data: pd.Series, 
                   factor_type: str = 'processed', metadata: Optional[Dict] = None):
        """
        保存因子数据
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        factor_data : pd.Series
            因子数据
        factor_type : str
            因子类型 ('raw', 'processed', 'orthogonal')
        metadata : dict, optional
            元数据
        """
        try:
            file_path = self.config_manager.get_factor_storage_path(factor_name, factor_type)
            
            # 准备保存的数据
            save_data = {
                'factor_data': factor_data,
                'metadata': {
                    'factor_name': factor_name,
                    'factor_type': factor_type,
                    'created_time': datetime.now().isoformat(),
                    'data_shape': factor_data.shape,
                    'data_range': {
                        'start_date': factor_data.index.get_level_values(0).min().strftime('%Y-%m-%d') if len(factor_data) > 0 else None,
                        'end_date': factor_data.index.get_level_values(0).max().strftime('%Y-%m-%d') if len(factor_data) > 0 else None,
                        'stocks_count': len(factor_data.index.get_level_values(1).unique()) if len(factor_data) > 0 else 0
                    },
                    'custom_metadata': metadata or {}
                }
            }
            
            # 根据配置选择保存格式
            storage_config = self.config_manager.get_storage_config()
            
            if storage_config.file_format == 'pickle':
                if storage_config.compression == 'gzip':
                    with gzip.open(file_path, 'wb') as f:
                        pickle.dump(save_data, f)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(save_data, f)
            elif storage_config.file_format == 'parquet':
                # 对于parquet格式，需要特殊处理
                factor_df = factor_data.reset_index()
                factor_df['factor_value'] = factor_data.values
                factor_df.to_parquet(file_path, compression='gzip' if storage_config.compression == 'gzip' else None)
            else:
                raise ValueError(f"不支持的文件格式: {storage_config.file_format}")
            
            logger.info(f"因子 {factor_name} 已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"保存因子 {factor_name} 失败: {e}")
            raise
    
    def load_factor(self, factor_name: str, factor_type: str = 'processed') -> Tuple[pd.Series, Dict]:
        """
        加载因子数据
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        factor_type : str
            因子类型 ('raw', 'processed', 'orthogonal')
            
        Returns:
        --------
        Tuple[pd.Series, Dict] : 因子数据和元数据
        """
        try:
            file_path = self.config_manager.get_factor_storage_path(factor_name, factor_type)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"因子文件不存在: {file_path}")
            
            storage_config = self.config_manager.get_storage_config()
            
            if storage_config.file_format == 'pickle':
                if storage_config.compression == 'gzip':
                    with gzip.open(file_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                
                factor_data = data['factor_data']
                metadata = data.get('metadata', {})
                
            elif storage_config.file_format == 'parquet':
                factor_df = pd.read_parquet(file_path)
                # 重构MultiIndex
                factor_data = factor_df.set_index(['TradingDates', 'StockCodes'])['factor_value']
                metadata = {}  # parquet格式下元数据需要单独存储
            
            else:
                raise ValueError(f"不支持的文件格式: {storage_config.file_format}")
            
            logger.info(f"成功加载因子 {factor_name} 从: {file_path}")
            return factor_data, metadata
            
        except Exception as e:
            logger.error(f"加载因子 {factor_name} 失败: {e}")
            raise
    
    def get_factor_summary(self, factor_name: str) -> Dict[str, Any]:
        """
        获取因子摘要信息
        
        Parameters:
        -----------
        factor_name : str
            因子名称
            
        Returns:
        --------
        Dict : 因子摘要信息
        """
        summary = {}
        
        try:
            # 从注册表获取基本信息
            factor_metadata = self.registry.get_factor(factor_name)
            if factor_metadata:
                summary.update({
                    'name': factor_metadata.name,
                    'type': factor_metadata.type.value if factor_metadata.type else 'unknown',
                    'category': factor_metadata.category,
                    'description': factor_metadata.description,
                    'formula': factor_metadata.formula,
                    'tags': factor_metadata.tags,
                    'created_date': factor_metadata.created_date,
                    'updated_date': factor_metadata.updated_date
                })
            
            # 尝试加载因子数据获取统计信息
            try:
                factor_data, metadata = self.load_factor(factor_name, 'processed')
                
                summary.update({
                    'data_available': True,
                    'data_points': len(factor_data),
                    'date_range': metadata.get('data_range', {}),
                    'last_update': metadata.get('created_time'),
                    'statistics': {
                        'mean': factor_data.mean(),
                        'std': factor_data.std(),
                        'min': factor_data.min(),
                        'max': factor_data.max(),
                        'missing_ratio': factor_data.isna().sum() / len(factor_data)
                    }
                })
            except:
                summary['data_available'] = False
            
            # 获取配置信息
            factor_config = self.config_manager.get_factor_config(factor_name)
            if factor_config:
                summary['configuration'] = factor_config
            
        except Exception as e:
            logger.error(f"获取因子摘要 {factor_name} 失败: {e}")
            summary = {'error': str(e)}
        
        return summary
    
    def validate_factor_data(self, factor_data: pd.Series, factor_name: str) -> Dict[str, Any]:
        """
        验证因子数据质量
        
        Parameters:
        -----------
        factor_data : pd.Series
            因子数据
        factor_name : str
            因子名称
            
        Returns:
        --------
        Dict : 验证结果
        """
        quality_config = self.config_manager.get_quality_config()
        
        validation_result = {
            'factor_name': factor_name,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # 基本统计信息
            validation_result['statistics'] = {
                'total_points': len(factor_data),
                'missing_points': factor_data.isna().sum(),
                'missing_ratio': factor_data.isna().sum() / len(factor_data) if len(factor_data) > 0 else 0,
                'mean': factor_data.mean(),
                'std': factor_data.std(),
                'min': factor_data.min(),
                'max': factor_data.max(),
                'infinite_count': np.isinf(factor_data).sum()
            }
            
            # 检查缺失值比例
            missing_ratio = validation_result['statistics']['missing_ratio']
            if missing_ratio > quality_config.max_missing_ratio:
                validation_result['errors'].append(
                    f"缺失值比例 ({missing_ratio:.2%}) 超过阈值 ({quality_config.max_missing_ratio:.2%})"
                )
                validation_result['is_valid'] = False
            elif missing_ratio > quality_config.max_missing_ratio * 0.5:
                validation_result['warnings'].append(
                    f"缺失值比例 ({missing_ratio:.2%}) 较高"
                )
            
            # 检查无穷值
            infinite_count = validation_result['statistics']['infinite_count']
            if infinite_count > 0:
                validation_result['warnings'].append(f"发现 {infinite_count} 个无穷值")
            
            # 检查数据分布（异常值检测）
            if len(factor_data.dropna()) > 0:
                clean_data = factor_data.dropna()
                q1 = clean_data.quantile(0.25)
                q3 = clean_data.quantile(0.75)
                iqr = q3 - q1
                
                outlier_threshold = quality_config.outlier_threshold
                lower_bound = q1 - outlier_threshold * iqr
                upper_bound = q3 + outlier_threshold * iqr
                
                outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
                outlier_ratio = outliers / len(clean_data)
                
                validation_result['statistics']['outliers'] = {
                    'count': outliers,
                    'ratio': outlier_ratio,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                
                if outlier_ratio > 0.1:  # 超过10%的异常值
                    validation_result['warnings'].append(
                        f"异常值比例 ({outlier_ratio:.2%}) 较高"
                    )
            
        except Exception as e:
            validation_result['errors'].append(f"验证过程出错: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result


# 全局因子管理器实例
_factor_manager = None

def get_factor_manager() -> FactorManager:
    """获取全局因子管理器实例"""
    global _factor_manager
    if _factor_manager is None:
        _factor_manager = FactorManager()
    return _factor_manager


# 便捷函数
def list_factors(factor_type: Optional[FactorType] = None) -> List[Dict[str, Any]]:
    """列出所有可用因子"""
    return get_factor_manager().list_available_factors(factor_type)

def calculate_factor(factor_name: str, data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    """计算单个因子"""
    return get_factor_manager().calculate_factor(factor_name, data, **kwargs)

def calculate_factors(factor_names: List[str], data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.DataFrame:
    """计算多个因子"""
    return get_factor_manager().calculate_multiple_factors(factor_names, data, **kwargs)

def get_factor_info(factor_name: str) -> Dict[str, Any]:
    """获取因子信息"""
    return get_factor_manager().get_factor_summary(factor_name)


if __name__ == "__main__":
    # 测试因子管理器
    manager = FactorManager()
    
    # 列出所有可用因子
    factors = manager.list_available_factors()
    print(f"发现 {len(factors)} 个可用因子:")
    
    for factor in factors[:10]:  # 显示前10个
        print(f"  - {factor['name']} ({factor['type']}): {factor['description']}")
    
    # 获取因子摘要
    if factors:
        first_factor = factors[0]['name']
        summary = manager.get_factor_summary(first_factor)
        print(f"\n{first_factor} 摘要信息:")
        for key, value in summary.items():
            if key != 'statistics':  # 统计信息可能很长
                print(f"  {key}: {value}")
    
    print("\n因子管理器初始化完成！")