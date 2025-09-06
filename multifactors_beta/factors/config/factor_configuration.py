#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子配置管理系统
管理因子的计算参数、存储路径、更新频率等配置信息
"""

import os
import json
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FactorStorageConfig:
    """因子存储配置"""
    base_path: str                              # 基础存储路径
    raw_factors_path: str                       # 原始因子存储路径
    processed_factors_path: str                 # 处理后因子存储路径
    orthogonal_factors_path: str               # 正交化因子存储路径
    backup_path: str                           # 备份路径
    compression: str = 'gzip'                  # 压缩格式
    file_format: str = 'pickle'                # 文件格式 (pickle, parquet, hdf5)
    max_backup_versions: int = 5               # 最大备份版本数


@dataclass
class FactorUpdateConfig:
    """因子更新配置"""
    update_frequency: str                       # 更新频率 (daily, weekly, monthly, quarterly)
    update_time: str                           # 更新时间 (e.g., "09:30", "15:30")
    lookback_days: int                         # 回看天数
    min_data_points: int                       # 最少数据点数
    skip_weekends: bool = True                 # 跳过周末
    skip_holidays: bool = True                 # 跳过假期
    force_update_on_error: bool = False        # 出错时是否强制更新
    max_retry_attempts: int = 3                # 最大重试次数
    

@dataclass
class FactorComputeConfig:
    """因子计算配置"""
    batch_size: int = 1000                     # 批处理大小
    parallel_workers: int = 4                  # 并行工作进程数
    memory_limit: str = "8GB"                  # 内存限制
    timeout_seconds: int = 3600                # 超时时间（秒）
    enable_caching: bool = True                # 启用缓存
    cache_ttl_hours: int = 24                  # 缓存生存时间（小时）
    enable_logging: bool = True                # 启用日志
    log_level: str = "INFO"                    # 日志级别


@dataclass
class FactorQualityConfig:
    """因子质量控制配置"""
    outlier_method: str = "IQR"                # 异常值处理方法
    outlier_threshold: float = 3.0             # 异常值阈值
    min_coverage_ratio: float = 0.8            # 最小覆盖率
    max_missing_ratio: float = 0.2             # 最大缺失率
    standardize_method: str = "zscore"         # 标准化方法
    neutralize_factors: List[str] = None       # 中性化因子
    industry_neutralize: bool = True           # 行业中性化
    market_cap_neutralize: bool = True         # 市值中性化


class FactorConfigurationManager:
    """因子配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Parameters:
        -----------
        config_path : str, optional
            配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            config_dir = os.path.dirname(__file__)
            config_path = os.path.join(config_dir, 'factor_config.yaml')
        
        self.config_path = config_path
        self.configs = {}
        self.load_configuration()
    
    def load_configuration(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.configs = yaml.safe_load(f)
                logger.info(f"加载因子配置: {self.config_path}")
            else:
                # 创建默认配置
                self.create_default_configuration()
                self.save_configuration()
                
        except Exception as e:
            logger.error(f"加载因子配置失败: {e}")
            self.create_default_configuration()
    
    def create_default_configuration(self):
        """创建默认配置"""
        # 获取项目根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        factors_dir = os.path.join(project_root, 'factor_data')
        
        self.configs = {
            'storage': asdict(FactorStorageConfig(
                base_path=factors_dir,
                raw_factors_path=os.path.join(factors_dir, 'raw'),
                processed_factors_path=os.path.join(factors_dir, 'processed'),
                orthogonal_factors_path=os.path.join(factors_dir, 'orthogonal'),
                backup_path=os.path.join(factors_dir, 'backup')
            )),
            
            'update': asdict(FactorUpdateConfig(
                update_frequency='daily',
                update_time='09:30',
                lookback_days=252,
                min_data_points=60
            )),
            
            'compute': asdict(FactorComputeConfig()),
            
            'quality': asdict(FactorQualityConfig(
                neutralize_factors=['market_cap', 'industry']
            )),
            
            'factor_specific': {}  # 特定因子的配置
        }
        
        # 确保目录存在
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """确保所有配置的目录都存在"""
        storage_config = self.configs.get('storage', {})
        
        for key, path in storage_config.items():
            if key.endswith('_path') and path:
                os.makedirs(path, exist_ok=True)
    
    def save_configuration(self):
        """保存配置"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.configs, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"保存因子配置: {self.config_path}")
            
        except Exception as e:
            logger.error(f"保存因子配置失败: {e}")
    
    def get_storage_config(self) -> FactorStorageConfig:
        """获取存储配置"""
        storage_dict = self.configs.get('storage', {})
        return FactorStorageConfig(**storage_dict)
    
    def get_update_config(self) -> FactorUpdateConfig:
        """获取更新配置"""
        update_dict = self.configs.get('update', {})
        return FactorUpdateConfig(**update_dict)
    
    def get_compute_config(self) -> FactorComputeConfig:
        """获取计算配置"""
        compute_dict = self.configs.get('compute', {})
        return FactorComputeConfig(**compute_dict)
    
    def get_quality_config(self) -> FactorQualityConfig:
        """获取质量控制配置"""
        quality_dict = self.configs.get('quality', {})
        if quality_dict.get('neutralize_factors') is None:
            quality_dict['neutralize_factors'] = []
        return FactorQualityConfig(**quality_dict)
    
    def get_factor_config(self, factor_name: str) -> Dict[str, Any]:
        """
        获取特定因子的配置
        
        Parameters:
        -----------
        factor_name : str
            因子名称
            
        Returns:
        --------
        dict : 因子特定配置
        """
        factor_specific = self.configs.get('factor_specific', {})
        return factor_specific.get(factor_name, {})
    
    def set_factor_config(self, factor_name: str, config: Dict[str, Any]):
        """
        设置特定因子的配置
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        config : dict
            配置字典
        """
        if 'factor_specific' not in self.configs:
            self.configs['factor_specific'] = {}
        
        self.configs['factor_specific'][factor_name] = config
        self.save_configuration()
        
        logger.info(f"更新因子 {factor_name} 的配置")
    
    def update_config_section(self, section: str, config: Dict[str, Any]):
        """
        更新配置的某个部分
        
        Parameters:
        -----------
        section : str
            配置部分名称 (storage, update, compute, quality)
        config : dict
            配置字典
        """
        if section in self.configs:
            self.configs[section].update(config)
        else:
            self.configs[section] = config
        
        self.save_configuration()
        logger.info(f"更新配置部分: {section}")
    
    def get_factor_storage_path(self, factor_name: str, 
                               factor_type: str = 'processed') -> str:
        """
        获取因子的存储路径
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        factor_type : str
            因子类型 ('raw', 'processed', 'orthogonal')
            
        Returns:
        --------
        str : 存储路径
        """
        storage_config = self.get_storage_config()
        
        if factor_type == 'raw':
            base_path = storage_config.raw_factors_path
        elif factor_type == 'processed':
            base_path = storage_config.processed_factors_path
        elif factor_type == 'orthogonal':
            base_path = storage_config.orthogonal_factors_path
        else:
            raise ValueError(f"未知因子类型: {factor_type}")
        
        # 创建因子特定的子目录
        factor_dir = os.path.join(base_path, factor_name)
        os.makedirs(factor_dir, exist_ok=True)
        
        # 生成文件名
        file_ext = {
            'pickle': '.pkl',
            'parquet': '.parquet',
            'hdf5': '.h5'
        }.get(storage_config.file_format, '.pkl')
        
        filename = f"{factor_name}{file_ext}"
        if storage_config.compression == 'gzip' and storage_config.file_format == 'pickle':
            filename += '.gz'
        
        return os.path.join(factor_dir, filename)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        验证配置的有效性
        
        Returns:
        --------
        dict : 验证结果，包含错误和警告信息
        """
        errors = []
        warnings = []
        
        # 验证存储配置
        storage_config = self.get_storage_config()
        if not os.path.exists(storage_config.base_path):
            errors.append(f"基础存储路径不存在: {storage_config.base_path}")
        
        # 验证更新配置
        update_config = self.get_update_config()
        if update_config.update_frequency not in ['daily', 'weekly', 'monthly', 'quarterly']:
            errors.append(f"无效的更新频率: {update_config.update_frequency}")
        
        # 验证计算配置
        compute_config = self.get_compute_config()
        if compute_config.parallel_workers <= 0:
            errors.append("并行工作进程数必须大于0")
        
        if compute_config.batch_size <= 0:
            errors.append("批处理大小必须大于0")
        
        # 验证质量控制配置
        quality_config = self.get_quality_config()
        if not 0 <= quality_config.min_coverage_ratio <= 1:
            errors.append("最小覆盖率必须在0-1之间")
        
        if not 0 <= quality_config.max_missing_ratio <= 1:
            errors.append("最大缺失率必须在0-1之间")
        
        return {
            'errors': errors,
            'warnings': warnings
        }
    
    def export_configuration_template(self, output_path: str):
        """导出配置模板"""
        template = {
            'storage': asdict(FactorStorageConfig(
                base_path='./factor_data',
                raw_factors_path='./factor_data/raw',
                processed_factors_path='./factor_data/processed',
                orthogonal_factors_path='./factor_data/orthogonal',
                backup_path='./factor_data/backup'
            )),
            'update': asdict(FactorUpdateConfig(
                update_frequency='daily',
                update_time='09:30',
                lookback_days=252,
                min_data_points=60
            )),
            'compute': asdict(FactorComputeConfig()),
            'quality': asdict(FactorQualityConfig()),
            'factor_specific': {
                'example_factor': {
                    'enabled': True,
                    'priority': 1,
                    'custom_parameters': {
                        'lookback_period': 252,
                        'min_periods': 60
                    }
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        logger.info(f"配置模板已导出到: {output_path}")


# 全局配置管理器实例
_config_manager = None

def get_factor_config_manager() -> FactorConfigurationManager:
    """获取全局因子配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = FactorConfigurationManager()
    return _config_manager


# 便捷函数
def get_storage_config() -> FactorStorageConfig:
    """获取存储配置"""
    return get_factor_config_manager().get_storage_config()


def get_update_config() -> FactorUpdateConfig:
    """获取更新配置"""
    return get_factor_config_manager().get_update_config()


def get_compute_config() -> FactorComputeConfig:
    """获取计算配置"""
    return get_factor_config_manager().get_compute_config()


def get_quality_config() -> FactorQualityConfig:
    """获取质量控制配置"""
    return get_factor_config_manager().get_quality_config()


def get_factor_storage_path(factor_name: str, factor_type: str = 'processed') -> str:
    """获取因子存储路径"""
    return get_factor_config_manager().get_factor_storage_path(factor_name, factor_type)


if __name__ == "__main__":
    # 测试配置管理器
    manager = FactorConfigurationManager()
    
    # 验证配置
    validation = manager.validate_configuration()
    if validation['errors']:
        print("配置错误:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("配置警告:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # 导出模板
    template_path = os.path.join(os.path.dirname(__file__), 'factor_config_template.yaml')
    manager.export_configuration_template(template_path)
    print(f"配置模板已导出到: {template_path}")