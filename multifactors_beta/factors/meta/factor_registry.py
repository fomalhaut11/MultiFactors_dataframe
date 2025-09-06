#!/usr/bin/env python3
"""
因子注册表和元数据管理系统
管理因子的元信息、版本、状态和关系
"""

import os
import json
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """因子类型枚举"""
    FUNDAMENTAL = "fundamental"      # 基本面因子
    TECHNICAL = "technical"          # 技术面因子
    MOMENTUM = "momentum"            # 动量因子
    REVERSAL = "reversal"            # 反转因子
    VOLATILITY = "volatility"        # 波动率因子
    QUALITY = "quality"              # 质量因子
    GROWTH = "growth"                # 成长因子
    VALUE = "value"                  # 价值因子
    LIQUIDITY = "liquidity"          # 流动性因子
    ALPHA191 = "alpha191"            # Alpha191因子
    MIXED = "mixed"                  # 混合因子
    DERIVED = "derived"              # 衍生因子


class NeutralizationCategory(Enum):
    """中性化类别枚举"""
    MUST_NEUTRALIZE = "must"         # 必须中性化
    OPTIONAL_NEUTRALIZE = "optional" # 可选中性化
    SKIP_NEUTRALIZE = "skip"         # 跳过中性化
    ALREADY_NEUTRAL = "neutral"      # 已经中性化


@dataclass
class FactorMetadata:
    """因子元数据类"""
    name: str                                    # 因子名称
    type: FactorType                            # 因子类型
    description: str                            # 因子描述
    formula: Optional[str] = None               # 计算公式
    neutralization_category: NeutralizationCategory = NeutralizationCategory.OPTIONAL_NEUTRALIZE
    
    # 版本信息
    raw_version: Optional[str] = None           # 原始版本路径
    orthogonal_version: Optional[str] = None    # 正交化版本路径
    
    # 生成信息
    created_date: Optional[str] = None          # 创建日期
    updated_date: Optional[str] = None          # 更新日期
    generator: Optional[str] = None             # 生成器名称
    generator_params: Optional[Dict] = None     # 生成器参数
    
    # 中性化信息
    is_orthogonalized: bool = False             # 是否已正交化
    orthogonalization_date: Optional[str] = None # 正交化日期
    control_factors: Optional[List[str]] = None  # 控制因子列表
    orthogonalization_method: Optional[str] = None # 正交化方法
    
    # 数据统计
    data_range: Optional[Dict[str, str]] = None # 数据时间范围
    coverage: Optional[float] = None            # 数据覆盖率
    update_frequency: Optional[str] = None      # 更新频率
    
    # 测试结果
    test_results: Optional[Dict] = None         # 最新测试结果
    performance_metrics: Optional[Dict] = None  # 性能指标
    
    # 相关性信息
    highly_correlated_factors: Optional[List[str]] = None # 高相关因子
    correlation_threshold: Optional[float] = None         # 相关性阈值
    
    # 标签和分类
    tags: List[str] = field(default_factory=list)        # 标签列表
    category: Optional[str] = None                        # 分类
    priority: int = 0                                     # 优先级
    
    # 状态信息
    is_active: bool = True                      # 是否激活
    quality_score: Optional[float] = None       # 质量评分
    last_validation: Optional[str] = None       # 最后验证时间
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = asdict(self)
        # 处理枚举类型
        result['type'] = self.type.value if self.type else None
        result['neutralization_category'] = self.neutralization_category.value if self.neutralization_category else None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FactorMetadata':
        """从字典创建实例"""
        # 处理枚举类型
        if 'type' in data and data['type']:
            data['type'] = FactorType(data['type'])
        if 'neutralization_category' in data and data['neutralization_category']:
            data['neutralization_category'] = NeutralizationCategory(data['neutralization_category'])
        
        return cls(**data)


class FactorRegistry:
    """因子注册表管理器"""
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        初始化因子注册表
        
        Parameters
        ----------
        registry_path : str, optional
            注册表文件路径，默认为当前目录下的factor_registry.json
        """
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(__file__), 'factor_registry.json'
        )
        self.factors: Dict[str, FactorMetadata] = {}
        self.load_registry()
    
    def load_registry(self):
        """加载注册表"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, metadata in data.items():
                        self.factors[name] = FactorMetadata.from_dict(metadata)
                logger.info(f"加载因子注册表: {len(self.factors)} 个因子")
            except Exception as e:
                logger.error(f"加载因子注册表失败: {e}")
        else:
            logger.info("因子注册表不存在，将创建新的注册表")
    
    def save_registry(self):
        """保存注册表"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            
            # 转换为可序列化的格式
            data = {name: metadata.to_dict() for name, metadata in self.factors.items()}
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"保存因子注册表: {len(self.factors)} 个因子")
        except Exception as e:
            logger.error(f"保存因子注册表失败: {e}")
    
    def register_factor(
        self,
        name: str,
        factor_type: FactorType,
        description: str,
        **kwargs
    ) -> FactorMetadata:
        """
        注册新因子
        
        Parameters
        ----------
        name : str
            因子名称
        factor_type : FactorType
            因子类型
        description : str
            因子描述
        **kwargs
            其他元数据参数
            
        Returns
        -------
        FactorMetadata
            注册的因子元数据
        """
        # 设置默认值
        if 'created_date' not in kwargs:
            kwargs['created_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metadata = FactorMetadata(
            name=name,
            type=factor_type,
            description=description,
            **kwargs
        )
        
        self.factors[name] = metadata
        self.save_registry()
        
        logger.info(f"注册因子: {name} ({factor_type.value})")
        return metadata
    
    def update_factor(self, name: str, **kwargs) -> Optional[FactorMetadata]:
        """
        更新因子元数据
        
        Parameters
        ----------
        name : str
            因子名称
        **kwargs
            要更新的元数据字段
            
        Returns
        -------
        FactorMetadata or None
            更新后的因子元数据，如果因子不存在则返回None
        """
        if name not in self.factors:
            logger.warning(f"因子不存在: {name}")
            return None
        
        metadata = self.factors[name]
        
        # 更新字段
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # 更新时间戳
        metadata.updated_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.save_registry()
        logger.info(f"更新因子元数据: {name}")
        return metadata
    
    def get_factor(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self.factors.get(name)
    
    def list_factors(
        self,
        factor_type: Optional[FactorType] = None,
        active_only: bool = True,
        has_orthogonal: Optional[bool] = None
    ) -> List[FactorMetadata]:
        """
        列出因子
        
        Parameters
        ----------
        factor_type : FactorType, optional
            筛选因子类型
        active_only : bool
            是否只返回激活的因子
        has_orthogonal : bool, optional
            是否筛选有/无正交化版本的因子
            
        Returns
        -------
        List[FactorMetadata]
            符合条件的因子列表
        """
        factors = list(self.factors.values())
        
        if factor_type:
            factors = [f for f in factors if f.type == factor_type]
        
        if active_only:
            factors = [f for f in factors if f.is_active]
        
        if has_orthogonal is not None:
            factors = [f for f in factors if f.is_orthogonalized == has_orthogonal]
        
        return factors
    
    def mark_orthogonalized(
        self,
        name: str,
        orthogonal_path: str,
        control_factors: List[str],
        method: str = 'OLS'
    ) -> bool:
        """
        标记因子已正交化
        
        Parameters
        ----------
        name : str
            因子名称
        orthogonal_path : str
            正交化版本文件路径
        control_factors : List[str]
            控制因子列表
        method : str
            正交化方法
            
        Returns
        -------
        bool
            是否成功标记
        """
        if name not in self.factors:
            logger.warning(f"因子不存在: {name}")
            return False
        
        metadata = self.factors[name]
        metadata.is_orthogonalized = True
        metadata.orthogonal_version = orthogonal_path
        metadata.orthogonalization_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata.control_factors = control_factors
        metadata.orthogonalization_method = method
        
        self.save_registry()
        logger.info(f"标记因子已正交化: {name}")
        return True
    
    def get_neutralization_candidates(self) -> List[FactorMetadata]:
        """获取需要中性化的因子列表"""
        return [
            f for f in self.factors.values()
            if f.neutralization_category in [
                NeutralizationCategory.MUST_NEUTRALIZE,
                NeutralizationCategory.OPTIONAL_NEUTRALIZE
            ] and not f.is_orthogonalized and f.is_active
        ]
    
    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子统计信息"""
        total = len(self.factors)
        active = len([f for f in self.factors.values() if f.is_active])
        orthogonalized = len([f for f in self.factors.values() if f.is_orthogonalized])
        
        type_counts = {}
        for factor_type in FactorType:
            count = len([f for f in self.factors.values() if f.type == factor_type])
            if count > 0:
                type_counts[factor_type.value] = count
        
        neutralization_counts = {}
        for category in NeutralizationCategory:
            count = len([f for f in self.factors.values() if f.neutralization_category == category])
            if count > 0:
                neutralization_counts[category.value] = count
        
        return {
            'total_factors': total,
            'active_factors': active,
            'orthogonalized_factors': orthogonalized,
            'orthogonalization_rate': orthogonalized / total if total > 0 else 0,
            'factor_types': type_counts,
            'neutralization_categories': neutralization_counts,
            'registry_path': self.registry_path,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def export_to_csv(self, filepath: str):
        """导出注册表到CSV文件"""
        if not self.factors:
            logger.warning("没有因子数据可导出")
            return
        
        # 转换为DataFrame
        data = [metadata.to_dict() for metadata in self.factors.values()]
        df = pd.DataFrame(data)
        
        # 保存CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出因子注册表到: {filepath}")
    
    def import_from_csv(self, filepath: str):
        """从CSV文件导入注册表"""
        try:
            df = pd.read_csv(filepath)
            
            for _, row in df.iterrows():
                data = row.to_dict()
                # 处理NaN值
                data = {k: v for k, v in data.items() if pd.notna(v)}
                
                name = data.pop('name')
                metadata = FactorMetadata.from_dict(data)
                self.factors[name] = metadata
            
            self.save_registry()
            logger.info(f"从CSV文件导入 {len(df)} 个因子")
            
        except Exception as e:
            logger.error(f"从CSV导入失败: {e}")


# 全局因子注册表实例
_global_registry = None

def get_factor_registry() -> FactorRegistry:
    """获取全局因子注册表实例"""
    global _global_registry
    if _global_registry is None:
        # 默认注册表路径
        from config import get_config
        registry_dir = os.path.join(get_config('main.paths.project_root', '.'), 'factors', 'meta')
        os.makedirs(registry_dir, exist_ok=True)
        registry_path = os.path.join(registry_dir, 'factor_registry.json')
        _global_registry = FactorRegistry(registry_path)
    return _global_registry