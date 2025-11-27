#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新因子注册表模块

管理实验因子的注册、状态跟踪和元数据管理
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging
from enum import Enum
from dataclasses import dataclass, asdict
import pickle

from config import get_config

logger = logging.getLogger(__name__)


class FactorStatus(Enum):
    """因子状态枚举"""
    REGISTERED = "registered"          # 已注册，未计算
    CALCULATED = "calculated"          # 已计算，未测试
    TESTING = "testing"                # 测试中
    TESTED = "tested"                  # 已测试，待评估
    VALIDATED = "validated"            # 验证通过，可提升
    FAILED = "failed"                  # 验证失败
    PROMOTED = "promoted"              # 已提升到正式repository
    ARCHIVED = "archived"              # 已归档


@dataclass
class ExperimentalFactor:
    """实验因子数据结构"""
    name: str
    description: str
    calculation_func: Callable
    status: FactorStatus
    category: str                      # financial/technical/mixed
    created_time: datetime
    last_updated: datetime
    
    # 元数据
    author: str = ""
    version: str = "1.0"
    dependencies: List[str] = None
    expected_frequency: str = "daily"  # daily/weekly/monthly
    
    # 计算相关
    calculation_params: Dict[str, Any] = None
    data_requirements: List[str] = None
    
    # 测试相关
    test_results: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    
    # 状态历史
    status_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.calculation_params is None:
            self.calculation_params = {}
        if self.data_requirements is None:
            self.data_requirements = []
        if self.test_results is None:
            self.test_results = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.status_history is None:
            self.status_history = []
    
    def update_status(self, new_status: FactorStatus, note: str = ""):
        """更新状态并记录历史"""
        old_status = self.status
        self.status = new_status
        self.last_updated = datetime.now()
        
        # 记录状态变更历史
        self.status_history.append({
            'timestamp': self.last_updated,
            'old_status': old_status.value,
            'new_status': new_status.value,
            'note': note
        })
        
        logger.info(f"因子 {self.name} 状态从 {old_status.value} 更新为 {new_status.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        data = asdict(self)
        # 处理不可序列化的字段
        data['calculation_func'] = None  # 函数对象不序列化
        data['status'] = self.status.value
        data['created_time'] = self.created_time.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        
        # 处理status_history中的datetime
        for history_item in data['status_history']:
            if isinstance(history_item['timestamp'], datetime):
                history_item['timestamp'] = history_item['timestamp'].isoformat()
        
        return data


class ExperimentalFactorRegistry:
    """
    实验因子注册表
    
    功能：
    1. 因子注册和元数据管理
    2. 状态跟踪和历史记录
    3. 持久化存储和加载
    4. 查询和筛选接口
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        初始化注册表
        
        Parameters:
        -----------
        registry_path : str, optional
            注册表存储路径，默认从config读取
        """
        if registry_path is None:
            try:
                base_path = get_config('main.paths.factors_data')
                self.registry_path = Path(base_path) / "experimental_lab" / "registry"
            except:
                self.registry_path = Path("data/factors/experimental_lab/registry")
        else:
            self.registry_path = Path(registry_path)
        
        # 确保目录存在
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # 注册表文件
        self.metadata_file = self.registry_path / "factor_metadata.json"
        self.functions_file = self.registry_path / "factor_functions.pkl"
        
        # 内存中的注册表
        self._registry: Dict[str, ExperimentalFactor] = {}
        self._functions: Dict[str, Callable] = {}
        
        # 加载现有数据
        self._load_registry()
        
        logger.info(f"实验因子注册表初始化完成，路径: {self.registry_path}")
    
    def register_factor(self, 
                       name: str, 
                       calculation_func: Callable,
                       description: str = "",
                       category: str = "experimental",
                       author: str = "",
                       **metadata) -> ExperimentalFactor:
        """
        注册新的实验因子
        
        Parameters:
        -----------
        name : str
            因子名称（唯一标识）
        calculation_func : Callable
            因子计算函数
        description : str
            因子描述
        category : str
            因子分类
        author : str
            作者
        **metadata : dict
            其他元数据
            
        Returns:
        --------
        ExperimentalFactor
            注册的因子对象
        """
        if name in self._registry:
            logger.warning(f"因子 {name} 已存在，将更新现有注册")
            existing_factor = self._registry[name]
            existing_factor.calculation_func = calculation_func
            existing_factor.description = description or existing_factor.description
            existing_factor.last_updated = datetime.now()
            # 更新其他元数据
            for key, value in metadata.items():
                setattr(existing_factor, key, value)
            
            self._functions[name] = calculation_func
            self._save_registry()
            return existing_factor
        
        # 创建新因子
        now = datetime.now()
        factor = ExperimentalFactor(
            name=name,
            description=description,
            calculation_func=calculation_func,
            status=FactorStatus.REGISTERED,
            category=category,
            created_time=now,
            last_updated=now,
            author=author,
            **metadata
        )
        
        # 添加到注册表
        self._registry[name] = factor
        self._functions[name] = calculation_func
        
        # 记录初始状态
        factor.update_status(FactorStatus.REGISTERED, "初始注册")
        
        # 持久化
        self._save_registry()
        
        logger.info(f"成功注册实验因子: {name}")
        return factor
    
    def get_factor(self, name: str) -> Optional[ExperimentalFactor]:
        """获取指定因子"""
        return self._registry.get(name)
    
    def get_calculation_function(self, name: str) -> Optional[Callable]:
        """获取因子计算函数"""
        return self._functions.get(name)
    
    def list_factors(self, status: Optional[FactorStatus] = None, 
                    category: Optional[str] = None) -> List[ExperimentalFactor]:
        """
        列出因子
        
        Parameters:
        -----------
        status : FactorStatus, optional
            按状态筛选
        category : str, optional
            按分类筛选
            
        Returns:
        --------
        List[ExperimentalFactor]
            符合条件的因子列表
        """
        factors = list(self._registry.values())
        
        if status:
            factors = [f for f in factors if f.status == status]
        
        if category:
            factors = [f for f in factors if f.category == category]
        
        # 按创建时间排序
        factors.sort(key=lambda f: f.created_time, reverse=True)
        return factors
    
    def update_factor_status(self, name: str, new_status: FactorStatus, note: str = ""):
        """更新因子状态"""
        if name not in self._registry:
            raise ValueError(f"因子 {name} 不存在")
        
        factor = self._registry[name]
        factor.update_status(new_status, note)
        self._save_registry()
    
    def update_test_results(self, name: str, test_results: Dict[str, Any], 
                           performance_metrics: Dict[str, float] = None):
        """更新测试结果"""
        if name not in self._registry:
            raise ValueError(f"因子 {name} 不存在")
        
        factor = self._registry[name]
        factor.test_results.update(test_results)
        
        if performance_metrics:
            factor.performance_metrics.update(performance_metrics)
        
        factor.last_updated = datetime.now()
        self._save_registry()
        
        logger.info(f"更新因子 {name} 测试结果")
    
    def remove_factor(self, name: str, reason: str = ""):
        """移除因子"""
        if name not in self._registry:
            logger.warning(f"因子 {name} 不存在，无需移除")
            return
        
        # 先更新状态为归档
        self.update_factor_status(name, FactorStatus.ARCHIVED, f"移除原因: {reason}")
        
        # 从内存中移除
        del self._registry[name]
        if name in self._functions:
            del self._functions[name]
        
        self._save_registry()
        logger.info(f"移除因子 {name}: {reason}")
    
    def get_factor_summary(self) -> pd.DataFrame:
        """获取因子汇总表"""
        if not self._registry:
            return pd.DataFrame()
        
        data = []
        for factor in self._registry.values():
            row = {
                'name': factor.name,
                'category': factor.category,
                'status': factor.status.value,
                'created_time': factor.created_time,
                'last_updated': factor.last_updated,
                'author': factor.author,
                'description': factor.description[:100] if factor.description else ""
            }
            
            # 添加性能指标
            if factor.performance_metrics:
                for metric, value in factor.performance_metrics.items():
                    row[f'metric_{metric}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('last_updated', ascending=False)
        return df
    
    def export_for_screening(self, status_filter: List[FactorStatus] = None) -> Dict[str, Any]:
        """
        导出数据供筛选器使用
        
        Parameters:
        -----------
        status_filter : List[FactorStatus], optional
            状态筛选列表，默认为已测试的因子
            
        Returns:
        --------
        Dict[str, Any]
            筛选器需要的数据格式
        """
        if status_filter is None:
            status_filter = [FactorStatus.TESTED, FactorStatus.VALIDATED]
        
        filtered_factors = [f for f in self._registry.values() 
                          if f.status in status_filter]
        
        export_data = {
            'factors': {},
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_factors': len(filtered_factors),
                'status_filter': [s.value for s in status_filter]
            }
        }
        
        for factor in filtered_factors:
            export_data['factors'][factor.name] = {
                'basic_info': {
                    'name': factor.name,
                    'category': factor.category,
                    'description': factor.description,
                    'author': factor.author
                },
                'performance': factor.performance_metrics,
                'test_results': factor.test_results,
                'status': factor.status.value
            }
        
        return export_data
    
    def _load_registry(self):
        """从文件加载注册表"""
        try:
            # 加载元数据
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 重建因子对象
                for name, data in metadata.items():
                    # 恢复datetime对象
                    data['created_time'] = datetime.fromisoformat(data['created_time'])
                    data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                    data['status'] = FactorStatus(data['status'])
                    
                    # 处理status_history
                    for history_item in data['status_history']:
                        history_item['timestamp'] = datetime.fromisoformat(history_item['timestamp'])
                    
                    # 创建因子对象（不包含函数）
                    data['calculation_func'] = None  # 函数从pickle文件加载
                    factor = ExperimentalFactor(**data)
                    self._registry[name] = factor
            
            # 加载函数
            if self.functions_file.exists():
                with open(self.functions_file, 'rb') as f:
                    self._functions = pickle.load(f)
                
                # 将函数重新关联到因子对象
                for name, func in self._functions.items():
                    if name in self._registry:
                        self._registry[name].calculation_func = func
            
            logger.info(f"加载注册表完成，共 {len(self._registry)} 个因子")
            
        except Exception as e:
            logger.warning(f"加载注册表失败: {e}，将创建新的注册表")
    
    def _save_registry(self):
        """保存注册表到文件"""
        try:
            # 保存元数据
            metadata = {}
            for name, factor in self._registry.items():
                metadata[name] = factor.to_dict()
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 保存函数
            with open(self.functions_file, 'wb') as f:
                pickle.dump(self._functions, f)
            
            logger.debug(f"保存注册表完成，共 {len(self._registry)} 个因子")
            
        except Exception as e:
            logger.error(f"保存注册表失败: {e}")
            raise


if __name__ == "__main__":
    # 测试代码
    def test_calculation():
        """测试计算函数"""
        import pandas as pd
        return pd.Series([1, 2, 3], name="test_factor")
    
    # 创建注册表实例
    registry = ExperimentalFactorRegistry()
    
    # 注册测试因子
    factor = registry.register_factor(
        name="test_momentum_factor",
        calculation_func=test_calculation,
        description="测试动量因子",
        category="technical",
        author="AI Assistant"
    )
    
    print(f"注册因子: {factor.name}, 状态: {factor.status.value}")
    
    # 获取汇总表
    summary = registry.get_factor_summary()
    print("\n因子汇总:")
    print(summary)