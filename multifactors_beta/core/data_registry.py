#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据更新注册器

管理和注册所有可用的数据更新器，提供统一的数据更新接口
并为用户提供数据可用性查询功能

Author: MultiFactors Team
Date: 2025-08-28
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from config import get_config

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型枚举"""
    PRICE = "price"
    FINANCIAL = "financial"
    CLASSIFICATION = "classification" 
    MARKET = "market"
    MACRO = "macro"
    PROCESSED = "processed"
    AUXILIARY = "auxiliary"


class UpdateFrequency(Enum):
    """更新频率枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    description: str
    data_type: DataType
    file_path: Path
    update_frequency: UpdateFrequency
    updater_class: str
    dependencies: List[str] = field(default_factory=list)
    
    # 数据状态
    is_available: bool = False
    last_update: Optional[datetime] = None
    data_range: Optional[Tuple[datetime, datetime]] = None
    file_size: Optional[int] = None
    record_count: Optional[int] = None
    
    # 更新配置
    auto_update: bool = True
    update_params: Dict[str, Any] = field(default_factory=dict)


class DataRegistry:
    """数据更新注册器"""
    
    def __init__(self):
        self.data_root = Path(get_config('main.paths.data_root'))
        self.datasets: Dict[str, DatasetInfo] = {}
        self._initialize_core_datasets()
        
        logger.info("数据注册器初始化完成")
    
    def _initialize_core_datasets(self):
        """初始化核心数据集"""
        
        # 价格数据
        self.register_dataset(DatasetInfo(
            name="price_data",
            description="股票日线价格数据",
            data_type=DataType.PRICE,
            file_path=self.data_root / "Price.pkl",
            update_frequency=UpdateFrequency.DAILY,
            updater_class="PriceDataUpdater",
            dependencies=[]
        ))
        
        # 财务数据
        for table_name, description in [
            ("lrb", "利润表数据"),
            ("fzb", "资产负债表数据"), 
            ("xjlb", "现金流量表数据")
        ]:
            self.register_dataset(DatasetInfo(
                name=f"financial_{table_name}",
                description=description,
                data_type=DataType.FINANCIAL,
                file_path=self.data_root / f"{table_name}.pkl",
                update_frequency=UpdateFrequency.DAILY,
                updater_class="FinancialDataUpdater",
                dependencies=[],
                update_params={"table_type": table_name}
            ))
        
        # 分类数据
        self.register_dataset(DatasetInfo(
            name="sector_changes",
            description="板块进出调整数据",
            data_type=DataType.CLASSIFICATION,
            file_path=self.data_root / "SectorChanges_data.pkl",
            update_frequency=UpdateFrequency.DAILY,
            updater_class="SectorChangesDataUpdater",
            dependencies=[]
        ))
        
        self.register_dataset(DatasetInfo(
            name="stock_classification",
            description="股票分类信息（计算得出）",
            data_type=DataType.CLASSIFICATION,
            file_path=self.data_root / "StockClassification_20250828.pkl",
            update_frequency=UpdateFrequency.ON_DEMAND,
            updater_class="SectorClassificationProcessor",
            dependencies=["sector_changes"]
        ))
        
        # 市场数据
        self.register_dataset(DatasetInfo(
            name="st_stocks",
            description="ST股票数据",
            data_type=DataType.MARKET,
            file_path=self.data_root / "ST_stocks.pkl",
            update_frequency=UpdateFrequency.DAILY,
            updater_class="STDataUpdater",
            dependencies=[]
        ))
        
        self.register_dataset(DatasetInfo(
            name="stop_price",
            description="涨跌停板数据",
            data_type=DataType.MARKET,
            file_path=self.data_root / "StopPrice.pkl",
            update_frequency=UpdateFrequency.DAILY,
            updater_class="StopPriceDataUpdater",
            dependencies=[]
        ))
        
        # 辅助数据（基础辅助文件）
        auxiliary_data = [
            ("trading_dates", "交易日期序列", "TradingDates.pkl", []),
            ("release_dates", "财报发布日期", "ReleaseDates.pkl", ["financial_lrb", "financial_fzb", "financial_xjlb"]),
            ("stock_info", "股票基本信息", "StockInfo.pkl", []),
            ("financial_data_unified", "统一财务数据", "FinancialData_unified.pkl", ["financial_lrb", "financial_fzb", "financial_xjlb"]),
            ("market_cap", "市值数据", "MarketCap.pkl", ["price_data"]),
        ]
        
        for name, description, filename, deps in auxiliary_data:
            self.register_dataset(DatasetInfo(
                name=name,
                description=description,
                data_type=DataType.AUXILIARY,
                file_path=self.data_root / f"auxiliary/{filename}",
                update_frequency=UpdateFrequency.ON_DEMAND,
                updater_class="AuxiliaryDataPreparer",
                dependencies=deps
            ))
            
        # 辅助数据（收益率数据）
        return_files = [
            ("LogReturn_daily_o2o", "日收益率(开到开)"),
            ("LogReturn_daily_vwap", "日收益率(VWAP)"),
            ("LogReturn_weekly_o2o", "周收益率(开到开)"),
            ("LogReturn_monthly_o2o", "月收益率(开到开)"),
            ("LogReturn_5days_o2o", "5天收益率"),
            ("LogReturn_20days_o2o", "20天收益率")
        ]
        
        # 股票池数据
        universe_files = [
            ("liquid_300", "沪深300流动性股票池"),
            ("liquid_500", "中证500流动性股票池"), 
            ("liquid_1000", "中证1000流动性股票池"),
            ("custom_universe", "自定义股票池数据")
        ]
        
        for file_name, description in return_files:
            self.register_dataset(DatasetInfo(
                name=file_name.lower(),
                description=description,
                data_type=DataType.AUXILIARY,  # 改为AUXILIARY
                file_path=self.data_root / f"auxiliary/{file_name}.pkl",  # 移到auxiliary目录
                update_frequency=UpdateFrequency.ON_DEMAND,
                updater_class="AuxiliaryDataPreparer",  # 改为AuxiliaryDataPreparer
                dependencies=["price_data", "trading_dates"],  # 明确依赖
                auto_update=False
            ))
        
        # 注册股票池数据
        try:
            # 尝试使用专用的股票池路径
            universe_root = Path(get_config('main.paths.stock_universe'))
        except:
            # 向后兼容
            universe_root = self.data_root / 'stock_universe'
        
        for universe_name, description in universe_files:
            self.register_dataset(DatasetInfo(
                name=f"universe_{universe_name}",
                description=description,
                data_type=DataType.AUXILIARY,
                file_path=universe_root / f"{universe_name}_universe.pkl",
                update_frequency=UpdateFrequency.ON_DEMAND,
                updater_class="StockUniverseManager",
                dependencies=["price_data", "market_cap", "trading_dates"],
                auto_update=False
            ))
        
        # 注册现有的高成交额股票池
        self.register_dataset(DatasetInfo(
            name="universe_high_turnover_5d_20m",
            description="高成交额股票池（5日平均成交额超过2000万元）",
            data_type=DataType.AUXILIARY,
            file_path=universe_root / "high_turnover_5d_20m.pkl",
            update_frequency=UpdateFrequency.ON_DEMAND,
            updater_class="StockUniverseManager",
            dependencies=["price_data"],
            auto_update=False
        ))
    
    def register_dataset(self, dataset_info: DatasetInfo):
        """注册数据集"""
        self.datasets[dataset_info.name] = dataset_info
        self._update_dataset_status(dataset_info.name)
        logger.debug(f"注册数据集: {dataset_info.name}")
    
    def _update_dataset_status(self, dataset_name: str):
        """更新数据集状态信息"""
        dataset = self.datasets[dataset_name]
        
        try:
            if dataset.file_path.exists():
                dataset.is_available = True
                dataset.file_size = dataset.file_path.stat().st_size
                dataset.last_update = datetime.fromtimestamp(
                    dataset.file_path.stat().st_mtime
                )
                
                # 尝试读取数据获取更多信息
                if dataset.file_path.suffix == '.pkl':
                    try:
                        data = pd.read_pickle(dataset.file_path)
                        if isinstance(data, pd.DataFrame):
                            dataset.record_count = len(data)
                            if hasattr(data, 'index') and isinstance(data.index, pd.MultiIndex):
                                # 对于MultiIndex，获取日期范围
                                if 'date' in data.index.names or len(data.index.names) > 0:
                                    dates = data.index.get_level_values(0)
                                    if pd.api.types.is_datetime64_any_dtype(dates):
                                        dataset.data_range = (dates.min(), dates.max())
                            elif pd.api.types.is_datetime64_any_dtype(data.index):
                                dataset.data_range = (data.index.min(), data.index.max())
                        elif isinstance(data, pd.Series):
                            dataset.record_count = len(data)
                    except Exception as e:
                        logger.debug(f"读取数据文件 {dataset.file_path} 失败: {e}")
            else:
                dataset.is_available = False
                dataset.file_size = None
                dataset.last_update = None
                dataset.record_count = None
                dataset.data_range = None
                
        except Exception as e:
            logger.error(f"更新数据集状态失败 {dataset_name}: {e}")
            dataset.is_available = False
    
    def get_available_datasets(self, data_type: Optional[DataType] = None) -> List[DatasetInfo]:
        """获取可用的数据集列表"""
        available = [ds for ds in self.datasets.values() if ds.is_available]
        
        if data_type:
            available = [ds for ds in available if ds.data_type == data_type]
        
        return sorted(available, key=lambda x: x.name)
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """获取数据集信息"""
        if dataset_name in self.datasets:
            self._update_dataset_status(dataset_name)
            return self.datasets[dataset_name]
        return None
    
    def list_all_datasets(self) -> pd.DataFrame:
        """列出所有数据集信息"""
        data = []
        for name, dataset in self.datasets.items():
            self._update_dataset_status(name)
            
            data.append({
                'name': name,
                'description': dataset.description,
                'data_type': dataset.data_type.value,
                'update_frequency': dataset.update_frequency.value,
                'is_available': dataset.is_available,
                'file_size_mb': round(dataset.file_size / 1024 / 1024, 2) if dataset.file_size else None,
                'record_count': dataset.record_count,
                'last_update': dataset.last_update.strftime('%Y-%m-%d %H:%M:%S') if dataset.last_update else None,
                'data_range': f"{dataset.data_range[0].date()} - {dataset.data_range[1].date()}" if dataset.data_range else None,
                'updater_class': dataset.updater_class,
                'dependencies': ', '.join(dataset.dependencies) if dataset.dependencies else None
            })
        
        df = pd.DataFrame(data)
        return df.sort_values(['data_type', 'name'])
    
    def get_update_plan(self, target_datasets: Optional[List[str]] = None) -> List[str]:
        """获取数据更新计划（考虑依赖关系）"""
        if target_datasets is None:
            target_datasets = list(self.datasets.keys())
        
        # 简单的拓扑排序，考虑依赖关系
        update_order = []
        visited = set()
        
        def visit(dataset_name: str):
            if dataset_name in visited:
                return
            
            dataset = self.datasets.get(dataset_name)
            if not dataset:
                return
                
            # 先处理依赖项
            for dep in dataset.dependencies:
                if dep in self.datasets:
                    visit(dep)
            
            if dataset_name not in update_order:
                update_order.append(dataset_name)
            visited.add(dataset_name)
        
        for dataset_name in target_datasets:
            visit(dataset_name)
        
        return update_order
    
    def check_data_freshness(self, hours_threshold: int = 24) -> Dict[str, bool]:
        """检查数据新鲜度"""
        threshold_time = datetime.now() - timedelta(hours=hours_threshold)
        freshness = {}
        
        for name, dataset in self.datasets.items():
            if dataset.is_available and dataset.last_update:
                freshness[name] = dataset.last_update > threshold_time
            else:
                freshness[name] = False
        
        return freshness
    
    def get_missing_datasets(self) -> List[str]:
        """获取缺失的数据集"""
        return [name for name, dataset in self.datasets.items() if not dataset.is_available]
    
    def print_data_summary(self):
        """打印数据摘要"""
        total = len(self.datasets)
        available = len([ds for ds in self.datasets.values() if ds.is_available])
        
        print(f"\n=== 数据注册器摘要 ===")
        print(f"总数据集数量: {total}")
        print(f"可用数据集: {available}")
        print(f"缺失数据集: {total - available}")
        
        # 按类型统计
        by_type = {}
        for dataset in self.datasets.values():
            data_type = dataset.data_type.value
            if data_type not in by_type:
                by_type[data_type] = {'total': 0, 'available': 0}
            by_type[data_type]['total'] += 1
            if dataset.is_available:
                by_type[data_type]['available'] += 1
        
        print(f"\n按数据类型统计:")
        for data_type, counts in by_type.items():
            print(f"  {data_type}: {counts['available']}/{counts['total']}")
        
        # 缺失的数据集
        missing = self.get_missing_datasets()
        if missing:
            print(f"\n缺失的数据集:")
            for name in missing:
                print(f"  - {name}: {self.datasets[name].description}")
        
        print("=" * 30)


# 全局注册器实例
_data_registry = None

def get_data_registry() -> DataRegistry:
    """获取数据注册器单例"""
    global _data_registry
    if _data_registry is None:
        _data_registry = DataRegistry()
    return _data_registry


if __name__ == "__main__":
    # 测试代码
    registry = get_data_registry()
    
    # 打印摘要
    registry.print_data_summary()
    
    # 显示所有数据集
    print("\n=== 所有数据集详情 ===")
    df = registry.list_all_datasets()
    print(df.to_string(index=False))
    
    # 检查数据新鲜度
    freshness = registry.check_data_freshness()
    print("\n=== 数据新鲜度检查 (24小时) ===")
    for name, is_fresh in freshness.items():
        status = "✅ 新鲜" if is_fresh else "⚠️ 过时"
        print(f"{name}: {status}")
    
    # 更新计划
    update_plan = registry.get_update_plan()
    print(f"\n=== 建议更新顺序 ===")
    for i, name in enumerate(update_plan, 1):
        print(f"{i}. {name}")