"""
灵活的数据适配器 - 减少对具体列名的硬依赖
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging
from ..config import get_column_name, factor_config

logger = logging.getLogger(__name__)


@dataclass
class DataSchema:
    """数据模式定义"""
    earnings_column: str = None
    equity_column: str = None
    revenue_column: str = None
    assets_column: str = None
    quarter_column: str = None
    year_column: str = None
    release_date_column: str = None
    
    @classmethod
    def from_config(cls, data_type: str = 'default') -> 'DataSchema':
        """从配置创建数据模式"""
        mappings = factor_config.COLUMN_MAPPINGS
        return cls(
            earnings_column=mappings.get('earnings'),
            equity_column=mappings.get('equity'),
            revenue_column=mappings.get('revenue'),
            assets_column=mappings.get('total_assets'),
            quarter_column=mappings.get('quarter'),
            year_column=mappings.get('year'),
            release_date_column=mappings.get('release_date')
        )
    
    def get_column(self, logical_name: str) -> Optional[str]:
        """根据逻辑名称获取列名"""
        column_map = {
            'earnings': self.earnings_column,
            'equity': self.equity_column,
            'revenue': self.revenue_column,
            'assets': self.assets_column,
            'quarter': self.quarter_column,
            'year': self.year_column,
            'release_date': self.release_date_column
        }
        return column_map.get(logical_name)


class FlexibleDataAdapter:
    """
    灵活的数据适配器，支持多种数据格式和列名映射
    """
    
    def __init__(self, schema: Optional[DataSchema] = None, 
                 custom_mappings: Optional[Dict[str, str]] = None):
        """
        初始化数据适配器
        
        Parameters:
        -----------
        schema : 数据模式定义
        custom_mappings : 自定义列名映射
        """
        self.schema = schema or DataSchema.from_config()
        self.custom_mappings = custom_mappings or {}
        
    def get_column_name(self, logical_name: str) -> str:
        """
        获取实际列名
        
        Parameters:
        -----------
        logical_name : 逻辑列名
        
        Returns:
        --------
        实际列名
        
        Raises:
        -------
        ValueError : 如果找不到对应的列名
        """
        # 首先检查自定义映射
        if logical_name in self.custom_mappings:
            return self.custom_mappings[logical_name]
        
        # 然后检查模式定义
        column_name = self.schema.get_column(logical_name)
        if column_name:
            return column_name
        
        # 最后使用配置中的默认映射
        default_name = get_column_name(logical_name)
        if default_name != logical_name:  # 如果找到了映射
            return default_name
        
        raise ValueError(f"No column mapping found for logical name: {logical_name}")
    
    def extract_data(self, 
                    data: pd.DataFrame, 
                    required_columns: List[str],
                    optional_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从数据中提取所需列
        
        Parameters:
        -----------
        data : 输入数据
        required_columns : 必需的逻辑列名
        optional_columns : 可选的逻辑列名
        
        Returns:
        --------
        提取的数据，使用逻辑列名
        """
        result_data = {}
        
        # 处理必需列
        for logical_name in required_columns:
            try:
                actual_name = self.get_column_name(logical_name)
                if actual_name not in data.columns:
                    raise ValueError(f"Required column '{actual_name}' not found in data")
                result_data[logical_name] = data[actual_name]
            except ValueError as e:
                logger.error(f"Failed to map required column '{logical_name}': {e}")
                raise
        
        # 处理可选列
        if optional_columns:
            for logical_name in optional_columns:
                try:
                    actual_name = self.get_column_name(logical_name)
                    if actual_name in data.columns:
                        result_data[logical_name] = data[actual_name]
                    else:
                        logger.info(f"Optional column '{actual_name}' not found, skipping")
                except ValueError:
                    logger.info(f"No mapping found for optional column '{logical_name}', skipping")
        
        # 保持原始索引
        result = pd.DataFrame(result_data, index=data.index)
        
        return result
    
    def validate_data_availability(self, 
                                  data: pd.DataFrame, 
                                  required_columns: List[str]) -> Dict[str, bool]:
        """
        验证数据可用性
        
        Parameters:
        -----------
        data : 输入数据
        required_columns : 必需的逻辑列名
        
        Returns:
        --------
        可用性检查结果
        """
        availability = {}
        
        for logical_name in required_columns:
            try:
                actual_name = self.get_column_name(logical_name)
                availability[logical_name] = actual_name in data.columns
            except ValueError:
                availability[logical_name] = False
        
        return availability
    
    def suggest_column_mappings(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        基于列名模糊匹配，建议可能的列映射
        
        Parameters:
        -----------
        data : 输入数据
        
        Returns:
        --------
        建议的列映射
        """
        suggestions = {}
        
        # 定义模糊匹配规则
        fuzzy_rules = {
            'earnings': ['profit', 'earning', '利润', '净利', 'netincome'],
            'equity': ['equity', 'owner', '权益', '股东', 'sharehol'],
            'revenue': ['revenue', 'income', 'sales', '收入', '营业'],
            'assets': ['asset', '资产', 'total'],
            'quarter': ['quarter', 'q', '季', '季度'],
            'year': ['year', 'y', '年', '年度'],
            'release_date': ['release', 'publish', 'date', '发布', '公布']
        }
        
        available_columns = data.columns.tolist()
        
        for logical_name, keywords in fuzzy_rules.items():
            matches = []
            for col in available_columns:
                col_lower = str(col).lower()
                for keyword in keywords:
                    if keyword.lower() in col_lower:
                        matches.append(col)
                        break
            
            if matches:
                suggestions[logical_name] = matches
        
        return suggestions
    
    def create_flexible_extractor(self, 
                                 required_columns: List[str],
                                 optional_columns: Optional[List[str]] = None):
        """
        创建灵活的数据提取器函数
        
        Parameters:
        -----------
        required_columns : 必需的逻辑列名
        optional_columns : 可选的逻辑列名
        
        Returns:
        --------
        数据提取器函数
        """
        def extractor(data: pd.DataFrame) -> pd.DataFrame:
            return self.extract_data(data, required_columns, optional_columns)
        
        return extractor


class ColumnMapperMixin:
    """
    列映射混入类，为因子类提供灵活的列访问能力
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_adapter = None
        self._custom_mappings = {}
    
    def set_column_mapping(self, logical_name: str, actual_name: str):
        """
        设置自定义列映射
        
        Parameters:
        -----------
        logical_name : 逻辑列名
        actual_name : 实际列名
        """
        self._custom_mappings[logical_name] = actual_name
        # 重新创建适配器
        self._data_adapter = FlexibleDataAdapter(custom_mappings=self._custom_mappings)
    
    def get_data_adapter(self) -> FlexibleDataAdapter:
        """获取数据适配器"""
        if self._data_adapter is None:
            self._data_adapter = FlexibleDataAdapter(custom_mappings=self._custom_mappings)
        return self._data_adapter
    
    def extract_required_data(self, 
                            data: pd.DataFrame, 
                            required_columns: List[str],
                            optional_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        提取所需数据
        
        Parameters:
        -----------
        data : 输入数据
        required_columns : 必需的逻辑列名
        optional_columns : 可选的逻辑列名
        
        Returns:
        --------
        提取的数据
        """
        adapter = self.get_data_adapter()
        return adapter.extract_data(data, required_columns, optional_columns)
    
    def suggest_mappings_for_data(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """为数据建议列映射"""
        adapter = self.get_data_adapter()
        return adapter.suggest_column_mappings(data)
    
    def validate_data_requirements(self, 
                                  data: pd.DataFrame, 
                                  required_columns: List[str]) -> bool:
        """
        验证数据是否满足需求
        
        Parameters:
        -----------
        data : 输入数据
        required_columns : 必需的逻辑列名
        
        Returns:
        --------
        是否满足需求
        """
        adapter = self.get_data_adapter()
        availability = adapter.validate_data_availability(data, required_columns)
        
        missing_columns = [col for col, available in availability.items() if not available]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # 提供建议
            suggestions = adapter.suggest_column_mappings(data)
            for missing_col in missing_columns:
                if missing_col in suggestions:
                    logger.info(f"Suggested mappings for '{missing_col}': {suggestions[missing_col]}")
            return False
        
        return True


# 便捷的工厂函数
def create_earnings_extractor(custom_mappings: Optional[Dict[str, str]] = None):
    """创建盈利数据提取器"""
    adapter = FlexibleDataAdapter(custom_mappings=custom_mappings)
    return adapter.create_flexible_extractor(['earnings', 'quarter'])


def create_balance_sheet_extractor(custom_mappings: Optional[Dict[str, str]] = None):
    """创建资产负债表数据提取器"""
    adapter = FlexibleDataAdapter(custom_mappings=custom_mappings)
    return adapter.create_flexible_extractor(
        required_columns=['equity'],
        optional_columns=['assets', 'revenue']
    )


def create_comprehensive_extractor(custom_mappings: Optional[Dict[str, str]] = None):
    """创建综合数据提取器"""
    adapter = FlexibleDataAdapter(custom_mappings=custom_mappings)
    return adapter.create_flexible_extractor(
        required_columns=['earnings', 'equity', 'quarter'],
        optional_columns=['revenue', 'assets', 'year']
    )