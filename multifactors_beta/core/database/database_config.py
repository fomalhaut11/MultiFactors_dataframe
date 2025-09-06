#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库表名配置管理

该模块提供统一的数据库表名配置管理，避免在代码中硬编码表名。
通过配置文件集中管理所有数据库表名，提高代码的可维护性和可移植性。

Author: MultiFactors Team  
Date: 2025-08-28
"""

from typing import Dict, Any, Optional
import logging

from config import get_config

logger = logging.getLogger(__name__)


class DatabaseTableConfig:
    """
    数据库表名配置管理器
    
    从config.yaml中读取数据库表名配置，提供统一的表名访问接口。
    支持嵌套配置访问，如 'financial.lrb'。
    """
    
    def __init__(self):
        """初始化配置管理器"""
        try:
            self.tables = get_config('main.database.tables', {})
            if not self.tables:
                logger.warning("数据库表名配置为空，请检查config.yaml中的database.tables配置")
            else:
                logger.info(f"成功加载数据库表名配置，包含 {len(self.tables)} 个表配置")
        except Exception as e:
            logger.error(f"加载数据库表名配置失败: {e}")
            self.tables = {}
    
    def get_table(self, table_type: str, default: str = '') -> str:
        """
        获取表名
        
        Parameters
        ----------
        table_type : str
            表类型，支持点分隔的嵌套配置，如 'financial.lrb'
        default : str
            默认值，当配置不存在时返回
            
        Returns
        -------
        str
            数据库表名
            
        Examples
        --------
        >>> config = DatabaseTableConfig()
        >>> config.get_table('price')
        '[stock_data].[dbo].[day5]'
        >>> config.get_table('financial.lrb') 
        '[stock_data].[dbo].[lrb]'
        """
        if not table_type:
            return default
        
        try:
            if '.' in table_type:
                # 处理嵌套配置如 'financial.lrb'
                keys = table_type.split('.')
                result = self.tables
                for key in keys:
                    if isinstance(result, dict) and key in result:
                        result = result[key]
                    else:
                        logger.warning(f"表配置 '{table_type}' 不存在，使用默认值: '{default}'")
                        return default
                return result if isinstance(result, str) else default
            else:
                # 直接访问一级配置
                table_name = self.tables.get(table_type, default)
                if table_name == default and default == '':
                    logger.warning(f"表配置 '{table_type}' 不存在")
                return table_name
        except Exception as e:
            logger.error(f"获取表名配置 '{table_type}' 失败: {e}")
            return default
    
    def validate_config(self) -> bool:
        """
        验证配置完整性
        
        Returns
        -------
        bool
            配置是否有效
        """
        required_tables = [
            'price',
            'financial.lrb',
            'financial.xjlb', 
            'financial.fzb',
            'stop_price',
            'all_stocks'
        ]
        
        all_valid = True
        for table_type in required_tables:
            if not self.get_table(table_type):
                logger.error(f"缺少必需的表配置: {table_type}")
                all_valid = False
        
        return all_valid
    
    # === 便捷属性，提供常用表名的快速访问 ===
    
    @property
    def price_table(self) -> str:
        """价格数据表"""
        return self.get_table('price')
    
    @property
    def stop_price_table(self) -> str:
        """涨跌停数据表"""
        return self.get_table('stop_price')
    
    @property
    def all_stocks_table(self) -> str:
        """全部股票信息表"""
        return self.get_table('all_stocks')
    
    @property
    def st_stocks_table(self) -> str:
        """ST股票信息表"""
        return self.get_table('st_stocks')
    
    @property
    def wind_index_table(self) -> str:
        """Wind指数数据表"""
        return self.get_table('wind_index')
    
    @property
    def concept_table(self) -> str:
        """概念板块数据表"""
        return self.get_table('concept')
    
    @property
    def sector_changes_table(self) -> str:
        """板块进出调整数据表"""
        return self.get_table('sector_changes')
    
    @property
    def foreshow_table(self) -> str:
        """预报数据表"""
        return self.get_table('foreshow')
    
    @property
    def macro_industry_table(self) -> str:
        """宏观行业数据表"""
        return self.get_table('macro_industry')
    
    @property
    def us_treasury_table(self) -> str:
        """美国国债数据表"""
        return self.get_table('us_treasury')
    
    # === 财务数据表便捷访问 ===
    
    @property
    def lrb_table(self) -> str:
        """利润表"""
        return self.get_table('financial.lrb')
    
    @property
    def xjlb_table(self) -> str:
        """现金流量表"""
        return self.get_table('financial.xjlb')
    
    @property
    def fzb_table(self) -> str:
        """资产负债表"""
        return self.get_table('financial.fzb')
    
    def get_financial_tables(self) -> Dict[str, str]:
        """
        获取所有财务数据表
        
        Returns
        -------
        Dict[str, str]
            财务表名映射 {'lrb': 表名, 'xjlb': 表名, 'fzb': 表名}
        """
        return {
            'lrb': self.lrb_table,
            'xjlb': self.xjlb_table,
            'fzb': self.fzb_table
        }
    
    def print_config(self):
        """打印当前配置信息"""
        print("\n📋 数据库表名配置")
        print("=" * 50)
        
        if not self.tables:
            print("⚠️ 无可用配置")
            return
        
        def print_nested(data, indent=0):
            """递归打印嵌套配置"""
            prefix = "  " * indent
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"{prefix}📁 {key}:")
                    print_nested(value, indent + 1)
                else:
                    print(f"{prefix}📋 {key}: {value}")
        
        print_nested(self.tables)
        print("=" * 50)
        
        # 验证配置
        if self.validate_config():
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败，请检查必需的表配置")


# 全局实例
_global_db_config = None

def get_db_table_config() -> DatabaseTableConfig:
    """
    获取全局数据库表名配置实例
    
    Returns
    -------
    DatabaseTableConfig
        数据库表名配置实例
    """
    global _global_db_config
    if _global_db_config is None:
        _global_db_config = DatabaseTableConfig()
    return _global_db_config


# 便捷函数
def get_table_name(table_type: str, default: str = '') -> str:
    """
    便捷函数：获取表名
    
    Parameters
    ----------
    table_type : str
        表类型
    default : str
        默认值
        
    Returns
    -------
    str
        数据库表名
    """
    return get_db_table_config().get_table(table_type, default)


if __name__ == "__main__":
    # 测试代码
    config = DatabaseTableConfig()
    config.print_config()
    
    # 测试各种表名获取
    print(f"\n🧪 测试表名获取:")
    print(f"价格表: {config.price_table}")
    print(f"利润表: {config.lrb_table}")
    print(f"涨跌停表: {config.stop_price_table}")
    
    # 测试便捷函数
    print(f"便捷函数获取价格表: {get_table_name('price')}")