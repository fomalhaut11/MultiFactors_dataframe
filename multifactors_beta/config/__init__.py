#!/usr/bin/env python3
"""
统一配置管理模块

提供统一的配置文件访问接口，支持配置热更新和验证。

使用方式:
    from config import get_config, ConfigManager
    
    # 简单方式
    db_host = get_config('main.database.host')
    
    # 完整方式
    config = ConfigManager()
    db_config = config.get('main.database')
"""

from .manager import ConfigManager, get_config, reload_config, config_manager

# 便捷函数
def get_database_config(db_name: str = 'database'):
    """
    获取数据库配置的便捷函数
    
    Parameters
    ----------
    db_name : str, optional
        数据库名称，默认为'database'
        
    Returns
    -------
    dict
        数据库配置字典
    """
    base_config = get_config('main.database')
    if not base_config:
        return {}
    
    if db_name == 'database':
        return base_config
    
    # 如果请求特定数据库，查找对应的数据库名配置
    specific_db = base_config.get(db_name)
    if specific_db:
        # 返回基础配置的副本，但替换数据库名
        result = base_config.copy()
        result['database'] = specific_db
        return result
    
    return base_config

__all__ = [
    'ConfigManager',
    'get_config', 
    'reload_config',
    'config_manager',
    'get_database_config'
]

__version__ = '1.0.0'