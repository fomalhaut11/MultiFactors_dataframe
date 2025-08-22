#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块
提供统一的数据库连接管理和SQL执行功能
"""

from .connection_manager import (
    DatabaseManager,
    ConnectionPool,
    db_manager,
    get_connection,
    test_connection,
    get_pool_status
)

from .sql_executor import (
    SQLExecutor,
    sql_executor,
    execute_query,
    execute_query_to_dataframe,
    execute_stock_data_query,
    get_table_info
)

# 导出主要接口
__all__ = [
    # 连接管理器
    'DatabaseManager',
    'ConnectionPool', 
    'db_manager',
    'get_connection',
    'test_connection',
    'get_pool_status',
    
    # SQL执行器
    'SQLExecutor',
    'sql_executor',
    'execute_query',
    'execute_query_to_dataframe', 
    'execute_stock_data_query',
    'get_table_info'
]