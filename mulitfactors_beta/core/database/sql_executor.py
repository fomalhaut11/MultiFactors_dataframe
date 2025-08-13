#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQL执行器
提供统一的SQL执行功能，支持查询、更新等操作
"""

import pandas as pd
import numpy as np
import logging
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from .connection_manager import get_connection, get_pool_status, cleanup_idle_connections

logger = logging.getLogger(__name__)


class SQLExecutor:
    """SQL执行器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.query_stats = {
            'total_queries': 0,
            'total_time': 0,
            'failed_queries': 0,
            'retried_queries': 0,
            'connection_errors': 0,
            'timeout_errors': 0
        }
        self.last_cleanup = datetime.now()
    
    def _should_retry(self, error: Exception) -> bool:
        """判断是否应该重试"""
        error_str = str(error).lower()
        # SQL Server常见的可重试错误
        retry_keywords = [
            'timeout', 'connection', 'network', 'broken pipe',
            'connection reset', 'server has gone away', 'lost connection'
        ]
        return any(keyword in error_str for keyword in retry_keywords)
    
    def _categorize_error(self, error: Exception):
        """对错误进行分类统计"""
        error_str = str(error).lower()
        if 'timeout' in error_str:
            self.query_stats['timeout_errors'] += 1
        elif any(keyword in error_str for keyword in ['connection', 'network']):
            self.query_stats['connection_errors'] += 1
    
    def _periodic_cleanup(self):
        """定期清理空闲连接"""
        if (datetime.now() - self.last_cleanup).total_seconds() > 1800:  # 30分钟
            try:
                cleanup_idle_connections(force=False)
                self.last_cleanup = datetime.now()
                logger.debug("执行定期连接池清理")
            except Exception as e:
                logger.warning(f"定期清理失败: {e}")
    
    def execute_query(self, 
                     sql: str, 
                     db_name: str = 'database',
                     params: Optional[Dict] = None,
                     timeout: int = 300) -> List[Tuple]:
        """
        执行SQL查询并返回原始结果（带重试机制）
        
        Args:
            sql: SQL查询语句
            db_name: 数据库名称
            params: 查询参数
            timeout: 查询超时时间（秒）
            
        Returns:
            查询结果列表
        """
        start_time = time.time()
        self.query_stats['total_queries'] += 1
        
        # 定期清理连接
        self._periodic_cleanup()
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with get_connection(db_name) as conn:
                    cursor = conn.cursor()
                    
                    # 设置查询超时
                    if hasattr(cursor, 'settimeout'):
                        cursor.settimeout(timeout)
                    
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)
                    
                    results = cursor.fetchall()
                    cursor.close()
                    
                    query_time = time.time() - start_time
                    self.query_stats['total_time'] += query_time
                    
                    if attempt > 0:
                        logger.info(f"重试成功，第{attempt}次重试后查询完成")
                    
                    logger.debug(f"SQL查询完成，耗时: {query_time:.2f}秒，返回 {len(results)} 条记录")
                    return results
                    
            except Exception as e:
                last_error = e
                self._categorize_error(e)
                
                if attempt == self.max_retries or not self._should_retry(e):
                    # 最后一次尝试或不可重试的错误
                    break
                
                # 需要重试
                self.query_stats['retried_queries'] += 1
                retry_delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                logger.warning(f"查询失败，将在{retry_delay:.1f}秒后重试 (第{attempt+1}次): {e}")
                time.sleep(retry_delay)
        
        # 所有重试都失败了
        self.query_stats['failed_queries'] += 1
        logger.error(f"SQL查询最终失败: {last_error}")
        logger.error(f"SQL语句: {sql[:200]}{'...' if len(sql) > 200 else ''}")
        raise last_error
    
    def execute_query_to_dataframe(self, 
                                  sql: str, 
                                  columns: List[str],
                                  db_name: str = 'database',
                                  params: Optional[Dict] = None,
                                  dtype_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行SQL查询并返回DataFrame
        
        Args:
            sql: SQL查询语句
            columns: 列名列表
            db_name: 数据库名称
            params: 查询参数
            dtype_mapping: 数据类型映射
            
        Returns:
            查询结果DataFrame
        """
        results = self.execute_query(sql, db_name, params)
        
        if not results:
            logger.warning("SQL查询返回空结果")
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(results, columns=columns)
        
        # 应用数据类型映射
        if dtype_mapping:
            for col, dtype in dtype_mapping.items():
                if col in df.columns:
                    try:
                        if dtype == 'datetime':
                            df[col] = pd.to_datetime(df[col], format="%Y%m%d")
                        else:
                            df[col] = df[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"列 {col} 类型转换失败: {e}")
        
        # 处理常见的缺失值标记
        df = df.replace(123456789.0, np.nan)
        
        logger.info(f"SQL查询转换为DataFrame完成，形状: {df.shape}")
        return df
    
    def execute_batch_queries(self, 
                             queries: List[Dict[str, Any]], 
                             db_name: str = 'database') -> List[pd.DataFrame]:
        """
        批量执行SQL查询
        
        Args:
            queries: 查询配置列表，每个包含sql、columns等信息
            db_name: 数据库名称
            
        Returns:
            查询结果DataFrame列表
        """
        results = []
        
        with get_connection(db_name) as conn:
            for i, query_config in enumerate(queries):
                try:
                    sql = query_config['sql']
                    columns = query_config['columns']
                    params = query_config.get('params')
                    
                    cursor = conn.cursor()
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)
                    
                    data = cursor.fetchall()
                    cursor.close()
                    
                    df = pd.DataFrame(data, columns=columns) if data else pd.DataFrame(columns=columns)
                    results.append(df)
                    
                    logger.debug(f"批量查询 {i+1}/{len(queries)} 完成")
                    
                except Exception as e:
                    logger.error(f"批量查询 {i+1} 失败: {e}")
                    results.append(pd.DataFrame(columns=query_config.get('columns', [])))
        
        logger.info(f"批量查询完成，共 {len(queries)} 个查询")
        return results
    
    def execute_stock_data_query(self, 
                                sql: str,
                                db_name: str = 'database',
                                date_columns: List[str] = None) -> pd.DataFrame:
        """
        执行股票数据查询，自动处理常见的股票数据格式
        
        Args:
            sql: SQL查询语句
            db_name: 数据库名称
            date_columns: 日期列名列表
            
        Returns:
            处理后的股票数据DataFrame
        """
        if date_columns is None:
            date_columns = ['tradingday', 'reportday']
        
        # 获取列信息
        with get_connection(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
        
        if not data:
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(data, columns=columns)
        
        # 处理日期列
        for date_col in date_columns:
            if date_col in df.columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d")
                except:
                    logger.warning(f"日期列 {date_col} 转换失败")
        
        # 处理缺失值标记
        df = df.replace(123456789.0, np.nan)
        
        logger.info(f"股票数据查询完成，形状: {df.shape}")
        return df
    
    def get_table_info(self, table_name: str, db_name: str = 'database') -> Dict[str, Any]:
        """
        获取表结构信息
        
        Args:
            table_name: 表名
            db_name: 数据库名称
            
        Returns:
            表信息字典
        """
        info_sql = f"""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        
        try:
            results = self.execute_query(info_sql, db_name)
            columns_info = []
            for row in results:
                columns_info.append({
                    'column_name': row[0],
                    'data_type': row[1],
                    'is_nullable': row[2],
                    'default_value': row[3]
                })
            
            # 获取行数统计
            count_sql = f"SELECT COUNT(*) FROM {table_name}"
            row_count = self.execute_query(count_sql, db_name)[0][0]
            
            return {
                'table_name': table_name,
                'columns': columns_info,
                'row_count': row_count
            }
            
        except Exception as e:
            logger.error(f"获取表 {table_name} 信息失败: {e}")
            return {}
    
    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        avg_time = (self.query_stats['total_time'] / self.query_stats['total_queries'] 
                   if self.query_stats['total_queries'] > 0 else 0)
        
        success_queries = self.query_stats['total_queries'] - self.query_stats['failed_queries']
        success_rate = (success_queries / self.query_stats['total_queries'] * 100 
                       if self.query_stats['total_queries'] > 0 else 0)
        
        return {
            'total_queries': self.query_stats['total_queries'],
            'success_queries': success_queries,
            'failed_queries': self.query_stats['failed_queries'],
            'retried_queries': self.query_stats['retried_queries'],
            'connection_errors': self.query_stats['connection_errors'],
            'timeout_errors': self.query_stats['timeout_errors'],
            'success_rate': success_rate,
            'total_time': self.query_stats['total_time'],
            'average_time': avg_time,
            'retry_rate': (self.query_stats['retried_queries'] / self.query_stats['total_queries'] * 100
                          if self.query_stats['total_queries'] > 0 else 0)
        }
    
    def get_connection_pool_status(self) -> Dict[str, Any]:
        """获取连接池状态"""
        return get_pool_status()
    
    def force_cleanup_connections(self, db_name: str = None):
        """强制清理连接池"""
        try:
            cleanup_idle_connections(db_name, force=True)
            logger.info(f"强制清理连接池完成: {db_name or 'all'}")
        except Exception as e:
            logger.error(f"强制清理连接池失败: {e}")
    
    def reset_stats(self):
        """重置查询统计"""
        self.query_stats = {
            'total_queries': 0,
            'total_time': 0,
            'failed_queries': 0,
            'retried_queries': 0,
            'connection_errors': 0,
            'timeout_errors': 0
        }
        logger.info("查询统计已重置")


# 全局SQL执行器实例
sql_executor = SQLExecutor()


# 便捷函数
def execute_query(sql: str, db_name: str = 'database', params: Optional[Dict] = None, timeout: int = 300) -> List[Tuple]:
    """执行SQL查询的便捷函数"""
    return sql_executor.execute_query(sql, db_name, params, timeout)


def get_query_stats() -> Dict[str, Any]:
    """获取查询统计信息的便捷函数"""
    return sql_executor.get_query_stats()


def get_connection_pool_status() -> Dict[str, Any]:
    """获取连接池状态的便捷函数"""
    return sql_executor.get_connection_pool_status()


def force_cleanup_connections(db_name: str = None):
    """强制清理连接池的便捷函数"""
    return sql_executor.force_cleanup_connections(db_name)


def execute_query_to_dataframe(sql: str, 
                              columns: List[str],
                              db_name: str = 'database',
                              params: Optional[Dict] = None,
                              dtype_mapping: Optional[Dict] = None) -> pd.DataFrame:
    """执行SQL查询并返回DataFrame的便捷函数"""
    return sql_executor.execute_query_to_dataframe(sql, columns, db_name, params, dtype_mapping)


def execute_stock_data_query(sql: str, 
                           db_name: str = 'database',
                           date_columns: List[str] = None) -> pd.DataFrame:
    """执行股票数据查询的便捷函数"""
    return sql_executor.execute_stock_data_query(sql, db_name, date_columns)


def get_table_info(table_name: str, db_name: str = 'database') -> Dict[str, Any]:
    """获取表信息的便捷函数"""
    return sql_executor.get_table_info(table_name, db_name)


if __name__ == "__main__":
    # 测试SQL执行器
    logger.info("测试SQL执行器...")
    
    try:
        # 测试简单查询
        test_sql = "SELECT COUNT(*) FROM day5"
        result = execute_query(test_sql, 'database')
        print(f"[OK] 简单查询测试通过，day5表共有 {result[0][0]} 条记录")
        
        # 测试DataFrame查询
        df_sql = "SELECT TOP 5 code, tradingday, c FROM day5"
        df = execute_stock_data_query(df_sql, 'database')
        print(f"[OK] DataFrame查询测试通过，返回形状: {df.shape}")
        
        # 打印统计信息
        stats = sql_executor.get_query_stats()
        print(f"查询统计: {stats}")
        
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")