#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库连接管理器
提供统一的数据库连接管理功能，支持连接池和多数据库
"""

import pymssql
import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from core.config_manager import get_database_config

logger = logging.getLogger(__name__)


class ConnectionInfo:
    """连接信息类"""
    def __init__(self, connection, created_at: datetime = None):
        self.connection = connection
        self.created_at = created_at or datetime.now()
        self.last_used = datetime.now()
        self.is_healthy = True
        self.use_count = 0
    
    def update_last_used(self):
        """更新最后使用时间"""
        self.last_used = datetime.now()
        self.use_count += 1
    
    def is_expired(self, max_idle_time: int = 3600) -> bool:
        """检查连接是否已过期（默认1小时）"""
        return (datetime.now() - self.last_used).total_seconds() > max_idle_time
    
    def is_too_old(self, max_lifetime: int = 28800) -> bool:
        """检查连接是否太老（默认8小时）"""
        return (datetime.now() - self.created_at).total_seconds() > max_lifetime


class ConnectionPool:
    """数据库连接池"""
    
    def __init__(self, db_config: Dict[str, Any], max_connections: int = 10, 
                 max_idle_time: int = 3600, max_lifetime: int = 28800):
        self.db_config = db_config
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time  # 最大空闲时间（秒）
        self.max_lifetime = max_lifetime    # 连接最大生存时间（秒）
        self.connections = {}  # 改为字典存储ConnectionInfo
        self.used_connections = set()
        self.lock = threading.Lock()
        self.last_cleanup = datetime.now()
        self.cleanup_interval = 300  # 清理间隔（5分钟）
        
    def _create_connection(self):
        """创建新的数据库连接"""
        try:
            conn = pymssql.connect(
                server=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                timeout=30,
                login_timeout=30
            )
            logger.debug(f"创建新连接到数据库: {self.db_config['database']}")
            return conn
        except Exception as e:
            logger.error(f"创建数据库连接失败: {e}")
            raise
    
    def get_connection(self):
        """从连接池获取连接"""
        with self.lock:
            # 定期清理过期连接
            self._cleanup_expired_connections()
            
            # 尝试从空闲连接中获取
            for conn_id, conn_info in list(self.connections.items()):
                if conn_info.connection not in self.used_connections:
                    if self._is_connection_healthy(conn_info):
                        self.used_connections.add(conn_info.connection)
                        conn_info.update_last_used()
                        logger.debug(f"从连接池获取连接，ID: {conn_id}")
                        return conn_info.connection
                    else:
                        # 连接不健康，移除
                        self._remove_connection(conn_id)
            
            # 如果没有可用连接且未达到最大连接数，创建新连接
            if len(self.connections) < self.max_connections:
                conn = self._create_connection()
                conn_id = id(conn)
                conn_info = ConnectionInfo(conn)
                self.connections[conn_id] = conn_info
                self.used_connections.add(conn)
                conn_info.update_last_used()
                logger.debug(f"创建新连接，ID: {conn_id}")
                return conn
            
            # 连接池已满，等待可用连接
            logger.warning("连接池已满，等待可用连接")
            raise Exception("数据库连接池已满")
    
    def return_connection(self, conn):
        """归还连接到连接池"""
        with self.lock:
            if conn in self.used_connections:
                self.used_connections.remove(conn)
                # 更新连接信息
                conn_id = id(conn)
                if conn_id in self.connections:
                    self.connections[conn_id].update_last_used()
                logger.debug(f"连接已归还到连接池，ID: {conn_id}")
    
    def _is_connection_healthy(self, conn_info: ConnectionInfo) -> bool:
        """检查连接是否健康"""
        try:
            # 检查连接是否过期或太老
            if conn_info.is_expired(self.max_idle_time) or conn_info.is_too_old(self.max_lifetime):
                logger.debug(f"连接已过期或太老，创建时间: {conn_info.created_at}, 最后使用: {conn_info.last_used}")
                return False
            
            # 测试连接是否有效
            cursor = conn_info.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn_info.is_healthy = True
            return True
        except Exception as e:
            logger.debug(f"连接健康检查失败: {e}")
            conn_info.is_healthy = False
            return False
    
    def _remove_connection(self, conn_id):
        """移除连接"""
        if conn_id in self.connections:
            conn_info = self.connections[conn_id]
            try:
                conn_info.connection.close()
            except:
                pass
            if conn_info.connection in self.used_connections:
                self.used_connections.remove(conn_info.connection)
            del self.connections[conn_id]
            logger.debug(f"移除连接，ID: {conn_id}")
    
    def _cleanup_expired_connections(self):
        """清理过期连接"""
        if (datetime.now() - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return
        
        expired_connections = []
        for conn_id, conn_info in self.connections.items():
            if conn_info.connection not in self.used_connections:
                if conn_info.is_expired(self.max_idle_time) or conn_info.is_too_old(self.max_lifetime):
                    expired_connections.append(conn_id)
        
        for conn_id in expired_connections:
            self._remove_connection(conn_id)
            logger.info(f"清理过期连接，ID: {conn_id}")
        
        self.last_cleanup = datetime.now()
        if expired_connections:
            logger.info(f"清理了 {len(expired_connections)} 个过期连接")
    
    def cleanup_idle_connections(self, force: bool = False):
        """手动清理空闲连接"""
        with self.lock:
            if force:
                # 强制清理所有空闲连接
                idle_connections = []
                for conn_id, conn_info in self.connections.items():
                    if conn_info.connection not in self.used_connections:
                        idle_connections.append(conn_id)
                
                for conn_id in idle_connections:
                    self._remove_connection(conn_id)
                
                logger.info(f"强制清理了 {len(idle_connections)} 个空闲连接")
            else:
                # 正常清理过期连接
                self._cleanup_expired_connections()
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn_id, conn_info in self.connections.items():
                try:
                    conn_info.connection.close()
                except:
                    pass
            self.connections.clear()
            self.used_connections.clear()
            logger.info("所有数据库连接已关闭")
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细的连接池状态"""
        with self.lock:
            total_connections = len(self.connections)
            used_connections = len(self.used_connections)
            available_connections = total_connections - used_connections
            
            # 统计连接状态
            healthy_connections = 0
            expired_connections = 0
            old_connections = 0
            
            for conn_info in self.connections.values():
                if conn_info.is_healthy:
                    healthy_connections += 1
                if conn_info.is_expired(self.max_idle_time):
                    expired_connections += 1
                if conn_info.is_too_old(self.max_lifetime):
                    old_connections += 1
            
            return {
                'total_connections': total_connections,
                'used_connections': used_connections,
                'available_connections': available_connections,
                'max_connections': self.max_connections,
                'healthy_connections': healthy_connections,
                'expired_connections': expired_connections,
                'old_connections': old_connections,
                'max_idle_time': self.max_idle_time,
                'max_lifetime': self.max_lifetime,
                'last_cleanup': self.last_cleanup.isoformat()
            }


class DatabaseManager:
    """数据库管理器"""
    
    _instance = None
    _pools = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._pools = {}
            self._initialized = True
            logger.info("数据库管理器初始化完成")
    
    def _get_pool(self, db_name: str = 'database') -> ConnectionPool:
        """获取指定数据库的连接池"""
        if db_name not in self._pools:
            db_config = get_database_config(db_name)
            self._pools[db_name] = ConnectionPool(db_config)
            logger.info(f"创建数据库连接池: {db_name}")
        return self._pools[db_name]
    
    @contextmanager
    def get_connection(self, db_name: str = 'database'):
        """获取数据库连接的上下文管理器"""
        pool = self._get_pool(db_name)
        conn = None
        try:
            conn = pool.get_connection()
            yield conn
        except Exception as e:
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            if conn:
                pool.return_connection(conn)
    
    def test_connection(self, db_name: str = 'database') -> bool:
        """测试数据库连接"""
        try:
            with self.get_connection(db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                logger.info(f"数据库连接测试成功: {db_name}")
                return True
        except Exception as e:
            logger.error(f"数据库连接测试失败 {db_name}: {e}")
            return False
    
    def close_all_pools(self):
        """关闭所有连接池"""
        for pool_name, pool in self._pools.items():
            pool.close_all()
            logger.info(f"关闭连接池: {pool_name}")
        self._pools.clear()
    
    def get_pool_status(self) -> Dict[str, Dict[str, Any]]:
        """获取连接池状态"""
        status = {}
        for pool_name, pool in self._pools.items():
            status[pool_name] = pool.get_detailed_status()
        return status
    
    def cleanup_idle_connections(self, db_name: str = None, force: bool = False):
        """清理空闲连接"""
        if db_name:
            if db_name in self._pools:
                self._pools[db_name].cleanup_idle_connections(force)
                logger.info(f"清理数据库 {db_name} 的空闲连接")
        else:
            for pool_name, pool in self._pools.items():
                pool.cleanup_idle_connections(force)
                logger.info(f"清理数据库 {pool_name} 的空闲连接")
    
    def health_check(self) -> Dict[str, bool]:
        """执行所有数据库的健康检查"""
        results = {}
        for pool_name in self._pools:
            results[pool_name] = self.test_connection(pool_name)
        return results


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 便捷函数
def get_connection(db_name: str = 'database'):
    """获取数据库连接的便捷函数"""
    return db_manager.get_connection(db_name)


def test_connection(db_name: str = 'database') -> bool:
    """测试数据库连接的便捷函数"""
    return db_manager.test_connection(db_name)


def get_pool_status() -> Dict[str, Dict[str, Any]]:
    """获取连接池状态的便捷函数"""
    return db_manager.get_pool_status()


def cleanup_idle_connections(db_name: str = None, force: bool = False):
    """清理空闲连接的便捷函数"""
    return db_manager.cleanup_idle_connections(db_name, force)


def health_check() -> Dict[str, bool]:
    """执行健康检查的便捷函数"""
    return db_manager.health_check()


if __name__ == "__main__":
    # 测试数据库连接管理器
    logger.info("测试数据库连接管理器...")
    
    # 测试主数据库连接
    if test_connection('database'):
        print("[OK] 主数据库连接测试通过")
    else:
        print("[FAIL] 主数据库连接测试失败")
    
    # 测试分钟数据库连接
    if test_connection('min_database'):
        print("[OK] 分钟数据库连接测试通过")
    else:
        print("[FAIL] 分钟数据库连接测试失败")
    
    # 打印连接池状态
    status = get_pool_status()
    print(f"连接池状态: {status}")