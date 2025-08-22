#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序退出处理器
提供程序退出时的资源清理功能，确保数据库连接等资源正确释放
"""

import atexit
import signal
import sys
import logging
import threading
from typing import List, Callable
from core.database.connection_manager import db_manager

logger = logging.getLogger(__name__)


class ExitHandler:
    """程序退出处理器"""
    
    def __init__(self):
        self._cleanup_functions: List[Callable] = []
        self._exit_in_progress = False
        self._lock = threading.Lock()
        self._registered = False
    
    def register_cleanup(self, cleanup_func: Callable, description: str = ""):
        """注册清理函数"""
        with self._lock:
            self._cleanup_functions.append((cleanup_func, description))
            logger.debug(f"注册清理函数: {description or cleanup_func.__name__}")
    
    def _execute_cleanup(self, reason: str = "程序退出"):
        """执行所有清理函数"""
        with self._lock:
            if self._exit_in_progress:
                return
            self._exit_in_progress = True
        
        logger.info(f"开始资源清理 - {reason}")
        
        # 首先清理数据库连接池
        try:
            db_manager.close_all_pools()
            logger.info("数据库连接池已清理")
        except Exception as e:
            logger.error(f"清理数据库连接池失败: {e}")
        
        # 执行注册的清理函数
        for cleanup_func, description in reversed(self._cleanup_functions):  # 逆序执行
            try:
                cleanup_func()
                logger.debug(f"清理完成: {description or cleanup_func.__name__}")
            except Exception as e:
                logger.error(f"清理函数执行失败 {description}: {e}")
        
        logger.info("资源清理完成")
    
    def _signal_handler(self, signum: int, frame):
        """系统信号处理器"""
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT (Ctrl+C)"
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")
        
        logger.info(f"接收到系统信号: {signal_name}")
        self._execute_cleanup(f"系统信号 {signal_name}")
        
        # 恢复默认信号处理器并重新发送信号
        signal.signal(signum, signal.SIG_DFL)
        sys.exit(128 + signum)
    
    def _atexit_handler(self):
        """程序正常退出处理器"""
        self._execute_cleanup("程序正常退出")
    
    def install(self):
        """安装退出处理器"""
        if self._registered:
            return
        
        try:
            # 注册atexit处理器
            atexit.register(self._atexit_handler)
            
            # 注册信号处理器（仅在支持的平台上）
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, self._signal_handler)
            
            self._registered = True
            logger.info("退出处理器已安装")
            
        except Exception as e:
            logger.error(f"安装退出处理器失败: {e}")
    
    def force_cleanup(self):
        """强制执行清理（用于测试或手动清理）"""
        self._execute_cleanup("手动强制清理")


# 全局退出处理器实例
exit_handler = ExitHandler()


def install_exit_handler():
    """安装退出处理器（便捷函数）"""
    exit_handler.install()


def register_cleanup_function(cleanup_func: Callable, description: str = ""):
    """注册清理函数（便捷函数）"""
    exit_handler.register_cleanup(cleanup_func, description)


def force_cleanup():
    """强制执行清理（便捷函数）"""
    exit_handler.force_cleanup()


# 一些常用的清理函数
def cleanup_temp_files(temp_dir: str = "temp"):
    """清理临时文件"""
    import os
    import shutil
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"临时目录已清理: {temp_dir}")
    except Exception as e:
        logger.error(f"清理临时目录失败: {e}")


def cleanup_pickle_temp_files():
    """清理临时pickle文件"""
    import os
    import glob
    try:
        temp_files = glob.glob("temp_*.pkl") + glob.glob("*_temp.pkl")
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                logger.debug(f"删除临时文件: {file}")
        if temp_files:
            logger.info(f"清理了 {len(temp_files)} 个临时pickle文件")
    except Exception as e:
        logger.error(f"清理临时pickle文件失败: {e}")


if __name__ == "__main__":
    # 测试退出处理器
    import time
    
    # 安装处理器
    install_exit_handler()
    
    # 注册一些测试清理函数
    register_cleanup_function(lambda: print("清理函数1执行"), "测试清理函数1")
    register_cleanup_function(lambda: print("清理函数2执行"), "测试清理函数2")
    
    print("退出处理器测试")
    print("按 Ctrl+C 测试信号处理")
    print("程序将在5秒后正常退出测试atexit处理")
    
    try:
        time.sleep(5)
        print("正常退出...")
    except KeyboardInterrupt:
        print("接收到中断信号...")
        sys.exit(1)