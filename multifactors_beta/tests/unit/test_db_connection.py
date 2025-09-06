#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库连接
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import test_connection, get_pool_status
from config import get_config, config_managernfig

def test_all_databases():
    """测试所有数据库连接"""
    print("=" * 60)
    print("测试数据库连接")
    print("=" * 60)
    
    # 测试主数据库
    print("1. 测试主数据库连接...")
    if test_connection('database'):
        print("[OK] 主数据库连接成功")
        config = get_database_config('database')
        print(f"   主机: {config['host']}")
        print(f"   数据库: {config['database']}")
    else:
        print("[FAIL] 主数据库连接失败")
    
    print()
    
    # 测试分钟数据库
    print("2. 测试分钟数据库连接...")
    if test_connection('min_database'):
        print("[OK] 分钟数据库连接成功")
    else:
        print("[FAIL] 分钟数据库连接失败")
    
    print()
    
    # 测试聚宽数据库
    print("3. 测试聚宽数据库连接...")
    if test_connection('jq_database'):
        print("[OK] 聚宽数据库连接成功")
    else:
        print("[FAIL] 聚宽数据库连接失败")
    
    print()
    
    # 测试Wind数据库
    print("4. 测试Wind数据库连接...")
    if test_connection('wind_database'):
        print("[OK] Wind数据库连接成功")
    else:
        print("[FAIL] Wind数据库连接失败")
    
    print()
    print("=" * 60)
    print("连接池状态:")
    status = get_pool_status()
    for pool_name, pool_info in status.items():
        print(f"{pool_name}: {pool_info}")
    print("=" * 60)

if __name__ == "__main__":
    test_all_databases()