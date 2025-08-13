#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库监控和连接管理功能
"""

import sys
import os

# 配置控制台编码（Windows兼容）
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() not in ['utf-8', 'utf8']:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'encoding') and sys.stderr.encoding.lower() not in ['utf-8', 'utf8']:
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
from datetime import datetime
from core.database.connection_manager import (
    db_manager, get_pool_status, cleanup_idle_connections, 
    health_check, test_connection
)
from core.database.sql_executor import (
    sql_executor, get_query_stats, get_connection_pool_status,
    force_cleanup_connections, execute_query
)


def test_connection_monitoring():
    """测试连接状态监控"""
    print("=" * 60)
    print("测试连接状态监控")
    print("=" * 60)
    
    # 1. 测试数据库连接
    print("\n1. 测试数据库连接...")
    databases = ['database', 'stock_min1', 'jqdata', 'Wind']
    
    for db_name in databases:
        try:
            result = test_connection(db_name)
            status = "[OK]" if result else "[FAIL]"
            print(f"  {db_name}: {status}")
        except Exception as e:
            print(f"  {db_name}: [ERROR] {e}")
    
    # 2. 显示连接池状态
    print("\n2. 连接池状态...")
    pool_status = get_pool_status()
    for pool_name, status in pool_status.items():
        print(f"  {pool_name}:")
        print(f"    总连接: {status['total_connections']}, "
              f"使用中: {status['used_connections']}, "
              f"可用: {status['available_connections']}")
        print(f"    健康: {status['healthy_connections']}, "
              f"过期: {status['expired_connections']}, "
              f"老旧: {status['old_connections']}")
    
    # 3. 执行一些查询来测试重试机制
    print("\n3. 测试查询和重试机制...")
    try:
        # 执行正常查询
        result = execute_query("SELECT COUNT(*) FROM day5", 'database')
        print(f"  正常查询: [OK] 返回 {result[0][0]} 条记录")
        
        # 执行会引起小延迟的查询
        result = execute_query("SELECT TOP 1000 * FROM day5", 'database')
        print(f"  大数据查询: [OK] 返回 {len(result)} 条记录")
        
    except Exception as e:
        print(f"  查询测试: [FAIL] {e}")
    
    # 4. 显示查询统计
    print("\n4. 查询统计...")
    stats = get_query_stats()
    print(f"  总查询数: {stats['total_queries']}")
    print(f"  成功查询: {stats['success_queries']}")
    print(f"  失败查询: {stats['failed_queries']}")
    print(f"  重试查询: {stats['retried_queries']}")
    print(f"  连接错误: {stats['connection_errors']}")
    print(f"  超时错误: {stats['timeout_errors']}")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    print(f"  平均耗时: {stats['average_time']:.2f}秒")
    print(f"  重试率: {stats['retry_rate']:.1f}%")


def test_connection_cleanup():
    """测试连接清理功能"""
    print("\n\n" + "=" * 60)
    print("测试连接清理功能")
    print("=" * 60)
    
    # 1. 显示清理前状态
    print("\n1. 清理前状态...")
    pool_status = get_pool_status()
    total_before = sum(status['total_connections'] for status in pool_status.values())
    available_before = sum(status['available_connections'] for status in pool_status.values())
    print(f"  总连接数: {total_before}, 可用连接: {available_before}")
    
    # 2. 执行正常清理（清理过期连接）
    print("\n2. 执行正常清理...")
    try:
        cleanup_idle_connections(force=False)
        print("  [OK] 正常清理完成")
    except Exception as e:
        print(f"  [FAIL] 正常清理失败: {e}")
    
    # 3. 显示清理后状态
    print("\n3. 清理后状态...")
    pool_status = get_pool_status()
    total_after = sum(status['total_connections'] for status in pool_status.values())
    available_after = sum(status['available_connections'] for status in pool_status.values())
    print(f"  总连接数: {total_after}, 可用连接: {available_after}")
    print(f"  清理了 {total_before - total_after} 个连接")
    
    # 4. 测试强制清理
    print("\n4. 测试强制清理...")
    try:
        force_cleanup_connections()
        print("  [OK] 强制清理完成")
    except Exception as e:
        print(f"  [FAIL] 强制清理失败: {e}")
    
    # 5. 显示最终状态
    print("\n5. 最终状态...")
    pool_status = get_pool_status()
    total_final = sum(status['total_connections'] for status in pool_status.values())
    available_final = sum(status['available_connections'] for status in pool_status.values())
    print(f"  总连接数: {total_final}, 可用连接: {available_final}")


def test_health_monitoring():
    """测试健康监控"""
    print("\n\n" + "=" * 60)
    print("测试健康监控")
    print("=" * 60)
    
    # 1. 执行健康检查
    print("\n1. 执行健康检查...")
    try:
        health_results = health_check()
        all_healthy = all(health_results.values())
        
        for db_name, is_healthy in health_results.items():
            status = "[OK]" if is_healthy else "[FAIL]"
            print(f"  {db_name}: {status}")
        
        overall_status = "[OK]" if all_healthy else "[FAIL]"
        print(f"\n  整体健康状态: {overall_status}")
        
    except Exception as e:
        print(f"  [FAIL] 健康检查失败: {e}")
    
    # 2. 生成监控报告
    print("\n2. 生成监控报告...")
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'pool_status': get_pool_status(),
            'health_check': health_check(),
            'query_stats': get_query_stats()
        }
        
        print("  监控报告摘要:")
        print(f"    连接池数量: {len(report['pool_status'])}")
        print(f"    健康数据库: {sum(1 for h in report['health_check'].values() if h)}/{len(report['health_check'])}")
        print(f"    总连接数: {sum(pool['total_connections'] for pool in report['pool_status'].values())}")
        print(f"    查询成功率: {report['query_stats']['success_rate']:.1f}%")
        
        # 保存报告到文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_db_monitor_report_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"    报告已保存到: {filename}")
        
    except Exception as e:
        print(f"  [FAIL] 生成报告失败: {e}")


def main():
    """主测试函数"""
    print("数据库监控和连接管理功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 测试连接监控
        test_connection_monitoring()
        
        # 等待一下让连接池有一些状态变化
        print(f"\n等待5秒让连接池状态变化...")
        time.sleep(5)
        
        # 测试连接清理
        test_connection_cleanup()
        
        # 测试健康监控
        test_health_monitoring()
        
        print("\n\n" + "=" * 60)
        print("所有测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] 测试过程发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()