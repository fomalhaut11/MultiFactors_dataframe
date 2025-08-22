#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库连接监控工具
提供数据库连接池的实时监控和管理功能
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


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)


def format_datetime(dt_str: str) -> str:
    """格式化日期时间字符串"""
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def show_pool_status():
    """显示连接池状态"""
    print_separator("连接池状态")
    
    status = get_pool_status()
    if not status:
        print("暂无活跃的连接池")
        return
    
    for pool_name, pool_info in status.items():
        print(f"\n数据库: {pool_name}")
        print(f"  总连接数: {pool_info['total_connections']}")
        print(f"  使用中: {pool_info['used_connections']}")
        print(f"  可用: {pool_info['available_connections']}")
        print(f"  最大连接数: {pool_info['max_connections']}")
        print(f"  健康连接: {pool_info['healthy_connections']}")
        print(f"  过期连接: {pool_info['expired_connections']}")
        print(f"  老旧连接: {pool_info['old_connections']}")
        print(f"  最大空闲时间: {pool_info['max_idle_time']}秒")
        print(f"  最大生存时间: {pool_info['max_lifetime']}秒")
        print(f"  最后清理时间: {format_datetime(pool_info['last_cleanup'])}")
        
        # 连接池使用率
        if pool_info['max_connections'] > 0:
            usage_rate = pool_info['total_connections'] / pool_info['max_connections'] * 100
            print(f"  使用率: {usage_rate:.1f}%")
        
        # 健康状态
        if pool_info['total_connections'] > 0:
            health_rate = pool_info['healthy_connections'] / pool_info['total_connections'] * 100
            health_status = "[OK]" if health_rate >= 80 else "[WARN]" if health_rate >= 50 else "[FAIL]"
            print(f"  健康状态: {health_status} {health_rate:.1f}%")


def perform_health_check():
    """执行健康检查"""
    print_separator("数据库健康检查")
    
    results = health_check()
    if not results:
        print("暂无数据库连接")
        return
    
    all_healthy = True
    for db_name, is_healthy in results.items():
        status = "[OK]" if is_healthy else "[FAIL]"
        print(f"  {db_name}: {status}")
        if not is_healthy:
            all_healthy = False
    
    overall_status = "[OK]" if all_healthy else "[FAIL]"
    print(f"\n整体状态: {overall_status}")


def cleanup_connections():
    """清理空闲连接"""
    print_separator("清理空闲连接")
    
    print("选择清理模式:")
    print("1. 清理过期连接（推荐）")
    print("2. 清理所有空闲连接（强制）")
    print("3. 返回主菜单")
    
    try:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == '1':
            cleanup_idle_connections(force=False)
            print("[OK] 过期连接清理完成")
        elif choice == '2':
            confirm = input("确认要清理所有空闲连接吗？(y/N): ").strip().lower()
            if confirm == 'y':
                cleanup_idle_connections(force=True)
                print("[OK] 所有空闲连接清理完成")
            else:
                print("操作已取消")
        elif choice == '3':
            return
        else:
            print("[FAIL] 无效选择")
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"[FAIL] 清理失败: {e}")


def test_database_connections():
    """测试数据库连接"""
    print_separator("测试数据库连接")
    
    # 预定义的数据库列表
    databases = ['database', 'stock_min1', 'jqdata', 'Wind']
    
    print("测试各数据库连接...")
    results = {}
    
    for db_name in databases:
        try:
            print(f"测试 {db_name}...", end=" ")
            is_connected = test_connection(db_name)
            results[db_name] = is_connected
            status = "[OK]" if is_connected else "[FAIL]"
            print(status)
        except Exception as e:
            results[db_name] = False
            print(f"[FAIL] {e}")
    
    # 汇总结果
    successful = sum(1 for result in results.values() if result)
    total = len(results)
    print(f"\n连接测试完成: {successful}/{total} 成功")
    
    if successful < total:
        print("\n失败的连接:")
        for db_name, result in results.items():
            if not result:
                print(f"  - {db_name}")


def monitor_realtime():
    """实时监控模式"""
    print_separator("实时监控模式")
    print("按 Ctrl+C 退出监控")
    
    try:
        while True:
            # 清屏（Windows和Linux兼容）
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"数据库连接监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            show_pool_status()
            
            print("\n按 Ctrl+C 退出监控...")
            time.sleep(5)  # 每5秒刷新一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")


def export_status_report():
    """导出状态报告"""
    print_separator("导出状态报告")
    
    try:
        # 收集所有状态信息
        report = {
            'timestamp': datetime.now().isoformat(),
            'pool_status': get_pool_status(),
            'health_check': health_check()
        }
        
        # 生成报告文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"db_status_report_{timestamp}.json"
        
        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 状态报告已导出到: {filename}")
        
        # 显示摘要
        pool_count = len(report['pool_status'])
        healthy_count = sum(1 for h in report['health_check'].values() if h)
        total_connections = sum(pool['total_connections'] for pool in report['pool_status'].values())
        
        print(f"\n报告摘要:")
        print(f"  连接池数量: {pool_count}")
        print(f"  健康数据库: {healthy_count}/{len(report['health_check'])}")
        print(f"  总连接数: {total_connections}")
        
    except Exception as e:
        print(f"[FAIL] 导出失败: {e}")


def main_menu():
    """主菜单"""
    while True:
        print_separator("数据库连接监控工具")
        print("1. 查看连接池状态")
        print("2. 执行健康检查")
        print("3. 清理空闲连接")
        print("4. 测试数据库连接")
        print("5. 实时监控模式")
        print("6. 导出状态报告")
        print("0. 退出")
        
        try:
            choice = input("\n请选择功能 (0-6): ").strip()
            
            if choice == '0':
                print("再见！")
                break
            elif choice == '1':
                show_pool_status()
            elif choice == '2':
                perform_health_check()
            elif choice == '3':
                cleanup_connections()
            elif choice == '4':
                test_database_connections()
            elif choice == '5':
                monitor_realtime()
            elif choice == '6':
                export_status_report()
            else:
                print("[FAIL] 无效选择，请重试")
                
            if choice != '0':
                input("\n按回车键继续...")
                
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n[FAIL] 操作失败: {e}")
            input("按回车键继续...")


if __name__ == "__main__":
    main_menu()