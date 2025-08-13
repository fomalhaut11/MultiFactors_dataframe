#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式数据更新工具
提供用户友好的菜单界面来管理各种数据的更新
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

from datetime import datetime
from scheduled_data_updater import ScheduledDataUpdater


def print_header():
    """打印程序头部信息"""
    print("=" * 70)
    print("多因子量化系统 - 交互式数据更新工具")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_menu():
    """打印主菜单"""
    print("\n请选择操作:")
    print("1. 价格数据增量更新 (Price.pkl)")
    print("2. 财务数据更新 (预留功能)")
    print("3. 行业数据更新 (预留功能)")
    print("4. 全部数据更新")
    print("5. 数据健康检查")
    print("6. 价格数据强制更新")
    print("7. 查看数据状态")
    print("0. 退出")


def handle_price_data_update(updater: ScheduledDataUpdater):
    """处理价格数据更新"""
    print("\n正在执行价格数据增量更新...")
    print("=" * 50)
    
    try:
        result = updater.run_price_data_update(force=False)
        
        if result.success:
            print(f"\n[OK] {result.message}")
            if result.duration:
                print(f"耗时: {result.duration:.1f}秒")
        else:
            print(f"\n[FAIL] {result.message}")
            
    except KeyboardInterrupt:
        print(f"\n[WARN] 用户中断了更新过程")
    except Exception as e:
        print(f"\n[FAIL] 更新过程发生错误: {e}")


def handle_financial_data_update(updater: ScheduledDataUpdater):
    """处理财务数据更新"""
    print("\n财务数据更新功能尚未实现")
    print("该功能将在未来版本中提供，用于更新：")
    print("- 资产负债表数据")
    print("- 利润表数据") 
    print("- 现金流量表数据")
    print("- 财务指标数据")


def handle_industry_data_update(updater: ScheduledDataUpdater):
    """处理行业数据更新"""
    print("\n行业数据更新功能尚未实现")
    print("该功能将在未来版本中提供，用于更新：")
    print("- 行业分类数据")
    print("- 概念板块数据")
    print("- 行业指数数据")
    print("- ST股票列表")


def handle_all_data_update(updater: ScheduledDataUpdater):
    """处理全部数据更新"""
    print("\n正在执行全部数据更新...")
    print("=" * 50)
    
    try:
        results = updater.run_all_updates(force=False)
        
        print(f"\n更新结果汇总:")
        success_count = 0
        
        for result in results:
            status = "[OK]" if result.success else "[FAIL]"
            print(f"  {result.data_type}: {status} {result.message}")
            if result.duration:
                print(f"    耗时: {result.duration:.1f}秒")
            
            if result.success:
                success_count += 1
        
        print(f"\n总计: {success_count}/{len(results)} 个数据类型更新成功")
        
    except KeyboardInterrupt:
        print(f"\n[WARN] 用户中断了更新过程")
    except Exception as e:
        print(f"\n[FAIL] 批量更新过程发生错误: {e}")


def handle_health_check(updater: ScheduledDataUpdater):
    """处理健康检查"""
    print("\n正在执行数据健康检查...")
    print("=" * 50)
    
    try:
        health_results = updater.run_health_check()
        
        print("\n健康检查结果:")
        all_healthy = True
        
        for data_type, health_status in health_results.items():
            status_color = {
                'healthy': '[OK]',
                'warning': '[WARN]',
                'error': '[FAIL]',
                'not_implemented': '[INFO]'
            }.get(health_status['status'], '[UNKNOWN]')
            
            print(f"\n{data_type.upper()} 数据: {status_color}")
            print(f"  状态消息: {health_status.get('message', 'N/A')}")
            
            if 'file_exists' in health_status:
                file_status = "存在" if health_status['file_exists'] else "不存在"
                print(f"  数据文件: {file_status}")
            
            if 'file_size_mb' in health_status:
                print(f"  文件大小: {health_status['file_size_mb']:.1f} MB")
            
            if 'local_latest_date' in health_status and health_status['local_latest_date']:
                print(f"  本地最新日期: {health_status['local_latest_date']}")
            
            if 'db_latest_date' in health_status and health_status['db_latest_date']:
                print(f"  数据库最新日期: {health_status['db_latest_date']}")
            
            if 'data_gap_days' in health_status:
                gap = health_status['data_gap_days']
                if gap > 0:
                    print(f"  数据差距: {gap} 天")
                else:
                    print(f"  数据差距: 无差距")
            
            if health_status['status'] not in ['healthy', 'not_implemented']:
                all_healthy = False
        
        overall_status = "[OK]" if all_healthy else "[WARN]"
        print(f"\n整体健康状态: {overall_status}")
        
    except Exception as e:
        print(f"\n[FAIL] 健康检查失败: {e}")


def handle_force_price_update(updater: ScheduledDataUpdater):
    """处理强制价格数据更新"""
    print("\n警告: 强制更新将忽略时间检查，立即执行数据更新")
    confirm = input("确认执行强制更新吗? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("操作已取消")
        return
    
    print("\n正在执行价格数据强制更新...")
    print("=" * 50)
    
    try:
        result = updater.run_price_data_update(force=True)
        
        if result.success:
            print(f"\n[OK] {result.message}")
            if result.duration:
                print(f"耗时: {result.duration:.1f}秒")
        else:
            print(f"\n[FAIL] {result.message}")
            
    except Exception as e:
        print(f"\n[FAIL] 强制更新失败: {e}")


def handle_data_status(updater: ScheduledDataUpdater):
    """处理数据状态查看"""
    print("\n数据状态概览:")
    print("=" * 50)
    
    try:
        # 获取价格数据详细信息
        price_updater = updater.updaters['price']
        info = price_updater.get_update_info()
        
        print("\n价格数据 (Price.pkl):")
        print(f"  文件路径: {info.get('price_file_path', 'N/A')}")
        print(f"  文件存在: {'是' if info['price_file_exists'] else '否'}")
        
        if info['price_file_exists']:
            print(f"  文件大小: {info['file_size_mb']:.1f} MB")
            print(f"  本地最新日期: {info['local_latest_date'] or 'N/A'}")
            print(f"  数据库最新日期: {info['db_latest_date'] or 'N/A'}")
            print(f"  需要更新: {'是' if info['need_update'] else '否'}")
        
        # 显示其他数据类型的基本状态
        print(f"\n财务数据: [未实现]")
        print(f"行业数据: [未实现]")
        
    except Exception as e:
        print(f"[FAIL] 获取数据状态失败: {e}")


def main():
    """主函数"""
    print_header()
    
    # 初始化更新器（只包含已实现的数据类型）
    updater = ScheduledDataUpdater(data_types=['price'])
    
    while True:
        try:
            print_menu()
            choice = input("\n请选择 (0-7): ").strip()
            
            if choice == '0':
                print("\n再见！")
                break
            elif choice == '1':
                handle_price_data_update(updater)
            elif choice == '2':
                handle_financial_data_update(updater)
            elif choice == '3':
                handle_industry_data_update(updater)
            elif choice == '4':
                handle_all_data_update(updater)
            elif choice == '5':
                handle_health_check(updater)
            elif choice == '6':
                handle_force_price_update(updater)
            elif choice == '7':
                handle_data_status(updater)
            else:
                print("[FAIL] 无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n[FAIL] 操作失败: {e}")
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()