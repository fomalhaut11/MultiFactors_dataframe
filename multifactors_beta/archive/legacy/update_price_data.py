#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价格数据更新工具
提供增量更新、状态查询等功能
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

import logging
from datetime import datetime
from data.fetcher.incremental_price_updater import IncrementalPriceUpdater

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def show_status():
    """显示当前数据状态"""
    updater = IncrementalPriceUpdater()
    info = updater.get_update_info()
    
    print("\n" + "=" * 60)
    print("当前数据状态")
    print("=" * 60)
    print(f"Price.pkl文件: {'存在' if info['price_file_exists'] else '不存在'}")
    
    if info['price_file_exists']:
        print(f"文件大小: {info['file_size_mb']:.1f} MB")
        print(f"本地最新日期: {info['local_latest_date'] or 'N/A'}")
    
    print(f"数据库最新日期: {info['db_latest_date'] or 'N/A'}")
    print(f"需要更新: {'是' if info['need_update'] else '否'}")
    
    if info['need_update'] and info['local_latest_date'] and info['db_latest_date']:
        from datetime import datetime
        local_date = datetime.strptime(info['local_latest_date'], '%Y-%m-%d')
        db_date = datetime.strptime(info['db_latest_date'], '%Y-%m-%d')
        gap_days = (db_date - local_date).days
        print(f"数据缺口: {gap_days} 天")
    
    print("=" * 60)


def update_data():
    """执行增量更新"""
    print("\n开始增量更新...")
    
    updater = IncrementalPriceUpdater()
    start_time = datetime.now()
    
    try:
        success = updater.update_price_file()
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            print(f"\n[OK] 增量更新完成！耗时: {duration}")
            
            # 显示更新后状态
            show_status()
            
            # 清理旧备份
            print("\n清理旧备份文件...")
            updater.clean_old_backups()
            
        else:
            print(f"\n[FAIL] 增量更新失败！")
            
    except KeyboardInterrupt:
        print(f"\n[WARN] 用户中断了更新过程")
    except Exception as e:
        print(f"\n[FAIL] 更新过程发生错误: {e}")


def force_update_recent(days: int = 5):
    """强制更新最近几天的数据（用于数据修复）"""
    print(f"\n开始强制更新最近 {days} 天的数据...")
    
    try:
        from datetime import timedelta
        from data.fetcher.chunked_price_fetcher import ChunkedPriceFetcher
        import pandas as pd
        
        updater = IncrementalPriceUpdater()
        
        # 计算开始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        begin_date = int(start_date.strftime('%Y%m%d'))
        end_date_int = int(end_date.strftime('%Y%m%d'))
        
        print(f"获取数据范围: {begin_date} - {end_date_int}")
        
        # 获取最近数据
        fetcher = ChunkedPriceFetcher(chunk_days=30, chunk_stocks=1000)
        new_data = fetcher.fetch_price_data_chunked(
            begin_date=begin_date,
            end_date=end_date_int,
            by_date=True,
            save_intermediate=False
        )
        
        if new_data.empty:
            print("没有获取到新数据")
            return
        
        print(f"获取到 {len(new_data)} 条记录")
        
        # 备份现有文件
        backup_file = updater.backup_price_file()
        
        # 读取现有数据
        if os.path.exists(updater.price_file):
            existing_data = pd.read_pickle(updater.price_file)
            
            # 删除重叠的日期数据
            start_date_only = start_date.date()
            mask = existing_data.index.get_level_values('TradingDates').date < start_date_only
            filtered_data = existing_data[mask]
            
            print(f"保留原有数据: {len(filtered_data)} 条")
            print(f"替换最近数据: {len(new_data)} 条")
            
            # 合并数据
            combined_data = updater.merge_data(filtered_data, new_data)
            
            # 保存更新后的数据
            combined_data.to_pickle(updater.price_file)
            
            print(f"[OK] 强制更新完成！最终数据: {combined_data.shape}")
            
        else:
            print("[FAIL] Price.pkl文件不存在，无法执行强制更新")
            
    except Exception as e:
        print(f"[FAIL] 强制更新失败: {e}")
        import traceback
        traceback.print_exc()


def clean_backups():
    """清理备份文件"""
    print("\n清理备份文件...")
    
    updater = IncrementalPriceUpdater()
    
    days = input("保留最近几天的备份？(默认7天): ").strip()
    try:
        keep_days = int(days) if days else 7
    except:
        keep_days = 7
    
    updater.clean_old_backups(keep_days)
    print("[OK] 备份文件清理完成")


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 60)
        print("Price.pkl 数据更新工具")
        print("=" * 60)
        print("1. 查看数据状态")
        print("2. 执行增量更新")
        print("3. 强制更新最近数据（数据修复）")
        print("4. 清理备份文件")
        print("5. 退出")
        print("=" * 60)
        
        try:
            choice = input("请选择操作 (1-5): ").strip()
            
            if choice == "1":
                show_status()
                
            elif choice == "2":
                update_data()
                
            elif choice == "3":
                days_input = input("强制更新最近几天的数据？(默认5天): ").strip()
                try:
                    days = int(days_input) if days_input else 5
                    if days > 30:
                        print("[WARN] 天数过多，限制为30天")
                        days = 30
                except:
                    days = 5
                
                confirm = input(f"确认强制更新最近 {days} 天的数据？(y/N): ").strip().lower()
                if confirm == 'y':
                    force_update_recent(days)
                else:
                    print("操作已取消")
                
            elif choice == "4":
                clean_backups()
                
            elif choice == "5":
                print("退出程序")
                break
                
            else:
                print("[FAIL] 无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n[FAIL] 操作失败: {e}")


if __name__ == "__main__":
    main()