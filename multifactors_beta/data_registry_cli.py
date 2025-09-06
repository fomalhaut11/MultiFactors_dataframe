#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据注册器命令行工具

提供便捷的数据查询和管理功能

使用方式:
    python data_registry_cli.py --summary        # 显示数据摘要  
    python data_registry_cli.py --list          # 列出所有数据集
    python data_registry_cli.py --info price_data    # 显示特定数据集信息
    python data_registry_cli.py --freshness     # 检查数据新鲜度
    python data_registry_cli.py --missing       # 列出缺失数据
    python data_registry_cli.py --update-plan   # 显示更新计划

Author: MultiFactors Team
Date: 2025-08-28
"""

import sys
import os
import argparse
from pathlib import Path

# 配置控制台编码（Windows兼容）
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() not in ['utf-8', 'utf8']:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'encoding') and sys.stderr.encoding.lower() not in ['utf-8', 'utf8']:
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_registry import get_data_registry, DataType


def print_dataset_info(dataset_info):
    """打印数据集详细信息"""
    print(f"\n=== {dataset_info.name} ===")
    print(f"描述: {dataset_info.description}")
    print(f"数据类型: {dataset_info.data_type.value}")
    print(f"更新频率: {dataset_info.update_frequency.value}")
    print(f"更新器类: {dataset_info.updater_class}")
    print(f"文件路径: {dataset_info.file_path}")
    print(f"是否可用: {'✅' if dataset_info.is_available else '❌'}")
    
    if dataset_info.is_available:
        print(f"文件大小: {dataset_info.file_size / 1024 / 1024:.2f} MB")
        if dataset_info.record_count:
            print(f"记录数: {dataset_info.record_count:,}")
        if dataset_info.last_update:
            print(f"最后更新: {dataset_info.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        if dataset_info.data_range:
            start_date, end_date = dataset_info.data_range
            print(f"数据范围: {start_date.date()} - {end_date.date()}")
    
    if dataset_info.dependencies:
        print(f"依赖项: {', '.join(dataset_info.dependencies)}")
    
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='数据注册器命令行工具')
    parser.add_argument('--summary', action='store_true', help='显示数据摘要')
    parser.add_argument('--list', action='store_true', help='列出所有数据集')
    parser.add_argument('--list-type', choices=['price', 'financial', 'classification', 'market', 'processed'], 
                       help='按数据类型列出数据集')
    parser.add_argument('--info', type=str, help='显示特定数据集信息')
    parser.add_argument('--freshness', action='store_true', help='检查数据新鲜度')
    parser.add_argument('--freshness-hours', type=int, default=24, help='新鲜度检查阈值(小时)')
    parser.add_argument('--missing', action='store_true', help='列出缺失的数据集')
    parser.add_argument('--update-plan', action='store_true', help='显示建议的更新计划')
    parser.add_argument('--available-only', action='store_true', help='只显示可用的数据集')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # 获取数据注册器
    registry = get_data_registry()
    
    try:
        if args.summary:
            registry.print_data_summary()
        
        if args.list:
            print("\n=== 所有数据集 ===")
            df = registry.list_all_datasets()
            if args.available_only:
                df = df[df['is_available'] == True]
            print(df.to_string(index=False))
        
        if args.list_type:
            data_type = DataType(args.list_type)
            datasets = registry.get_available_datasets(data_type)
            print(f"\n=== {args.list_type.upper()} 类型数据集 ===")
            for dataset in datasets:
                print_dataset_info(dataset)
        
        if args.info:
            dataset_info = registry.get_dataset_info(args.info)
            if dataset_info:
                print_dataset_info(dataset_info)
            else:
                print(f"未找到数据集: {args.info}")
                print("可用的数据集:")
                for name in registry.datasets.keys():
                    print(f"  - {name}")
        
        if args.freshness:
            freshness = registry.check_data_freshness(args.freshness_hours)
            print(f"\n=== 数据新鲜度检查 ({args.freshness_hours}小时阈值) ===")
            
            fresh_count = sum(1 for is_fresh in freshness.values() if is_fresh)
            total_count = len(freshness)
            
            print(f"新鲜数据集: {fresh_count}/{total_count}")
            print("\n详细状态:")
            
            for name, is_fresh in sorted(freshness.items()):
                dataset = registry.get_dataset_info(name)
                status_icon = "✅" if is_fresh else "⚠️"
                status_text = "新鲜" if is_fresh else "过时"
                
                last_update = "无" if not dataset or not dataset.last_update else \
                            dataset.last_update.strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"  {status_icon} {name:<25} {status_text:<4} (最后更新: {last_update})")
        
        if args.missing:
            missing = registry.get_missing_datasets()
            print(f"\n=== 缺失的数据集 ({len(missing)}个) ===")
            
            if missing:
                for name in missing:
                    dataset = registry.datasets[name]
                    print(f"  ❌ {name}")
                    print(f"     描述: {dataset.description}")
                    print(f"     路径: {dataset.file_path}")
                    print(f"     更新器: {dataset.updater_class}")
                    if dataset.dependencies:
                        print(f"     依赖: {', '.join(dataset.dependencies)}")
                    print()
            else:
                print("  🎉 所有数据集都已可用!")
        
        if args.update_plan:
            update_plan = registry.get_update_plan()
            print(f"\n=== 建议更新计划 ===")
            print("按依赖关系排序的更新顺序:")
            
            for i, name in enumerate(update_plan, 1):
                dataset = registry.datasets[name]
                status_icon = "✅" if dataset.is_available else "❌"
                print(f"  {i:2d}. {status_icon} {name:<25} ({dataset.description})")
            
            print(f"\n提示:")
            print(f"  - 可以使用以下命令进行数据更新:")
            print(f"  - python scheduled_data_updater.py --data-type <类型>")
            print(f"  - 可用类型: price, stop_price, financial, sector_changes, st, all")
    
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()