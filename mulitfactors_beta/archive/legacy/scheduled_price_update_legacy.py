#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向后兼容的价格数据定时更新脚本
此文件保持原有接口，内部调用新的 scheduled_data_updater
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduled_data_updater import ScheduledDataUpdater, main as new_main
import argparse

# 为了完全向后兼容，重新定义main函数
def main():
    """向后兼容的主函数"""
    parser = argparse.ArgumentParser(description='定时价格数据更新（向后兼容版本）')
    parser.add_argument('--force', action='store_true', help='强制更新，忽略时间检查')
    parser.add_argument('--health-check', action='store_true', help='只执行健康检查')
    parser.add_argument('--quiet', action='store_true', help='静默模式，减少输出')
    
    args = parser.parse_args()
    
    # 将参数转换为新系统的参数格式
    sys.argv = ['scheduled_data_updater.py', '--data-type', 'price']
    
    if args.force:
        sys.argv.append('--force')
    if args.health_check:
        sys.argv.append('--health-check')  
    if args.quiet:
        sys.argv.append('--quiet')
    
    # 调用新的主函数
    new_main()


if __name__ == "__main__":
    main()