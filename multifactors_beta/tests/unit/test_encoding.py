#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试编码修复效果
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

def test_console_output():
    """测试控制台输出"""
    print("=" * 60)
    print("编码测试 - 显示各种字符")
    print("=" * 60)
    
    # 测试中文显示
    print("中文测试: 这是一个中文测试字符串")
    
    # 测试ASCII状态字符（替代Unicode字符）
    print("状态显示测试:")
    print("  [OK] 操作成功")
    print("  [FAIL] 操作失败") 
    print("  [WARN] 警告信息")
    print("  [INFO] 信息提示")
    
    # 测试数据显示
    print(f"数据测试: 文件大小 123.4 MB，记录数 1,234,567 条")
    print(f"日期测试: 最新日期 2025-01-15，数据缺口 3 天")
    
    # 测试特殊字符处理
    print("特殊字符测试:")
    print("  进度: [||||||||||||||||||||] 100%")
    print("  状态: 数据已是最新，无需更新")
    print("  操作: 增量更新完成，耗时 0:02:15")
    
    print("=" * 60)
    print("编码测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_console_output()